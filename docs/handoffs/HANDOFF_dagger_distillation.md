# HANDOFF — Implement `dagger` distillation (one-shot BC has a confirmed structural ceiling)

## STATUS: one-shot BC diagnosed, instrumented, and pushed as far as it goes — still doesn't walk closed-loop. Root cause confirmed against community practice. `dagger` is the next real lever, not implemented yet.

This session picked up from `HANDOFF_distillation_hidden_state_bug.md` (that
bug — teacher/student LSTM hidden state not reset on episode boundaries — is
fixed and confirmed, don't re-touch it). Goal this session: figure out why
`stable_distilled`/`stable_distilled_fixed` still don't walk like `stable`
despite a "reasonable" `final_bc_loss`, and fix it. **Verdict: fixed
everything fixable within one-shot behavior cloning; the clone still drifts
sideways and falls. This is `behavior_cloning`'s own structural ceiling
(covariate shift), not a bug — confirmed against external sources (see
bottom). `dagger` is the documented way past it, and is the next thing to
implement.**

## What got built this session — reusable, don't redo

### 1. Diagnostic-only motor-behavior metrics (`legged_gym/envs/base/legged_robot.py`)

`episode_sums` now also tracks `actual_lin_vel_y`, `actual_ang_vel_yaw`,
`actual_base_height`, `actual_torque_abs_mean` — same pattern as the
pre-existing `actual_lin_vel_x` (dt-scaled, never added to `rew_buf`, free
logging via the `"rew_"`-prefixed `extras["episode"]` plumbing). Read
straight off `self.simulator.*`, already-computed every step for the reward
functions — zero training-time cost.

### 2. Rollout-coverage diagnostics for distillation (`legged_gym/control/distillation.py`)

`collect_rollout()` now returns a 4th value, `ground_truth`: a dict of
`{"commands", "base_lin_vel", "base_ang_vel"}`, each `(steps, envs, 3)`,
recorded straight from `env.commands`/`env.simulator.*` every step.
`summarize_rollout(ground_truth, action_buf)` turns that into per-field
`{mean, std, min, max}` stats — printed by `web_distill.py`/`rugiar distill`,
and threaded through `result.json` → `TrainingJob.rollout_diagnostics` →
`meta.json["distillation"]["rollout_diagnostics"]` (see `training.py`).

**Important: this does NOT decode `obs_buf` by column slice — do not add
that back.** First version did, and it was wrong: `g1`'s own
`G1Robot.compute_observations()` (`legged_gym/envs/g1/g1.py:57-69`)
overrides the base `LeggedRobot.compute_observations()` with a completely
different layout (no `base_lin_vel` in the visible obs at all; order is
`ang_vel[0:3], gravity[3:6], commands[6:9], dof_pos[9:21], dof_vel[21:33],
actions[33:45], sin_phase[45], cos_phase[46]`, `num_obs=47` not the base
task's 48). A fixed-slice decode silently read `dof_pos` deltas and reported
them as "commands" — plausible-looking, completely wrong numbers, presented
confidently for a full turn of this conversation before being caught. Any
other task (`tron1sf`, `k1`, `go2*`, etc.) likely has yet another layout —
see `grep -rn "def compute_observations" legged_gym/envs/`. Reading
simulator state directly (what `ground_truth` does) is layout-independent by
construction; keep it that way.

### 3. Command-range bias infrastructure for rollout collection

- `legged_gym/scripts/collect_rollout_variant.py` — standalone worker,
  collects ONE rollout under an optional `--cmd_vx_range/--cmd_vy_range/
  --cmd_yaw_range` override (same `env_cfg.commands.ranges.*` mechanism
  `rugiar train` already uses), dumps `{obs, action, dones, ground_truth,
  coverage}` to a `.pt` file. Two non-obvious cfg gotchas it fixes, both
  worth remembering for any future rollout-collection code:
  - `cfg.commands.zero_cmd_prob` defaults to **0.4** — on every resample,
    40% chance the command is forced to zero **regardless of the configured
    range**, "to encourage standing still." Must pass `--zero_cmd_prob 0` to
    make a deliberately-biased corner actually hold.
  - `cfg.commands.heading_command` defaults to **True** — when on, the yaw
    command (`commands[:,2]`) is NOT held at its resampled value; it's
    recomputed every step as proportional control toward a target heading,
    self-correcting toward 0 once reached. `--cmd_yaw_range` only bounds the
    *clip* on that computed value, it doesn't force a sustained turn. Must
    pass `--heading_command off` for a real held turn command.
  - `cfg.commands.resampling_time = 10s` (500 control steps @ 50Hz) means
    the periodic resample **never fires within a shorter rollout** unless
    the robot happens to fall and get reset first — commands silently stay
    at their post-construction all-zero state the whole time otherwise
    (confirmed empirically: a 200-step smoke test reported EXACTLY zero
    commanded velocity for every variant, no exceptions). The script now
    calls `env._resample_commands(torch.arange(env.num_envs, device=...))`
    immediately after `make_env()` to force the bias to take effect from
    step 0 instead of maybe never.
- `legged_gym/scripts/distill_multi_variant.py` — orchestrator. Launches 7
  variants (`forward`, `backward`, `turn_left`, `turn_right`, `strafe_left`,
  `strafe_right`, `full_range` as an unbiased control group) as parallel OS
  subprocesses (cheap — CPU-only, single-env each, this machine has 10
  cores), concatenates every variant's rollout into one dataset (`torch.cat`
  along the env axis — same shape `bc_train()` already expects, no changes
  needed there), runs one `bc_train()` pass over the union, and registers
  the result via `TrainingManager.finalize_policy()` exactly like
  `rugiar distill` does (fine-tunable via `--from_policy`, `meta.json`
  intact, shows up in the control web after Refresh).

**Confirmed working end-to-end**: `stable_distilled_multivariant`
(`--rollout_steps 4000 --bc_epochs 20`, all 7 variants) — merged coverage
`vx∈[-0.99,0.99] vy∈[-0.99,0.96] yaw∈[-1.0,1.0]` (vs. the single-rollout
baseline's `vx∈[-0.59,0.41]`, heavily negative-skewed), `final_bc_loss
=0.0247` (lower than `stable_distilled_fixed`'s 0.037). **Still doesn't walk
well** — user's direct report: "camina para el costado, se cae, hace algunos
movimientos coherentes, pero claramente no puede caminar." Broader command
coverage and a lower loss did NOT translate into a working closed-loop
walker. This is the key negative result motivating the root-cause section
below — don't read it as "the multi-variant infra failed," read it as "we
now have good evidence one-shot BC itself is the ceiling, not data
coverage."

## Root cause (confirmed against external sources this session, not just this repo's own code)

**One-shot behavior cloning (`distillation.DISTILL_METHODS["behavior_cloning"]`,
the only implemented method) has a hard, well-documented ceiling: covariate
shift / compounding error.** The student is trained only on states the
TEACHER visited. The moment the student's own imperfect actions push it even
slightly off that trajectory, it's in a state distribution it never trained
on — no correction signal exists there, so the error compounds roughly
`O(T²ε)` over an episode instead of just `O(Tε)`. This matches the reported
symptom exactly: "some coherent movement" (near the teacher's trajectory,
early on) degrading into "walks sideways, falls" (drifted off-distribution).

External confirmation gathered this session (see chat for full write-up):

- A real Isaac Lab G1 distillation pipeline (PMT — teacher → distill →
  finetune) uses **~4000 *iterations* of policy distillation** — i.e. an
  iterative retrain loop, not a single rollout + fixed-epoch BC pass like
  this repo's current `rugiar distill`. Confirms our whole approach has been
  operating roughly 2-3 orders of magnitude below what a working reference
  pipeline actually does.
- Imitation-learning literature is unanimous that one-shot BC's fix is
  **DAgger** (Dataset Aggregation): roll the CURRENT STUDENT out closed-loop,
  have the TEACHER relabel the actual states the student visited, aggregate
  those (state, expert-action) pairs into the training set, retrain, repeat.
  This is precisely what turns "teacher's clean trajectory only" into
  "student's own mistake-recovery states, correctly labeled" — closing the
  gap covariate shift opens.
- `distillation.DISTILL_METHODS["dagger"]` is already stubbed
  `available: False` in this codebase — this was anticipated, just never
  built.

## Recommended plan for next session: implement `dagger`

**Core loop** (add as a new function in `legged_gym/control/distillation.py`,
alongside `collect_rollout`/`bc_train`, e.g. `dagger_train()`):

1. Build the student once (`build_student()`, as today).
2. For `N` rounds:
   a. **Roll the env out under a MIX of student and teacher actions**
      (classic DAgger uses a decaying `beta` — round 0 is ~100% teacher so
      the very first rollout isn't a totally untrained student immediately
      face-planting and only ever visiting "on the ground" states; `beta`
      decays toward 0 over rounds so later rounds are dominated by the
      student's own actions/mistakes). Record the full obs sequence visited.
   b. **Relabel**: replay that SAME obs sequence through the teacher
      (`teacher_backend.step(obs_t)` in order, t=1..T) to get what the
      teacher would have done from each of those states. This must preserve
      the teacher's own hidden-state continuity across the sequence — don't
      query isolated states out of order, a recurrent teacher's action
      depends on its running hidden state just like the student's does (this
      is exactly the class of bug `HANDOFF_distillation_hidden_state_bug.md`
      already fixed once for the collection path — watch for the same
      failure mode here).
   c. **Aggregate**: append `(obs, teacher_action)` from this round into a
      growing dataset (vanilla DAgger keeps everything from every round, not
      just the latest — the whole point is a training set that covers both
      old and newly-discovered states).
   d. **Retrain** the student via `bc_train()` (already exists, reusable
      as-is) on the aggregated dataset — either from scratch each round or
      continued from the previous round's weights (continued is standard
      and much cheaper; confirm this doesn't destabilize recurrent training
      given `bc_train()`'s existing chunked-BPTT approach).
3. Wire up: flip `DISTILL_METHODS["dagger"]["available"]` to `True`,
   dispatch `--method dagger` in `web_distill.py`/`start_distillation()` to
   this new function instead of the current `collect_rollout()`+`bc_train()`
   pair (CLI/UI plumbing for `--method` already exists end-to-end, it's just
   gated on `available` today).

**Reusable from this session, with one structural caveat**: the
command-range-bias infra (`collect_rollout_variant.py`'s `--cmd_*_range
--zero_cmd_prob --heading_command` overrides, and the resample-timing/
zero-cmd-prob gotchas above) is directly applicable to DAgger's rollout step
too — but `collect_rollout()` as it exists today steps the env with the
TEACHER's actions only. DAgger's rollout step needs the env stepped by the
STUDENT (or a student/teacher mix) instead, which is a different function,
not a trivial extension of the existing one. Don't try to bolt this onto
`collect_rollout()`'s existing signature — write it as its own loop, or add
an `actor_backend` parameter that `dagger_train()` can vary round-to-round.

**Cost/scheduling note**: this will NOT fit in the "~60-70s, safe to just
run and wait" budget one-shot distillation had — multiple rollout+relabel+
retrain rounds means meaningfully more wall-clock. If any single round
individually stays under ~20-25 minutes this should still be fine to run to
completion in one shot depending on how many rounds/round-size end up
needed; if a full multi-round run threatens to run long, apply the same
chunking discipline `.claude/skills/rugiar/SKILL.md`'s "Running a long
training job unattended" section already documents for `rugiar train`
(background-bash processes get killed after ~28-30 min regardless of what's
running).

**Acceptance test — same discipline as always, don't skip it**: a lower
`final_bc_loss` (or good `rollout_diagnostics` coverage) is NOT sufficient
evidence of success on its own — this whole conversation's lesson, restated.
Watch the result walk closed-loop (`play.py --viewer=viser` or load into
`rugiar_driver.py` and drive it) before declaring `dagger` fixed the
problem. If it still doesn't walk well after a real DAgger implementation,
the next suspect is `stable`'s own obs-convention mismatch (flagged earlier
this conversation, not yet ruled out) — but don't reach for that until
DAgger itself has had a real trial, since one-shot BC's covariate shift
alone is sufficient to explain everything observed so far.

## Context: why this matters

User's goal (stated across this whole conversation): turn `stable` — an
externally-sourced policy that walks well but has no `train_checkpoint.pt`,
so it can never be fine-tuned further — into something that both walks like
`stable` AND can keep training via `rugiar train --from_policy`. The
plumbing for that (distill → register → fine-tune) has worked end-to-end
since the previous handoff. The remaining gap is purely behavioral fidelity,
and this session's work narrowed that gap from "unknown cause" to "a named,
externally-validated limitation of the currently-implemented method, with a
concrete next implementation." The user explicitly asked to pause further
one-shot-BC experimentation until this handoff exists — don't restart
tuning `--rollout_steps`/variants/etc. next session without first attempting
`dagger`, that door is now understood to be a dead end on its own.
