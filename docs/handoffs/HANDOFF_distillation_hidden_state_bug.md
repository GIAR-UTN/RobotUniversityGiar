# HANDOFF — Distill policy: LSTM hidden state not reset on episode boundaries

## STATUS (follow-up session): hidden-state fix implemented, tested, and confirmed to help — clone no longer falls over, but gait is still jittery. Covariate shift (this file's own step 4) is the next suspect, not yet attempted.

Implemented exactly the fix this file recommended below:
`collect_rollout()` (`legged_gym/control/distillation.py`) now captures
`dones` from `env.step()` and calls `teacher_backend.reset()` on episode
boundaries; `bc_train()`'s recurrent branch now takes a `dones_buf` and
calls the student's masked `reset(dones=...)` at every real episode
boundary instead of only at chunk starts (approach (a) from below, not the
segment-split approach (b) — simpler given the existing per-step loop, and
`ActorCriticRecurrent.reset(dones)` already supports per-env masking).
`web_distill.py` and `tests/test_distillation.py` updated to match (3 new
tests covering the reset-on-boundary behavior; all 151 tests pass).

Re-ran the exact repro (`--teacher stable --task g1 --rollout_steps 4000
--bc_epochs 20 --num_envs 1`) as `stable_distilled_fixed`:
`final_bc_loss=0.0371` (vs `0.0425` before — a modest drop, not the 10x gap
`g1_crouch_stability` showed, plausibly because `stable`'s own single 4000-
step trajectory may rarely hit an episode boundary to begin with — this
teacher walks well and may just not fall in 4000 steps, so the bug's actual
exposure in this specific rollout could be small).

**Watched it live in the control web (not just the loss number, per this
file's own rule)**: switched to `stable_distilled_fixed` under `g1`, drove
it with simultaneous back/strafe/turn stress commands for ~20s. Result: it
never fell — orientation stayed near-upright (`gravity ≈ [~0, ~-0.05,
-0.99]`) and base height stayed in a plausible standing range (0.66–0.83m)
throughout, a real behavioral improvement over "doesn't walk well" — but
angular velocity kept oscillating ±1–1.9 rad/s rather than settling, i.e.
it's recovering/wobbling continuously rather than walking cleanly like
`stable` itself. Not a clean pass.

**Next step, per this file's own step 4 below (not yet attempted this
session)**: inject small action noise into the teacher during rollout
collection (DART) to broaden state coverage beyond the teacher's own clean
trajectory — the user explicitly deferred this to a later session rather
than continuing immediately.

---

## STATUS (original session): feature shipped and working end-to-end; distilled clones don't walk well yet — root cause diagnosed, not yet fixed

The DESTILATE feature (`rugiar distill` / control web's "⏳ Distill policy…"
panel) is fully implemented and merged — see the plan this session started
from for the full design (`legged_gym/control/distillation.py`,
`legged_gym/scripts/web_distill.py`, `TrainingManager.start_distillation()`,
CLI/UI wiring — all 148 tests pass, `SIMULATOR=genesis .venv/bin/python -m
pytest tests/ -q`). It runs end-to-end: cloning `stable` (an externally-
sourced, un-fine-tunable checkpoint — no `train_checkpoint.pt`) into
`stable_distilled` produces a real, PPO-fine-tunable checkpoint, confirmed
this session via `rugiar train --from_policy stable_distilled`.

**What's NOT done:** the distilled clones don't actually walk well when
deployed closed-loop, despite a reasonable-looking training loss. Diagnosed
below — this file exists so next session implements the fix directly
instead of re-diagnosing from scratch.

## Evidence gathered this session

Three distillation runs, same teacher (`stable`), same task (`g1`),
`--num_envs 1` (required — see "already fixed" below):

| rollout_steps | bc_epochs | final_bc_loss | walks like `stable`? |
|---|---|---|---|
| 1000 | 10 | 0.234 | no |
| 4000 | 20 (CLI defaults) | 0.0425 | **no** (user's direct report) |

For comparison, distilling `g1_crouch_stability` (a checkpoint **this repo
itself trained**, not externally-sourced) at the exact same 4000/20/1
settings reached `final_bc_loss` **0.0042** — 10x lower. That gap, plus
"loss went down 5x but still doesn't walk," is what points at something
more specific than "just needs more rollout_steps/bc_epochs."

## Root cause (researched + confirmed against the actual code this session — not yet fixed)

**Primary suspect: the teacher's LSTM hidden state is never reset on episode
boundaries during rollout collection, and the student's hidden state isn't
reset at TRUE episode boundaries during BC training either — both violate
the standard practice for recurrent BC (reset exactly on `done`).**

1. **`collect_rollout()`** (`legged_gym/control/distillation.py:124-149`):
   ```python
   obs = env.get_observations()
   for step in range(num_steps):
       actions = teacher_backend.step(obs.detach())
       ...
       obs, _, _, _, _ = env.step(actions.detach())
   ```
   `env.step()` auto-resets any env whose robot fell/episode ended
   (standard vectorized-env convention this whole codebase relies on
   elsewhere — see `legged_robot.py`) — but the `dones`/`reset_buf` return
   value (4th tuple element, thrown away above as `_`) is never looked at,
   and `teacher_backend.reset()` (which DOES exist —
   `legged_gym/control/policy.py`'s `ExplicitStatePolicy.reset()` /
   `InternalStatePolicy.reset()`) is **never called**. So right after any
   episode reset mid-rollout, the teacher keeps computing actions
   conditioned on stale LSTM hidden state from the terminated episode,
   while `obs` now reflects a freshly-reset robot — a context mismatch. The
   collected `(obs, action)` label for those steps is wrong, and BC dutifully
   trains the student to reproduce that wrongness.

2. **`bc_train()`**'s recurrent branch (`distillation.py:153-...`, the
   `for epoch in range(epochs): student.reset(...)` loop) resets the
   student's hidden state **once per epoch**, then walks the whole buffer in
   `chunk_len`-step chunks (default 25) with only a `.detach()` (not a
   reset) between chunks — already flagged in that function's own docstring
   as "v1 simplification: episode terminations *within* a chunk aren't
   specially handled." Same class of bug as #1, on the training side instead
   of the collection side.

**Why this matches the observed symptom** (per this session's community
research — see chat for the full writeup, summarized here): "reasonable
training-distribution loss, poor closed-loop walking" is the textbook
signature of exactly this kind of BC data-quality problem, compounded by the
inherent covariate-shift limitation of one-shot BC (no DAgger — the student
never acts during data collection, so it's never corrected on states it
would drift into on its own; `distillation.DISTILL_METHODS`'s `dagger` entry
is already listed as `available: False` for this reason). With `num_envs=1`
and a single deterministic teacher trajectory, there's no diversity to
average the damage away either — every reset-boundary corruption in that one
trajectory directly pollutes the dataset.

## What's already confirmed fixed/working — don't re-touch

- `--num_envs` defaulting to 1 (not 64) for distillation — REQUIRED for
  `stable`-style externally-sourced teachers (unitree_rl_gym's own exports
  bake a fixed batch=1 into the TorchScript module's hidden-state buffers;
  `--num_envs 64` crashes with `Expected hidden[0] size (1, 64, 64), got
  [1, 1, 64]`). This is unrelated to the bug above — don't conflate them.
- The whole job/RPC/CLI/UI plumbing (`start_distillation`, `rugiar distill`,
  the web panel, `finalize_policy()`'s distill branch, live-refresh via
  `refresh_local_policies` without a restart) — exercised successfully this
  session end to end, nothing to redo there.
- `check_dimensions_compatible()`'s fail-fast dimension check — working as
  designed, catches real shape mismatches (confirmed via the `--num_envs 64`
  crash above being surfaced as a clean error, not a silent garbage rollout).
- `policies/g1_crouch_stability.pt`/`.onnx` and `policies/undertrained_dummy.pt`
  were deleted this session (confirmed garbage/broken by the user), and
  `policies/stable.pt` was normalized into `policies/stable/checkpoint.pt` +
  `meta.json` (folder convention) so it's now deletable/manageable through
  the same UI/RPC path as every other local policy — unrelated cleanup, also
  don't redo.
- `HANDOFF`/skill docs: `.claude/skills/rugiar/SKILL.md` already has a full
  "Distilling policies (`rugiar distill`)" section plus 2 Troubleshooting
  entries covering the `--num_envs` crash and the "doesn't walk despite OK
  loss" symptom (pointing back at this file's diagnosis) — keep in sync if
  the fix below changes any documented behavior/defaults.

## Recommended fix (untested — implement and verify next session)

1. **`collect_rollout()`**: capture `dones` from `env.step()` (currently
   discarded as `_`) and call `teacher_backend.reset()` — but note
   `ExplicitStatePolicy.reset()`/`InternalStatePolicy.reset()`
   (`policy.py`) currently zero the **entire batch's** hidden state, not a
   per-env selective mask (unlike `ActorCriticRecurrent.reset(dones)`,
   which already supports per-env masking on the student side). At
   `num_envs=1` a full reset IS the correct per-env reset, so this is safe
   to ship as-is for the `stable` case — but if `num_envs>1` teachers are
   ever supported, `policy.py`'s backends need a `reset(dones)` masked
   variant too, or this only half-fixes the bug for that case. Track that
   as a follow-up, don't block on it now.
2. **`bc_train()`**'s recurrent branch: pass through episode boundaries
   (the `dones` `collect_rollout()` now captures) into the training loop —
   either (a) reset the student's hidden state exactly at those steps
   instead of only at chunk starts, or (b) simpler and more robust: split
   `obs_buf`/`action_buf` into per-episode segments up front and chunk
   `chunk_len` within each segment only, never crossing a real episode
   boundary. (b) is probably the smaller, safer change given the existing
   chunked-training structure.
3. After the fix, re-run the SAME comparison this session made (`--teacher
   stable --task g1 --rollout_steps 4000 --bc_epochs 20 --num_envs 1`) and
   check whether `final_bc_loss` drops meaningfully below 0.0425 AND — more
   importantly, per this whole codebase's own "don't trust the numbers"
   rule (see `.claude/skills/rugiar/SKILL.md`'s "How to know if a checkpoint
   actually walks") — **actually watch it walk** in the control web before
   declaring it fixed. A lower loss alone doesn't prove closed-loop
   behavior improved.
4. If step 3 still doesn't walk well after the hidden-state fix, the next
   suspect per this session's research is covariate shift itself — the
   cheapest mitigation without implementing full DAgger is injecting small
   action noise into the teacher during rollout collection (the DART
   technique — broadens state coverage so the student sees more
   near-recovery states than the teacher's own clean trajectory alone
   provides) before reaching for the bigger lift of an actual `dagger`
   method implementation (`distillation.DISTILL_METHODS["dagger"]` is
   already stubbed `available: False`, ready to fill in).

## Context: why this matters

The user's actual goal (stated this session): take `stable` — a policy that
walks well but can never be fine-tuned further because it has no
`train_checkpoint.pt` — and turn it into something that both walks AND can
keep training. The plumbing for that now fully works; the remaining gap is
purely "does the clone match its teacher's behavior," which is what this
bug blocks. Once fixed, the acceptance test is: `stable_distilled` (or a
freshly-named clone) should visibly walk like `stable` when switched to live
in the control web, not just report a low loss.
