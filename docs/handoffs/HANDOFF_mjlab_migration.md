> **Migration decided and underway as of 2026-08-18 — see
> `docs/mjlab_migration.md` for the live plan, phases, and progress.** This
> file stays as the decision narrative; don't duplicate updates here.

# HANDOFF — Evaluate/execute migrating motion-imitation (and maybe the whole stack) to `unitree_rl_mjlab`

## STATUS: full-body motion-imitation pipeline is wired up and working on our own Genesis/`g1_deepmimic` stack. A real, separately-trained checkpoint from a teammate (Javier) turned out to be built on a different, actively-maintained upstream (`unitree_rl_mjlab`) that our own upstream (`unitree_rl_gym`) has no equivalent for. Evidence strongly favors migrating rather than maintaining two stacks — but migration itself has NOT started. This session deliberately stopped short of building any coexistence/bridge UI. Next session: pick this up fresh and decide/execute the migration, don't build a "Movements" panel bridging both backends.

This session integrated a teammate's (Ing. Javier Villalba) motion-capture/
motion-imitation work into the repo. Along the way it surfaced a real
architectural fork in the road: **our whole `legged_gym`/Genesis/`rugiar`
stack is built on `unitree_rl_gym`, which has been unmaintained for ~1 year
and never had motion-imitation support at all — while Javier's own pipeline
(and, separately, the state-of-the-art G1 motion-tracking research
community) has moved to `unitree_rl_mjlab`.** See
`docs/motion_imitation_integration.md` for the full narrative; this file is
the action-oriented handoff for whoever picks up the migration decision.

## What got built this session — reusable, don't redo

1. **`rugiar train --motion_file`** (`web_train.py`, `training.py`,
   `rugiar` CLI) — pick which `.pkl` under `resources/reference_motion/` to
   train `g1_deepmimic` against, instead of the config's hardcoded default.
2. **`category`** — optional, purely-cosmetic metadata field on policies
   (`meta.json` → `TrainingManager.register_source()`/
   `discover_local_policies()` → `catalog()` → Fuse/Clone-from panels in
   `web/app.js`). Distinguishes an imported/external source from one this
   repo trained itself, without touching the `task` compatibility gate.
3. **Reference-motion "ghost" overlay, live in the actual web control UI**
   (`legged_gym/utils/viser_viewer.py::update_reference_motion()`) — red
   point-cloud at the target key-body positions, updated every tick
   alongside the real robot mesh. The data was already computed by
   `g1_deepmimic.py` every tick; it just never reached viser before (only
   Genesis's native debug-vis, which this stack doesn't actually use — see
   "Genesis's native viewer has a rendering bug on Mac" in
   `.claude/skills/rugiar/SKILL.md`).
4. **A real dance clip converted end-to-end**: `exptech/g1-moves`'s
   `B_DadDance` (Hugging Face, CC-BY-4.0) → `resources/reference_motion/
   unitree_g1/{raw,genesis}_run/g1moves_B_DadDance*.pkl`, via the
   pre-existing (never previously exercised end-to-end)
   `legged_gym/scripts/process_reference_motion.py`. g1-moves' 29-DOF joint
   order and quaternion convention turned out to be IDENTICAL to
   `G1Flat29DofCommonCfg.dof_names` — zero reordering needed, a genuinely
   useful confirmed fact for evaluating any other retargeted-G1 dataset.
5. **Six bugs fixed**, all pre-existing, all surfaced by running
   `g1_deepmimic` end-to-end for the first time (nobody had before — zero
   trained policies existed for this task before this session):
   - `web_train.py`: unresolved `train_cfg.runner.load_run` sentinel
     crashing export for any non-recurrent-actor task.
   - `policy.py::load_policy_backend()`: misclassified a stateless
     (non-recurrent) jit export as the recurrent "internal state"
     convention — fixed via `forward()` schema arg-count detection, added
     `StatelessPolicy`.
   - `g1_deepmimic.py::_init_buffers()`: passed `self.dt` where
     `num_key_bodies` belonged in the `MotionLoader` constructor.
   - `process_reference_motion.py`: crashed under `--headless` on an
     unguarded debug-vis call.
   - `discover_local_policies()`: only recognized `checkpoint.pt`, not
     `checkpoint.onnx` — needed for external ONNX imports (below).
6. **Two of Javier's real trained checkpoints imported**
   (`policies/javier_mjlab_model_7000/`, `policies/javier_mjlab_
   dance1_subject2/`) — pulled via `kaggle kernels output jvillalba007/
   unitree-rl-mimic` (credentials already on this machine). Both are
   **honestly marked incompatible-as-is** and registered under an
   intentionally-unregistered task name (`g1_mjlab_mimic_unregistered`) so
   no driver auto-loads them into a dimension mismatch. `note` field in
   each `meta.json` has the full story. `dance1_subject2` turned out to be
   `unitree_rl_mjlab`'s own bundled tutorial example motion, not something
   Javier retargeted himself — don't credit it to him in any future write-up.

All pushed to branch `g1-fullbody-motion-imitation` (PR #3 against
`GIAR-UTN/RobotUniversityGiar`), two commits. Live-tested via
`rugiar_driver.py --task g1_deepmimic --headless` (boots, loads, switches
between two policies live) and via a real (non-headless) session on
`:9017`/`:9006` with the ghost overlay visibly rendering.

## The actual finding: this isn't a preference question, it's a maintenance-and-capability gap

Checked directly against both repos' own activity, not just chat hearsay:

| | `unitree_rl_gym` (this fork's upstream) | `unitree_rl_mjlab` |
|---|---|---|
| Last push | 2025-07-25 (~1 year stale) | 2026-04-13 (active) |
| Open issues | 59 | 42 |
| Motion imitation | **None** — `legged_gym/envs/` only ever had base/g1/go2/h1/h1_2 | **First-class** — CSV→NPZ motion import, dedicated `*-Tracking-*` tasks, BeyondMimic integration docs |

`g1_deepmimic`/`MotionLoader` in this repo were hand-built from scratch by
a teammate because `unitree_rl_gym` had nothing to build on. Javier's choice
of `mjlab` was not environment preference — it's the actively-maintained
project that already solves what our upstream never did.

**Broader community check (not just Unitree's own roadmap):** the current
state-of-the-art G1 motion-tracking research — BeyondMimic, SONIC,
ResMimic, all validated on real G1 hardware in 2026 — is built on Isaac
Lab's manager-based API. `mjlab` is explicitly designed to mirror that same
API with MuJoCo instead of Isaac Sim as the physics backend. Javier's stack
is aligned with where the field moved, not an outlier choice.

**Hard incompatibility confirmed, not just theorized:** both imported
checkpoints expose an onnx `obs` input of `[1, 154]` (mjlab's own
single-frame convention). Our `g1_deepmimic` task's own
`num_observations` is `1380` (`frame_stack=5 × (151 proprioceptive + 125
ref-motion features)`, see `g1_deepmimic_config.py`). This is a dimension
mismatch, not a semantic one — `rugiar distill`'s own
`check_dimensions_compatible()` would refuse this teacher outright.

## Second opinion sought this session (Opus), and the user's decision

Given a plan under discussion ("keep both stacks alive side by side, add a
small 'Movements' panel to the web UI that surfaces motion data from
either, converge properly at some later point"), an Opus-model agent was
consulted directly for a critique before any of that got built. Verbatim
gist, worth re-reading in full if this handoff gets picked up later:

> Coexistence as a *state* is fine — don't rewrite a working stack on a
> hunch. The trap is the **bridge UI**: a panel that assumes two backends
> encodes the split into the product surface and makes it more expensive to
> remove every week it exists. The 154 vs 1380 obs mismatch is a hard
> wall — no UI abstraction hides it, you'd just be shipping a mode switch.
> The evidence given isn't "two viable options" — dead upstream (1yr, 59
> issues, never had mimic) vs. an actively maintained, purpose-built
> target that the wider research community (BeyondMimic/SONIC/ResMimic)
> also converged on — that's a migration that hasn't been scheduled yet,
> not an open question.
>
> If Javier's checkpoints need to be usable at all before a full decision,
> the one piece of integration work that survives migration in EITHER
> direction is making them runnable end-to-end as their own separate
> backend/runner process (own obs builder, own adapter) — NOT a shim
> inside `g1_deepmimic`, with only the WebSocket control protocol shared.
> Only after that would a minimal UI (policy-source label + motion list, no
> dual-backend generality) be worth adding.
>
> Migration triggers to write down NOW, not decide later in the moment:
> a second mjlab-trained policy outperforms our Genesis equivalent →
> migrate. Anyone spends >1 day fixing the same thing twice, once per
> stack → migrate. Any BeyondMimic-lineage feature (retargeting, tracking
> rewards, motion library) is needed → migrate, don't reimplement in
> `g1_deepmimic`. 90 days elapse with both alive → forced explicit
> keep/kill decision, no silent default.

**User's decision this session, given that critique: no coexistence UI.**
Write this handoff instead, and tackle the actual migration decision head-on
in a fresh session — not incrementally, not by bolting a panel onto the
current stack. The "Movements" panel idea (list/select reference-motion
clips in the web UI, live-switchable like Policies) is **on hold, not
cancelled** — revisit it only for clips within our OWN pipeline (e.g. more
g1-moves clips already in our 1380-dim convention), never as a mechanism to
paper over the mjlab obs mismatch.

## Recommended plan for next session

1. **Read `docs/motion_imitation_integration.md` in full first** — it has
   the complete narrative (Javier's chat history findings, the g1-moves
   dataset compatibility analysis, the Hugging Face landscape survey of
   other G1 motion datasets) that this file deliberately doesn't repeat.
2. **Decide, don't drift**: given the evidence above, the default framing
   should be "when do we migrate," not "should we." If there's a real
   reason to stay on Genesis/`unitree_rl_gym` found next session that isn't
   visible from here (e.g. Genesis-specific deployment work already
   committed elsewhere, team bandwidth), write that reason down explicitly
   — don't let the decision lapse into permanent coexistence by default
   (see Opus's "90 days" trigger above).
3. **If migrating**: this is a real architecture project, not a config
   change — different physics backend (MuJoCo vs Genesis), different
   task/obs API (mjlab's manager-based config vs this repo's
   `LeggedRobotCfg` class-inheritance style), different checkpoint/export
   convention. Scope it as its own multi-session effort; don't try to
   land it in one sitting. `mjlab`'s own docs
   (https://mujocolab.github.io/mjlab/index.html) and
   `unitree_rl_mjlab`'s README are the starting references — both already
   fetched/read this session, see chat history if this repo's memory
   system doesn't have them by the time this is picked up.
4. **If NOT migrating (yet)**: per Opus, the only integration work worth
   doing with Javier's existing checkpoints is a standalone runner (own
   process, own obs pipeline, shares only the WebSocket control protocol)
   — not a panel, not a shim in `g1_deepmimic`. Don't reach for the
   "Movements panel" idea as a substitute for this decision.

## Context: why this matters

User's goal across this whole integration effort: bring Javier's
motion-capture/motion-imitation work into this repo so he can keep working
from here instead of his private setup — see `docs/
motion_imitation_integration.md`'s opening. That goal is now blocked on an
architecture decision bigger than motion-imitation alone (which simulator/
framework the team standardizes on), which is exactly why this session
stopped short of shipping UI for it and wrote this handoff instead of
guessing.
