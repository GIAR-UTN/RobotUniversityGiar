# Migrating to mjlab — plan and progress

Successor to `HANDOFF_mjlab_migration.md` (repo root), which has the
decision history and narrative. This doc is the live migration plan and
technical reference. Written 2026-08-17/18, phased execution starts here.

## §0 — Scope correction: the target is `mjlab`, not `unitree_rl_mjlab`

The handoff framed the target as `unitreerobotics/unitree_rl_mjlab` (the
repo Javier's Kaggle checkpoints were trained against). Reading both repos
closely changes that: **the target is `mjlab` itself** (PyPI `mjlab==1.6.0`),
with `unitree_rl_mjlab` demoted to "reference to selectively vendor from".
Evidence:

1. `unitree_rl_mjlab`'s tracking task is a near-verbatim copy of mjlab's
   own. `diff` of `unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py`
   vs `mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py` is three hunks:
   an import rewrite, `JointPositionActionCfg.scale` `0.5 → 0.25`, and
   `base_com` x-range `±0.025 → ±0.05`. `mdp/observations.py` and
   `mdp/terminations.py` are byte-identical.
2. mjlab already registers the exact task Javier trained:
   `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation`
   (`mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py`), with
   `experiment_name="g1_tracking"` — matching Javier's checkpoint path
   `logs/rsl_rl/g1_tracking/2026-08-13_06-08-32/policy.onnx` exactly.
3. mjlab's own `train`/`play` already support a local `--motion-file`.
   Unitree's fork adds nothing there, and its `play.py:78` references
   `cfg.registry_name`, a field its own `PlayConfig` dataclass never
   declares — an `AttributeError` on that code path.
4. `unitree_rl_mjlab` is not installable as a library: no `pyproject.toml`,
   `setup.py` declares `packages=["src"]` (a top-level module literally
   named `src`), and pins `mjlab==1.2.0`/`mujoco-warp==3.5.0` (current:
   1.6.0/3.11.0).

What it's still worth vendoring three files from — see
`third_party/unitree_rl_mjlab/README.md` for what and why.

## §1 — Headline result (proven before any repo code was written)

Both of Javier Villalba's ONNX checkpoints ran closed-loop, 400 steps, in
stock mjlab on this machine (Apple M1 Pro, CPU-only):

```
javier_mjlab_dance1_subject2: 400 steps, terminations=0, min_root_h=0.676, mean_body_pos_err=0.0705 m
javier_mjlab_model_7000:      400 steps, terminations=6, min_root_h=0.752, mean_body_pos_err=0.0679 m
```

`dance1_subject2` didn't fall once in 8 simulated seconds, tracking to
~7cm mean body-position error. `model_7000` terminated 6 times — almost
certainly because it was trained against a different, unknown motion (see
R3 below). Use `dance1_subject2` as the reference policy, not `model_7000`.

## §2 — The 154-dim actor observation, fully specified

Verified empirically (`ObservationManager.group_obs_term_dim` printed on
this machine) and cross-checked against
`third_party/unitree_rl_mjlab/State_Mimic.cpp.reference` +
`g1_mimic_deploy.yaml.reference`:

| Slice | Term | Dims | Definition |
|---|---|---|---|
| `[0:58]` | `command` | 58 | ref `joint_pos[t]` (29) + ref `joint_vel[t]` (29) |
| `[58:64]` | `motion_anchor_ori_b` | 6 | first 2 columns of the rotation matrix from the robot's anchor frame (`torso_link`) to the reference anchor frame |
| `[64:67]` | `base_ang_vel` | 3 | IMU angular velocity sensor |
| `[67:96]` | `joint_pos` | 29 | `joint_pos_biased - default_joint_pos` |
| `[96:125]` | `joint_vel` | 29 | `joint_vel - default_joint_vel` |
| `[125:154]` | `actions` | 29 | last raw action |

Order = Python dict insertion order in mjlab's
`make_tracking_env_cfg()`. Critic group is 286-dim (adds
`motion_anchor_pos_b`, `body_pos`, `body_ori`, `base_lin_vel`).
`tests/test_mjlab_env_smoke.py` asserts this table stays true across mjlab
version bumps.

## §3 — What survives unchanged vs. what's rewritten

See the full migration-plan writeup (this session's architecture agent) for
the complete subsystem-by-subsystem mapping. Short version:

**Survives:** the WebSocket control protocol (`legged_gym/control/transport.py`),
the whole `legged_gym/control/` engine (service/supervisor/safety/selector —
already backend-agnostic per `adapter.py`'s own docstring), `policy.py`'s
ONNX loading (already handles Javier's `[1,154]→[1,29]` format with zero
changes), the `policies/<name>/` catalog convention, the `rugiar` CLI argv
surface, the web UI's Family-panel extension point (mjlab becomes a third
family, not a bridge panel), Kaggle as the training backend pattern.

**Rewritten:** task config (mjlab manager-dict style replaces
`LeggedRobotCfg` inheritance), physics stepping (~4.7k lines of
`legged_gym/simulator/*.py` become a dependency), motion data format (`.pkl`
xyzw → `.npz` wxyz, converter ported from
`third_party/unitree_rl_mjlab/csv_to_npz.py.reference`), training entrypoint.

## §4 — Phases

0. **Env + fixture + smoke test.** ✅ done — `.venv-mjlab/`, `mjlab==1.6.0`,
   `resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz`,
   `tests/test_mjlab_env_smoke.py` (154-dim actor obs confirmed on this
   machine).
1. **Motion converter: our `.pkl` → mjlab `.npz`.** ✅ done —
   `legged_gym/scripts/process_reference_motion_mjlab.py`, ported from
   `third_party/unitree_rl_mjlab/csv_to_npz.py.reference` (same
   MotionLoader/run_sim math, xyzw→wxyz on load instead of on a CSV
   column). Converted `g1moves_B_DadDance.pkl` (2509 frames @ 60fps input
   → 2090 @ 50fps output) end-to-end: schema matches the Phase 0 fixture,
   the converter's own `torch.testing.assert_close` sanity checks passed,
   and the output loads into a real `Mjlab-Tracking-*` env and steps
   without falling. See `tests/test_process_reference_motion_mjlab.py`.
2. **G1 robot config parity check.** ✅ done (docs-only, no port needed —
   mjlab ships G1 29-DoF natively). See §6 below for the diff.
3. **Register `Rugiar-G1-Mimic`, our own mjlab tracking task.** ✅ done —
   `mjlab_tasks/` package at repo root (named that, not `src`, avoiding
   unitree_rl_mjlab's own mistake, §0). Calls mjlab's own
   `unitree_g1_flat_tracking_env_cfg()` unmodified (not a fork), so its
   observation contract is bit-for-bit what Javier's checkpoints were
   trained against — only `task_id`/`experiment_name` are ours. Surfaced
   a real path-ordering bug while wiring this up: see R1's update.
   `tests/test_mjlab_tasks_registration.py` confirms the registered task's
   154-dim actor obs and term order match the stock task exactly.
4. **Flip Javier's checkpoints to the registered task; closed-loop
   validation.** ✅ done — **this is the milestone**. Both
   `policies/javier_mjlab_*/meta.json` now say `"task":
   "Rugiar-G1-Mimic"` (were `"g1_mjlab_mimic_unregistered"`, marked
   "INCOMPATIBLE AS-IS"). `tests/test_javier_checkpoints_track.py` drives
   each ONNX checkpoint directly (onnxruntime, no external runner) for
   400 steps against the `dance1_subject2` reference motion:
   `dance1_subject2` — 0 falls, ~10cm mean body-pos error;
   `model_7000` — 6 falls (unknown training motion, see R3), ~18cm.
   Both within the test's margin-padded thresholds. This is the same
   result the architecture agent measured outside the repo in §1, now
   reproduced *inside* the repo, against a properly registered task,
   as a repeatable test.
5. **Wire into `rugiar_driver_mjlab.py` / `MjlabAdapter` / web UI (third
   family, existing extension point — no new panel).** ✅ done —
   `Rugiar-G1-Mimic` is now drivable from the same control web
   (`web/index.html`) as every Genesis task, over the same WebSocket
   protocol, with zero new UI panels.

   Built: `legged_gym/control/mjlab_adapter.py` (`MjlabAdapter`,
   `backend_name="mjlab"`, `capabilities={"restart": True}`, mapping
   `robot.data.joint_pos/joint_vel/default_joint_pos/root_link_quat_w/
   root_link_ang_vel_b/root_link_lin_vel_b/projected_gravity_b/
   root_link_pos_w[:,2]` into `RobotState`) and
   `legged_gym/scripts/rugiar_driver_mjlab.py` (third sibling driver:
   env + adapter + viewer bridge only — `ControlService`/`ControlServer`/
   `PolicySupervisor`/`SafetyGovernor`/`load_policy` are reused as-is).
   `tests/test_mjlab_adapter_driver.py` covers both.

   Three things the plan assumed and reading corrected:
   - **`ViserPlayViewer` was the right guess** (`mjlab/viewer/viser/viewer.py`),
     but its `run()` *is* the sim loop — it calls `policy(obs)` then
     `env.step()` itself. So there is no `while True` in this driver: the
     per-tick control work (`drain_commands` → `service.tick` →
     `publish_status`) lives in the policy callback, which the viewer calls
     on its own main thread — ARCHITECTURE.md's sim-thread invariant holds
     unchanged. It accepts an external `viser_server`, which is how
     `--viser_port` is honored. The reference-motion ghost is mjlab's own
     (`MotionCommandCfg.debug_vis` + `MjlabViserScene`); nothing was
     reimplemented.
   - **The control engine had exactly one Genesis coupling**, and it wasn't
     obvious from the docs: `adapter.py` imported
     `legged_gym.utils.math_utils`, which executes `legged_gym/utils/__init__.py`
     → `helpers.py` → this repo's *vendored* `rsl_rl` (`ActorCriticTSDepth`),
     which doesn't exist under `.venv-mjlab`. Now imported lazily inside the
     one method that uses it. Plus `legged_gym/__init__.py` accepts
     `SIMULATOR=mjlab` (imports no simulator at all) so `legged_gym.control`
     is importable from the mjlab venv.
   - **Family routing needed `ControlService` to widen its task list.**
     `list_families()` enumerated `task_registry.task_classes` only, which
     can never contain an mjlab task id. It now returns the union of
     registered legged_gym tasks and tasks any local `policies/<name>/meta.json`
     declares (`_switchable_families()`), so `Rugiar-G1-Mimic` shows up as a
     third family from a Genesis session — and, symmetrically, the Genesis
     families show up from an mjlab one (where `task_registry` isn't
     importable at all and the import is caught). `_script_for_task()` in
     both Genesis drivers routes an unregistered (⇒ mjlab) task to
     `rugiar_driver_mjlab.py`, and `_relaunch_for_family()` now picks the
     **interpreter** too (`.venv-mjlab/bin/python` + `SIMULATOR=mjlab` out;
     `.venv/bin/python` + `SIMULATOR=genesis` back), returning instead of
     exiting when the target venv is missing.

   **R9 resolved:** `applyStatus()` did *not* gate the Command HUD
   generically. It now hides the Command and Stress-Stimuli panels when
   `status` omits `command`/`random_events` — the same "absent key means
   this adapter doesn't support it" rule already used for
   `episode_timeout_s`/`operator_speed_limit` — and `sendCruiseCommand()`
   no-ops, so W/A/S/D can't fire at a backend with no velocity command
   either. Backend-name-agnostic: any future commandless adapter gets it
   free. No new panel, no bridge UI.

   **R10 resolved:** `tests/test_driver_family_parity.py` carries a written
   `MJLAB_EXEMPT_DRIVER` exemption plus a test that the exemption is still
   load-bearing (fails if the mjlab driver's `_script_for_task` /
   `_relaunch_for_family` ever become identical to the Genesis ones).

   Verified on this machine (Apple M1 Pro, CPU-only), real output:
   - `CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python legged_gym/scripts/rugiar_driver_mjlab.py --task Rugiar-G1-Mimic --headless`
     → `obs=154 actions=29`, both Javier checkpoints loaded, live switch at
     step 40 (`active=javier_mjlab_dance1_subject2` → `javier_mjlab_model_7000`),
     `Headless smoke test done.`
   - Real session on ports 9012/9013 (`las ports free` — 9006/9017 were in
     use by a running Genesis session and deliberately left alone): a
     scripted WebSocket client got `backend=mjlab`,
     `current_task=Rugiar-G1-Mimic`, `capabilities={'restart': True}`, live
     telemetry (`base_height=0.688`, `projected_gravity=[0.099, 0.084, -0.992]`),
     `list_families` listing `Rugiar-G1-Mimic` with both policies, a
     confirmed live `request_switch`, working `pause`/`resume`/`restart`, and
     `set_command` correctly answered
     `NotImplementedError: MjlabAdapter does not support set_command`.
   - `examples/joystick_controller.py ws://localhost:9012 --demo` connected
     with **zero edits** and read live status; its velocity commands get that
     same clean per-call error (there is no velocity command on a tracking
     task) without dropping the connection.
   - `SIMULATOR=genesis .venv/bin/python -m pytest tests/ -q` → **188 passed,
     5 skipped** (baseline before this phase: 187 passed, 4 skipped — +1 new
     parity test, +1 file skipped for having no mjlab).
   - `SIMULATOR=mjlab CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest tests/ -q`
     → **201 passed**. Note the `SIMULATOR=mjlab` prefix: it is now the
     canonical way to run the FULL suite in that venv (without it, three
     pre-existing test files fail collection on `legged_gym`'s
     SIMULATOR check — that was already true before this phase).

   Venv delta: `.venv-mjlab` gained `fastapi` + `uvicorn` (+`starlette`,
   `anyio`, `h11`, `annotated-doc`) — `ControlServer`'s transport, unchanged.

   Known limitations, deliberately not fixed here: the Create-Policy /
   Fuse / Distill panels don't work against an mjlab session (they shell out
   to Genesis `web_train.py`); mjlab training is Phase 6. A family switch
   *out of* mjlab always routes through `rugiar_driver.py`, which
   re-dispatches to the target-aware driver if needed (one extra ~15s relaunch in
   that one case — this process can't import `task_registry` to tell).
6. Training path (Kaggle), first self-trained mjlab policy, and the
   deprecation of `g1_deepmimic`/Genesis for this family (walking tasks
   stay on Genesis — no trigger to move them).

## §5 — Risks (R1–R10)

**R1 — `rsl_rl` name collision (verified, hard).** This repo vendors a
top-level `rsl_rl/` package. Running `.venv-mjlab/bin/python` from the repo
root (e.g. `python -c "..."`) resolves `import rsl_rl` to the **vendored**
copy, not PyPI `rsl-rl-lib==5.4.2` — confirmed on this machine. Secondary:
`genesis` extra pulls `mujoco==3.10.0`; mjlab needs `~=3.11.0`. Both are why
this is a fully separate venv, not an extra in the main one.

Fix, refined during Phase 3: **`-I` (isolated mode) alone is only correct
for scripts that don't import this repo's own `mjlab_tasks/` package.**
`-I` strips the cwd/repo-root entry from `sys.path` entirely — that fixes
`rsl_rl` (resolves to `.venv-mjlab/site-packages`, confirmed) but also
makes `mjlab_tasks` unimportable, since it's *only* available via the repo
root. Anything that needs both (Phase 3 onward) must instead **reorder**
`sys.path` rather than strip it — repo root moved to the *end*, not
removed:

```python
import sys
sys.path = [p for p in sys.path if p not in ("", ".")] + [""]
```

Run before any `import rsl_rl` / `import mjlab_tasks` — confirmed on this
machine: `rsl_rl` still resolves to the pip-installed copy (site-packages
entries now come first) and `mjlab_tasks` still resolves (nothing else
provides that name). `tests/conftest.py` applies this for the whole test
session so individual test files don't need to.

**R2 — Obs schema.** Resolved — see §2. Residual: pin `mjlab==1.6.0`
exactly; a future mjlab release could add/reorder actor terms and silently
break Javier's checkpoints. Phase 4's test is the canary.

**R3 — `model_7000` is not a quality baseline.** 6 terminations/400 steps.
Unknown which motion it was trained on — Javier's Kaggle log dir
(`g1_tracking/2026-08-13_06-08-32`) doesn't record it. **Open question for
Javier.**

**R4 — MuJoCo vs Genesis contact/numerics differ; policies don't transfer
across them.** Genesis and mjlab policy catalogs are permanently disjoint —
no fusing or distilling across stacks (`fusion.py` requires identical
rsl_rl architecture anyway; different backends produce structurally
different checkpoints). Also: MuJoCo Warp is **not deterministic**
(upstream `mujoco_warp#562`) — don't write tests assuming bit-exact
rollouts; use thresholds with margin, as §1's numbers do.

**R5 — Apple M1 Pro / CPU-only: evaluation works, training doesn't.**
Confirmed (mjlab's own docs: "macOS is supported for evaluation only").
Same situation as today's Genesis setup — not a regression, but Phase 6
(Kaggle GPU training under mjlab) is genuinely untested and is the single
largest unverified step in this whole migration.

**R6 — Real-robot deployment (`deploy_real/`) is explicitly out of
scope.** Unitree's real path is a C++ FSM
(`third_party/unitree_rl_mjlab/State_Mimic.cpp.reference`), not our Python
`RealAdapter`. Bridging them is a separate project.

**R7 — Gain mismatch.** mjlab derives joint stiffness from
`armature * (2π·10Hz)²`; `g1_mimic_deploy.yaml.reference` uses hand-tuned
per-joint arrays. Different numbers for the same robot — irrelevant in
sim-only work, but Phase 2's diff exists to make this visible before anyone
plugs a policy into hardware.

**R8 — One motion per policy, same as today.** mjlab's tracking task loads
one `motion_file` at env construction, no resampling. The shelved
"Movements panel" idea doesn't get easier under mjlab — switching dances
still means switching policies, same as `g1_deepmimic` today.

**R9 — Web UI gating for commandless tasks.** Tracking tasks have no
velocity command (`set_command`/`set_operator_speed_limit` are
meaningless). Check `web/app.js`'s `applyStatus()` handles this generically
(there's precedent in `--real` mode) before assuming it's free in Phase 5.

**R10 — `tests/test_driver_family_parity.py` AST-checks verbatim bodies
across driver scripts.** `rugiar_driver_mjlab.py` legitimately can't match
(different framework) — must be explicitly exempted with a written reason,
not silently broken or contorted to pass.

## §6 — G1 asset parity (Phase 2)

Joint *order* is confirmed identical everywhere in this migration (§0, §3,
and `docs/motion_imitation_integration.md`) — this section is about
*gains*, which are **not** identical, and that's fine (sim-only work),
documented so nobody is surprised later when plugging a policy into
hardware (see R7).

**Base pose.** Ours (`G1Flat29DofCommonCfg.init_state.pos`): `z=0.8`. mjlab
(`g1_constants.HOME_KEYFRAME.pos`): `z=0.783675`. ~1.6cm different resting
height — cosmetic, both are "standing roughly upright" starting points.

**Default joint angles.** Ours hand-tunes all 29 (`default_joint_angles`,
`common_cfgs.py:211-241` — e.g. `hip_pitch=-0.1`, `knee=0.3`,
`ankle_pitch=-0.2`, `shoulder_pitch=0.3`, `elbow=0.97`). mjlab's
`HOME_KEYFRAME.joint_pos` only overrides 7 regex-matched groups
(`hip_pitch=-0.1`, `knee=0.3`, `ankle_pitch=-0.2`, `shoulder_pitch=0.2`,
`elbow=1.28`, `shoulder_roll=±0.2`) — everything else defaults to 0.
Different standing pose, not a bug: the tracking task doesn't use a fixed
default pose as its behavioral target the way a walking task does (the
*motion command* is the target), so this matters less than it would for
`g1`/`go2`.

**Gains — genuinely different, by design.** mjlab derives stiffness from
motor physics: `armature * (2π·10Hz)²` per actuator family
(`STIFFNESS_5020=14.25`, `STIFFNESS_7520_14=40.18`,
`STIFFNESS_7520_22=99.10`, `STIFFNESS_4010=16.78`, each with a matched
damping). Ours (`G1Flat29DofCommonCfg.control`, `common_cfgs.py:245-265`)
is hand-tuned per joint *group* (not per motor family):
`hip=100, knee=150, ankle=40, waist_yaw=200, waist_roll/pitch=40,
shoulder/elbow/wrist=40`. **Action scale** is the starkest difference:
ours is a single flat `0.25` for every joint; mjlab's `G1_ACTION_SCALE` is
per-joint-group, ranging `0.075` (wrist pitch/yaw) to `0.548` (hip
pitch/yaw, waist yaw) — a ~7x spread ours doesn't have.

**Conclusion, per the migration plan: document, don't fix.** These are two
independently-tuned gain sets for the same robot, both presumably valid
for their own reward/action-scale conventions. Mixing them (e.g. porting
mjlab's action scale into the Genesis stack, or vice versa) is not a
drop-in change and isn't needed for this migration — mjlab tasks use
mjlab's own gains end-to-end. The only place this would ever matter is a
real-hardware deploy, where `deploy.yaml`'s hand-tuned arrays (a *third*,
independently-tuned set, see R7) are the ones that actually apply.

## §7 — The 90-day clock

Per the handoff's Opus critique: write down triggers now, don't drift.
`g1_deepmimic`/`g1_motion_vis`/Genesis-based `MotionLoader` get deprecated
(not deleted) once Phase 4 proves the mjlab path works — **by 2026-11-16,
make an explicit keep-or-kill call on the old Genesis motion-imitation
code.** Walking tasks (`g1`, `go2`, etc.) are unaffected and stay on
Genesis — no migration trigger has fired for them.
