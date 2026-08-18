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
1. Motion converter: our `.pkl` → mjlab `.npz`
   (`legged_gym/scripts/process_reference_motion_mjlab.py`).
2. G1 robot config parity check (docs-only diff, no port needed —
   mjlab ships G1 29-DoF natively).
3. Register `Rugiar-G1-Mimic`, our own mjlab tracking task (`mjlab_tasks/`
   package at repo root).
4. Flip Javier's checkpoints to the registered task; closed-loop test
   with the thresholds from §1 (with margin — MuJoCo Warp is not
   deterministic, see R4).
5. Wire into `rugiar_driver_mjlab.py` / `MjlabAdapter` / web UI (third
   family, existing extension point — no new panel).
6. Training path (Kaggle), first self-trained mjlab policy, and the
   deprecation of `g1_deepmimic`/Genesis for this family (walking tasks
   stay on Genesis — no trigger to move them).

## §5 — Risks (R1–R10)

**R1 — `rsl_rl` name collision (verified, hard).** This repo vendors a
top-level `rsl_rl/` package. Running `.venv-mjlab/bin/python` from the repo
root (e.g. `python -c "..."`) resolves `import rsl_rl` to the **vendored**
copy, not PyPI `rsl-rl-lib==5.4.2` — confirmed on this machine. Fix:
**always invoke with `-I`** (isolated mode) — confirmed this correctly
resolves to `.venv-mjlab/lib/.../site-packages/rsl_rl`. Secondary: `genesis`
extra pulls `mujoco==3.10.0`; mjlab needs `~=3.11.0`. Both are why this is a
fully separate venv, not an extra in the main one.

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

## §6 — The 90-day clock

Per the handoff's Opus critique: write down triggers now, don't drift.
`g1_deepmimic`/`g1_motion_vis`/Genesis-based `MotionLoader` get deprecated
(not deleted) once Phase 4 proves the mjlab path works — **by 2026-11-16,
make an explicit keep-or-kill call on the old Genesis motion-imitation
code.** Walking tasks (`g1`, `go2`, etc.) are unaffected and stay on
Genesis — no migration trigger has fired for them.
