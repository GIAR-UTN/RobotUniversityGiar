# Motion imitation — handoff to the motion team

> **Paired document.** This file has a twin in Spanish:
> [`motion_team_handoff.es.md`](motion_team_handoff.es.md). Both are meant to
> stay in sync — if you change one, change the other.

**Audience:** the collaborator who built the motion-imitation / mimic-motion
work (dance retargeting, `unitree_rl_mjlab` training, the Kaggle checkpoints)
in a separate repo, and now needs to keep working *inside* RUgiar.

**Purpose:** orientation, not history. Where each thing lives now, what is
actually validated, what was simplified, and — mainly — where and how to keep
building. For the narrative of how we got here, read
[`docs/mjlab_migration.md`](mjlab_migration.md) and
[`docs/motion_imitation_integration.md`](motion_imitation_integration.md);
don't re-derive that research.

---

## 1. The 60-second version

Your work landed via PR #3 (`g1-fullbody-motion-imitation`, merged as
`0d65112`). The integration made one big architectural decision:

**Motion imitation in RUgiar now runs on `mjlab`, not on the repo's original
Genesis stack.** We did *not* fork `unitree_rl_mjlab`; we depend on `mjlab`
itself (PyPI, pinned `mjlab==1.6.0`) and registered **our own task,
`Rugiar-G1-Mimic`**, which calls mjlab's stock
`unitree_g1_flat_tracking_env_cfg()` unmodified. That means the 154-dim
observation contract your checkpoints were trained against is **bit-for-bit
preserved** — only `task_id` and `experiment_name` are ours.

Consequences you should internalize:

- Your two ONNX checkpoints load and run **closed-loop** here, unmodified.
- Everything around them (control web, WebSocket protocol, policy catalog,
  `rugiar` CLI, training jobs, reward charts) is the repo's existing machinery
  — motion imitation is a third "family", not a bolt-on subsystem.
- The old Genesis motion path (`g1_deepmimic`, `MotionLoader`,
  `g1_motion_vis`) still exists but is **deprecated for this family**. Don't
  invest there — see §7.

---

## 2. Map — where everything lives

| What | Path |
|---|---|
| **Our mjlab task registration** (`Rugiar-G1-Mimic`) | `mjlab_tasks/__init__.py`, `mjlab_tasks/tracking/` |
| Env config (thin wrapper over mjlab's own) | `mjlab_tasks/tracking/g1_env_cfg.py` |
| PPO / runner hyperparameters | `mjlab_tasks/tracking/rl_cfg.py` |
| **Motion converter** raw `.pkl` → mjlab `.npz` | `legged_gym/scripts/process_reference_motion_mjlab.py` |
| **Training entrypoint** (mjlab) | `legged_gym/scripts/mjlab_train.py` |
| Training dispatch / backend registry | `legged_gym/control/training.py` (`BACKENDS`, `training_backend_for_task()`, `resolve_training_backend()`) |
| **Driver** (sim + control server, mjlab) | `legged_gym/scripts/rugiar_driver_mjlab.py` |
| Backend adapter (robot state ↔ control engine) | `legged_gym/control/mjlab_adapter.py` |
| Motion clip listing / switching RPCs | `legged_gym/control/service.py` (`list_motions()`, `switch_motion()`, `motion_clip_rows()`) |
| Web UI (Motion panel, Create Policy) | `web/index.html`, `web/app.js` |
| CLI (`rugiar train`, `rugiar drive`) | `legged_gym/cli/rugiar.py`; shortcuts in `Makefile` |
| **Motion clips, mjlab format** | `resources/reference_motion/unitree_g1/mjlab_run/*.npz` |
| Motion clips, raw source format | `resources/reference_motion/unitree_g1/raw_run/*.pkl` |
| **Checkpoints** | `policies/<name>/` (`checkpoint.onnx` / `checkpoint.pt` + `meta.json`) |
| Read-only upstream references | `third_party/unitree_rl_mjlab/*.reference` |
| Design contracts / plans | `docs/mjlab_migration.md`, `docs/mjlab_training_contract.md` |
| Tests | `tests/test_mjlab_*.py`, `tests/test_javier_checkpoints_track.py`, `tests/test_process_reference_motion_mjlab.py` |

### The two interpreters (this trips everyone up once)

There are **two virtualenvs on purpose**:

- `.venv` — Genesis stack (`SIMULATOR=genesis`), locomotion tasks.
- `.venv-mjlab` — mjlab stack (`SIMULATOR=mjlab`), motion tracking.

They can't be merged: this repo vendors a top-level `rsl_rl/` package that
shadows PyPI `rsl-rl-lib`, and the `genesis` extra pins `mujoco==3.10.0` while
mjlab needs `~=3.11.0`. See `docs/mjlab_migration.md` **R1**. Any new mjlab
script must reorder `sys.path` (repo root **last**, not stripped) before
importing `rsl_rl` / `mjlab_tasks` — copy the header of
`legged_gym/scripts/mjlab_train.py` verbatim; `tests/conftest.py` does the
same for the test session.

You never need `.venv-mjlab/bin/rugiar` — it doesn't exist. `rugiar` runs from
`.venv` and dispatches to the right interpreter itself.

---

## 3. What is validated (real runs, not import checks)

- **`policies/javier_mjlab_dance1_subject2`** — the reference mjlab policy.
  400-step closed-loop rollout against its own `dance1_subject2.npz`:
  **0 falls**, ~0.07–0.10 m mean body-position error. Use this as the quality
  baseline. Covered by `tests/test_javier_checkpoints_track.py`.
- **`policies/javier_mjlab_model_7000`** — loads and runs, but **6 falls** in
  the same 400 steps against `dance1_subject2` (~0.18 m error). Almost
  certainly because it was trained against a different, unknown clip (see §8,
  open question). **Not a quality baseline.**
- **The 154-dim observation table** is fully specified and asserted in a test
  (`tests/test_mjlab_env_smoke.py`), so an mjlab version bump can't silently
  reorder it under your checkpoints. See `docs/mjlab_migration.md` §2.
- **Motion conversion** — `g1moves_B_DadDance.pkl` (2509 frames @ 60 fps) →
  `g1moves_B_DadDance.npz` (2090 @ 50 fps), loads into a real tracking env and
  steps without falling. `tests/test_process_reference_motion_mjlab.py`.
- **Driving from the control web** — `Rugiar-G1-Mimic` runs over the same
  WebSocket protocol as every Genesis task: live policy switch, pause/resume/
  restart, telemetry, reference-motion ghost overlay, motion-clip selector.
- **Training end-to-end, locally** — a real short run through
  `TrainingManager.start()` reached `status="done"`, exported a stateless ONNX,
  and the resulting policy was auto-discovered and **hot-loaded into a running
  session with no relaunch** (`tests/test_mjlab_training_hotload.py`).
- **Damping-only session** — the driver survives having zero local policies,
  so you can preview a clip's ghost with the robot just holding a pose
  (`tests/test_mjlab_damping_only_session.py`).

---

## 4. What is NOT done / was simplified

Be explicit about these before planning work:

1. **No GPU training run has ever been executed on the mjlab path.** Only
   CPU/mujoco-warp. `--device cuda:0` exists in `mjlab_train.py` but is
   untested. This is the single largest unverified step (migration **R5**).
   macOS is evaluation-only per mjlab's own docs.
2. **Kaggle backend is explicitly rejected for mjlab tasks.** Kaggle's
   bootstrap is IsaacGym-specific. `rugiar train --backend kaggle` errors on an
   mjlab task rather than launching something doomed. Genesis tasks are
   unaffected. **A real cloud-GPU path for mimic training does not exist yet.**
3. **One motion per policy.** mjlab's tracking task loads a single
   `motion_file` at env construction — no resampling, no multi-clip policy.
   Switching dances means switching policies *and* relaunching the process
   (that's why `switch_motion()` relaunches instead of swapping in place).
   Migration **R8**.
4. **No self-trained mimic policy of real quality exists yet.** What's in
   `policies/` for this family is: your two imported checkpoints, plus smoke
   policies (`g1_deepmimic_smoke`, `g1_deepmimic_daddance_smoke` — 20
   iterations, Genesis-era, wiring proofs only, *not* good mimics).
5. **Only 2 clips exist in mjlab format**: `dance1_subject2.npz` (vendored
   upstream fixture) and `g1moves_B_DadDance.npz`. The other 15 clips in
   `raw_run/` are AMASS-style locomotion mocap and have only been converted to
   the Genesis format.
6. **g1-moves' own `policy_154/*.onnx` checkpoints are NOT drop-in.** Their
   obs/action dims match ours exactly, but they take a second input,
   `time_step`, and our `load_onnx_backend()` would treat any input after
   `obs` as recurrent hidden state and feed the wrong tensor back — silently
   wrong, no crash. Needs a real `OnnxPhaseConditionedPolicy` backend before
   any of them gets registered. Details in
   `HANDOFF_mimic_motion_library_ux.md`.
7. **Real-robot deployment is out of scope.** Unitree's real path is a C++ FSM
   (`third_party/unitree_rl_mjlab/State_Mimic.cpp.reference`), not our Python
   `RealAdapter`. Also note gains differ between mjlab, our Genesis configs,
   and the deploy YAML — three independently tuned sets (migration **R6**/**R7**,
   §6).
8. **Genesis and mjlab policy catalogs are permanently disjoint.** No fusing,
   no distilling across stacks — different physics, structurally different
   checkpoints (**R4**). Also: mujoco-warp is not deterministic, so never write
   a test asserting bit-exact rollouts; use thresholds with margin.

---

## 5. How to do the four things you'll actually want to do

### 5.1 Add a new motion clip

The pipeline is: **source `.pkl` → `raw_run/` → converter → `mjlab_run/*.npz`**.

1. Get a retarget-stage `.pkl` with `fps`, `root_pos (N,3)`,
   `root_rot (N,4)` **xyzw**, `dof_pos (N,29)`. A `exptech/g1-moves`
   `retarget/<clip>.pkl` works as-is — its 29-DOF joint order is confirmed
   identical to ours, and its root quaternion is already xyzw. (The
   training-stage `.npz` uses **wxyz** — don't take `root_rot` from there.)
2. Drop it at
   `resources/reference_motion/unitree_g1/raw_run/<clip>.pkl` (set
   `local_body_pos` / `link_body_list` to `None` if absent).
3. Convert:

   ```bash
   CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python \
       legged_gym/scripts/process_reference_motion_mjlab.py \
       --motion_file unitree_g1/raw_run/<clip>.pkl \
       --motion_out_dir unitree_g1/mjlab_run
   ```

4. **Look at it before training on it.** Preview the clip's ghost overlay with
   no policy needed:

   ```bash
   rugiar drive mjlab --motion_file resources/reference_motion/unitree_g1/mjlab_run/<clip>.npz
   ```

   Or pick it from the **Motion** panel in the control web (it lists every
   `.npz` in that directory and badges which ones already have a policy —
   clips with no policy are deliberately *not* disabled).

If your private pipeline (LLM → video → 3D reconstruction → retargeting) can
emit that same retarget-`.pkl` shape, step 1 becomes a no-op and the whole
pipeline is yours for free. If it emits something else, the right place to add
an input adapter is `process_reference_motion_mjlab.py` — swap the input side,
keep the MotionLoader/`run_sim` math.

**Licensing:** `g1-moves` is CC-BY-4.0 and requires attribution. Record the
source in the clip's derived policy `meta.json` `note`, same as the existing
entries do.

### 5.2 Train a policy against a clip

`rugiar train` auto-detects the backend from the task — same flags either way,
you never pick the interpreter:

```bash
rugiar train --list_motions --task Rugiar-G1-Mimic     # what exists, what has a policy
rugiar train --task Rugiar-G1-Mimic --list_reward_scales

rugiar train --task Rugiar-G1-Mimic --name mimic_dance \
    --num_envs 4096 --max_iterations 3000 \
    --motion_file resources/reference_motion/unitree_g1/mjlab_run/<clip>.npz
```

Notes:

- `--motion_file` is **required** for a tracking task; the CLI errors up front.
  It must be the `.npz` under `mjlab_run/`, not the `.pkl`.
- Genesis-only flags (`--cmd_vx_range`, `--push_interval_s`, …) are rejected —
  a tracking task has no velocity command.
- The **9 reward terms** for this task (this is the vocabulary; do not reuse
  Genesis's `tracking_lin_vel` etc.):
  `motion_global_root_pos` (0.5), `motion_global_root_ori` (0.5),
  `motion_body_pos` (1.0), `motion_body_ori` (1.0), `motion_body_lin_vel` (1.0),
  `motion_body_ang_vel` (1.0), `action_rate_l2` (−0.1), `joint_limit` (−10.0),
  `self_collisions` (−10.0). Override with
  `--reward_scale motion_body_pos 2.0`.
- The same job can be launched from the web's **Create Policy** panel, which
  shows mjlab-shaped fields (motion-clip picker, no velocity envelope).
- ETA estimates are calibrated per backend — mjlab and Genesis throughput
  histories are never pooled.
- **Never trust the reward curve alone.** Load the resulting
  `checkpoint.onnx` in the driver and watch it against the ghost overlay.

### 5.3 Register an externally-trained checkpoint

Create `policies/<name>/` with the ONNX (and `.onnx.data` if the export is
split) plus a `meta.json`:

```json
{
  "task": "Rugiar-G1-Mimic",
  "trained_via": "external-import",
  "simulator": "mjlab",
  "category": "g1-mjlab-mimic",
  "motion_file": "resources/reference_motion/unitree_g1/mjlab_run/<clip>.npz",
  "note": "provenance: where it came from, license, what it was trained against, how it was validated"
}
```

`task` is what drives compatibility checks; `category` is cosmetic (it just
distinguishes imported from self-trained in the UI); `motion_file` is what
makes the Motion panel's `has_policy` badge exact instead of a name heuristic.
Policies are auto-discovered — no registry file to edit. Prove it with a real
rollout before calling it working (follow the pattern in
`tests/test_javier_checkpoints_track.py`).

### 5.4 Change the task itself (rewards, DR, hyperparameters)

- Reward weights / domain-randomization ranges / anything env-side: mutate the
  cfg returned by `rugiar_g1_mimic_env_cfg()` in
  `mjlab_tasks/tracking/g1_env_cfg.py`. **Add a delta, don't copy mjlab's
  config in** — that's the whole point of calling their factory, so upstream
  fixes keep flowing.
  ⚠️ Any change here changes the observation/reward contract your existing
  checkpoints assume. If you touch the *observation* terms, Javier's
  checkpoints stop being valid, and `tests/test_mjlab_env_smoke.py` will (by
  design) tell you.
- PPO hyperparameters, `max_iterations`, `save_interval`, `experiment_name`:
  `mjlab_tasks/tracking/rl_cfg.py`.
- Training-worker behavior (chunking, progress/result files, ONNX export
  shape): `legged_gym/scripts/mjlab_train.py`, spec'd in
  `docs/mjlab_training_contract.md`. Note the export deliberately uses the
  parent runner's `export_policy_to_onnx` to get a clean 1-input/1-output
  graph — the tracking runner's auto-export is a 2-input/7-output graph our
  loader would drive incorrectly.
- Adding a whole new training backend (a GPU box, a second cloud): append one
  `TrainingBackend(...)` descriptor to `BACKENDS` in
  `legged_gym/control/training.py` — no edits to `start()`.

---

## 6. Running it day to day

```bash
make drive-mjlab                       # control web, task Rugiar-G1-Mimic
make drive-genesis                     # the locomotion side, for contrast
rugiar drive mjlab --task Rugiar-G1-Mimic --motion_file <npz>
rugiar drive mjlab --headless          # scripted smoke test, no viewer
```

Control web on `:9017`, raw viser viewer on `:9006` by default. The launcher
stops whatever is already on the control port first — one session per port,
never start a second one alongside.

Tests:

```bash
SIMULATOR=genesis .venv/bin/python -m pytest tests/ -q
SIMULATOR=mjlab CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest tests/ -q
```

The `SIMULATOR=` prefix is mandatory in both cases (`legged_gym/__init__.py`
gates on it). Run **both** suites — a change to `legged_gym/control/` affects
both stacks.

---

## 7. Where *not* to spend effort

- **`g1_deepmimic` / `g1_motion_vis` / `legged_gym/utils/motion_loader.py`** —
  the Genesis-era motion-imitation path. It works, and its 1380-dim observation
  (`frame_stack=5 × (151 + 125)`) is fundamentally incompatible with the mjlab
  154-dim contract. It's deprecated, not deleted, with an explicit
  keep-or-kill decision due **2026-11-16** (`docs/mjlab_migration.md` §7).
  New mimic work goes to mjlab.
- **Forking / depending on `unitree_rl_mjlab`.** We deliberately don't. Its
  tracking task is a near-verbatim copy of mjlab's own, it pins outdated
  versions, and it isn't pip-installable (top-level package literally named
  `src`). We vendored exactly three read-only reference files under
  `third_party/unitree_rl_mjlab/`; take more from there only when you need it,
  as reference. Rationale: `docs/mjlab_migration.md` §0.
- **Building a "bridge" panel between Genesis and mjlab families.** Rejected
  by design — mjlab is a third family in the existing Family panel, no bridge.
- **Bumping `mjlab` off 1.6.0 casually.** The pin protects your checkpoints'
  observation contract. If you bump it, run
  `tests/test_mjlab_env_smoke.py` and `tests/test_javier_checkpoints_track.py`
  first; they're the canary.

---

## 8. Open questions for you

1. **What motion clip was `model_7000` trained against?** Your Kaggle log dir
   (`g1_tracking/2026-08-13_06-08-32`) doesn't record it, and it's the most
   likely explanation for its 6 falls against `dance1_subject2`
   (migration **R3**).
2. **What format does your private pipeline (LLM → video → 3D reconstruction →
   retargeting) emit?** If it's already the retarget-`.pkl` shape (fps,
   root_pos, root_rot xyzw, dof_pos ×29), §5.1's converter works for you
   unchanged and there's nothing to build. If not, tell us the shape and the
   input adapter is a small, one-time addition.
3. **Do you have intermediate `model_N.pt` checkpoints, or other
   `deploy/robots/g1/config/policy/` exports worth importing?** We discarded
   the 664 MB `unitree_rl_mjlab` clone after extracting the two ONNX files; we
   can re-download, but if you have them at hand it's faster.
4. **Do you have access to an NVIDIA GPU box for training?** That's the
   blocking gap (§4.1/§4.2) — everything else is ready and CPU-validated.

---

## 9. Suggested next steps, in order

1. **Answer §8.1** (which clip `model_7000` targets) — cheap, unblocks
   interpreting the only quality datapoint we have besides `dance1_subject2`.
2. **Get one GPU training run through** `rugiar train --task Rugiar-G1-Mimic`
   at real scale (`num_envs=4096`, thousands of iterations) against
   `g1moves_B_DadDance.npz`. That produces the repo's **first genuinely good
   self-trained mimic policy** and simultaneously closes the biggest untested
   risk in the migration.
3. **Bulk-import clips from `exptech/g1-moves`** (~61 clips; only
   `retarget/<clip>.pkl` is needed, ~230 MB of actually-useful files) through
   §5.1's converter, so the Motion panel becomes a real library instead of two
   entries.
4. **If you want their pretrained policies too**, first fix the `time_step`
   input handling (§4.6) — write the phase-conditioned ONNX backend, verify
   against a real rollout, *then* register any of them.
5. **Wire your own retargeting pipeline's output** into `raw_run/` so new
   dances flow in without a manual step.
