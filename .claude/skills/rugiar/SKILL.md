---
name: rugiar
description: Create/fine-tune RobotUniversityGiar policies from the command line with the `rugiar` CLI — picking the right simulator per OS (Genesis on Mac/Linux CPU, IsaacGym/IsaacLab on Linux+NVIDIA), or training on Kaggle's free cloud GPU. Use whenever the user wants to train a policy, fine-tune one, discover tasks/reward scales/local policies, or set up Kaggle for cloud training.
allowed-tools: Bash(rugiar:*) Bash(.venv/bin/rugiar:*) Bash(pip install:*) Bash(python3 -c:*) Bash(mkdir -p ~/.kaggle:*) Bash(chmod 600 ~/.kaggle/kaggle.json) Bash(mv:*) Bash(export SIMULATOR=*)
---

# rugiar — CLI for creating RobotUniversityGiar policies

`rugiar train` is a thin, argument-complete front end onto
`legged_gym.control.training.TrainingManager` — the exact engine the control
web's "Create Policy" panel uses (`legged_gym/cli/rugiar.py`). It launches
`legged_gym/scripts/web_train.py` as a subprocess, streams its log live, and
on success writes a self-contained `./policies/<name>/` folder
(`checkpoint.pt`, `train_checkpoint.pt`, `train.log`, `meta.json`) —
identical in shape to a policy trained through the browser UI.

Installed as a real command via `[project.scripts]` in `pyproject.toml`; if
`rugiar` isn't on PATH, use `.venv/bin/rugiar` (or `source .venv/bin/activate`
first).

**The live, authoritative flag reference is `rugiar train --help` — every
group (compute budget, fine-tuning, command envelope, stability targets,
push perturbation, reward shaping, backend, discovery) is documented there
with defaults. Run it before guessing a flag name.** What follows is the
part `--help` can't tell you: which simulator to use where, and how to get
Kaggle cloud training working.

## Prerequisite: SIMULATOR must be set

Every `rugiar` command (even `--list_tasks`) imports `legged_gym`, which
**refuses to import at all** unless `SIMULATOR` is set:

```bash
export SIMULATOR=genesis    # or isaaclab — see "Choosing a simulator" below
```

Forgetting this produces `ValueError: Unsupported SIMULATOR type...` before
any argparse error — if a `rugiar` command fails immediately with that, this
is why.

## Quick start

```bash
export SIMULATOR=genesis
rugiar train --list_tasks                       # what can I train?
rugiar train --task g1 --list_reward_scales      # what reward terms can I tune?
rugiar train --list_policies                     # what's already local, fine-tunable?

# train from scratch, stop after 15 minutes
rugiar train --task g1 --name crouch --max_minutes 15 \
    --base_height_target 0.45 --push_robots off

# fine-tune an existing local policy
rugiar train --task g1 --name crouch_v2 --from_policy crouch \
    --max_iterations 500 --reward_scale action_rate -0.1

# train on Kaggle's free GPU instead of this machine
rugiar train --task g1 --name cloud_walk --max_iterations 1000 --backend kaggle
```

Ctrl-C during a run terminates the training subprocess and leaves the policy
**unregistered** (nothing gets written to `./policies/`) — safe to interrupt.

## Choosing a simulator per OS/target

Simulator selection in this repo is **not** a single uniform switch — it's a
property of *where the job runs*, driven by `legged_gym/__init__.py` (env var
`SIMULATOR` for genesis/isaaclab) and, for Isaac Gym specifically, by which
**Python interpreter** runs the job (`legged_gym/__init__.py` hardcodes
`SIMULATOR="isaacgym"` for any interpreter `<=3.8`, and rejects the string
`"isaacgym"` outright on `>=3.10`). `switch_simulator.sh` (repo root)
automates this locally via three conda envs (`lr_gym`, `lr_gen`, `lr_lab`).

| Where you're training | Simulator | Why |
|---|---|---|
| **macOS** (Apple Silicon or Intel, no NVIDIA GPU) | **Genesis** — `SIMULATOR=genesis` | The only one of the three that runs with no GPU at all; this is what the whole repo was built and proven on (README §1/§2). CPU-only works fine; Genesis's Metal GPU path on macOS was flaky enough that it isn't depended on. |
| **Linux, no NVIDIA GPU / CPU-only** | **Genesis** — `SIMULATOR=genesis` | Same as macOS — Genesis is CPU-portable, IsaacGym/IsaacLab are not. |
| **Linux with an NVIDIA GPU** | **Genesis** (`SIMULATOR=genesis`, `GENESIS_BACKEND=cuda` if using Docker Compose's GPU overlay) for the same setup everywhere, **or IsaacGym** (needs a Python **≤3.8** env — `switch_simulator.sh isaacgym` / conda env `lr_gym`) **or IsaacLab** (`SIMULATOR=isaaclab`, Python ≥3.10, `pip install -e .[isaaclab]`) if you specifically need one of those ecosystems | All three are real options on Linux+NVIDIA; Genesis stays the simplest (same commands as Mac), IsaacGym/IsaacLab are there for parity with the upstream `unitree_rl_gym` pipeline. |
| **Windows** | Not natively documented — use **Docker Compose** (`docker compose up --build`, works via Docker Desktop/WSL2 on any host arch) | Same recommended path as any host where a native Python+Genesis setup is inconvenient; see README §2 "Docker Compose". |
| **Kaggle (cloud GPU)** | **IsaacGym**, always — regardless of what `SIMULATOR` is set to locally | `rugiar train --backend kaggle` bootstraps its own throwaway Python 3.8 + Isaac Gym venv *inside the Kaggle kernel* (see `legged_gym/control/kaggle_backend.py`). Kaggle's free tier hands out a Pascal (sm_60) GPU — Genesis's GPU JIT needs Volta+ (sm_70+) and cannot run there at all, while Isaac Gym's PhysX GPU pipeline works on Pascal and gets a genuine speedup (confirmed, see `HANDOFF_kaggle_cloud_gpu.md`). Your local `--backend local` runs still use whatever `SIMULATOR` you have set — the two are independent. |

`rugiar train`'s own `--backend {local,kaggle}` only picks *where the job
runs*; for `local` it inherits whatever `SIMULATOR` is currently exported —
it does not itself switch simulators, so get the environment right first
(`switch_simulator.sh` or `export SIMULATOR=...`) before running a local job.

## Setting up Kaggle for cloud training

One-time setup, then every future `--backend kaggle` job just works:

1. **Create a free Kaggle account** at kaggle.com if you don't have one.
2. **Verify your phone number** — Settings → Phone Verification. This is
   required to unlock Kaggle's GPU quota (free tier gives ~30 GPU-hours/week);
   without it, kernels run CPU-only or fail to start.
3. **Create an API token** — click your avatar → *Settings* → *API* section →
   **Create New Token**. This downloads a `kaggle.json` file.
4. **Install it locally**:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
5. **Install the `kaggle` package** (only needed for the Kaggle backend):
   ```bash
   pip install -e .[cloud]     # or: pip install kaggle
   ```
6. **Verify credentials are picked up**:
   ```bash
   python3 -c "from legged_gym.control.kaggle_backend import kaggle_credentials_available; print(kaggle_credentials_available())"
   # should print True
   ```
7. **Train on Kaggle**:
   ```bash
   export SIMULATOR=genesis   # still required locally, even though the job itself runs isaacgym remotely
   rugiar train --task g1 --name cloud_walk --max_iterations 1000 --backend kaggle
   ```

Things worth knowing about Kaggle jobs specifically:

- **The Kaggle kernel clones this repo fresh from GitHub** (it's public) at
  push time — a Kaggle job trains whatever is on the remote `main` branch's
  HEAD, **not your uncommitted local changes**. Commit and push first if you
  need a specific change reflected in a cloud run.
- **Isaac Gym bootstrap costs ~3-4 minutes** before training even starts —
  normal, not a hang.
- Kaggle enforces a **~6-hour session cap**; `rugiar` bails out well before
  that so a stuck kernel can't wedge the wait loop forever.
- `--from_policy NAME` works with `--backend kaggle` too — the local
  policy's checkpoint is uploaded as a private Kaggle Dataset automatically.
- A Kaggle-trained policy's `meta.json` records `"simulator": "isaacgym"` —
  it's a genuine sim2sim transfer relative to this repo's Genesis-trained
  policies (different contact/PD dynamics), not guaranteed identical
  behavior. `rugiar train --list_policies` doesn't show this — check
  `policies/<name>/meta.json` or the control web's info popup.
- No credentials found → `rugiar` (via `TrainingManager.start()`) fails
  fast with a clear error before launching anything; re-run steps 1-6 above.

## Common recipes

```bash
# crouch instead of walk: zero velocity commands, target a lower base height
rugiar train --task g1 --name crouch --max_iterations 1000 \
    --cmd_vx_range 0 0 --cmd_vy_range 0 0 --cmd_yaw_range 0 0 \
    --base_height_target 0.45

# cautious gait: penalize torque/joint-velocity harder from an existing base
rugiar train --task g1 --name cautious --from_policy stable \
    --max_iterations 800 --reward_scale torques -0.001 --reward_scale dof_vel -0.01

# push-robustness training, pushes biased from behind
rugiar train --task g1 --name push_robust --max_iterations 1200 \
    --push_robots on --max_push_vel_xy 1.5 --push_interval_s 4 --push_dir behind

# lower PPO exploration noise if 'Mean action noise std' was trending up last run
rugiar train --task g1 --name retrain --from_policy retrain \
    --max_iterations 500 --entropy_coef 0.001
```

## Troubleshooting

- **`ValueError: Unsupported SIMULATOR type...`** → `export SIMULATOR=genesis`
  (or `isaaclab`) before anything else — see "Prerequisite" above.
- **`give at least one of --max_iterations / --max_minutes`** → both are
  optional individually but at least one stopping condition is required.
- **`unknown reward scale(s) for task '<task>': ...`** → run
  `rugiar train --task <task> --list_reward_scales` to see valid names and
  current defaults for that task before retrying `--reward_scale`.
- **`unknown base policy '<name>'` / fine-tuning fails** → run
  `rugiar train --list_policies`; only entries with `fine-tunable=yes` (i.e.
  they have a `train_checkpoint.pt`) work with `--from_policy`.
- **`no Kaggle credentials found at ~/.kaggle/kaggle.json`** → follow
  "Setting up Kaggle for cloud training" above.
- Job failed with a subprocess exit code → the error message names the exact
  log file (`logs/_web_training/<job_id>.log`) — `rugiar` already streamed it
  live, but it's still there to re-read.
