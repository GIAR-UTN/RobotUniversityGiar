---
name: rugiar
description: Create/fine-tune RobotUniversityGiar policies from the command line with the `rugiar` CLI — picking the right simulator per OS (Genesis on Mac/Linux CPU, IsaacGym/IsaacLab on Linux+NVIDIA), or training on Kaggle's free cloud GPU. Also configures the local-policy display order (`rugiar order`) that a separate downstream control app can read. Use whenever the user wants to train a policy, fine-tune one, discover tasks/reward scales/local policies, set/view policy display order, or set up Kaggle for cloud training.
allowed-tools: Bash(rugiar:*) Bash(.venv/bin/rugiar:*) Bash(pip install:*) Bash(python3 -c:*) Bash(mkdir -p ~/.kaggle:*) Bash(chmod 600 ~/.kaggle/kaggle.json) Bash(mv:*) Bash(export SIMULATOR=*)
---

# rugiar — CLI for creating RobotUniversityGiar policies

## ⚠️ Known open problem: no policy this repo's own pipeline produced has been confirmed to actually walk

As of 2026-08-09, **every G1 policy in `./policies/` known to genuinely walk when
tested (`stable`, `kaggle_g1`, `kaggle_g1_mid5000`) was imported from an EXTERNAL,
already-fully-converged run of stock `unitree_rl_gym`** (10000/10000 iterations,
trained outside this repo entirely — see their `meta.json`'s `"trained_via":
"kaggle-imported"` / `"source"` fields). **Nothing this repo's own training pipeline
has produced — via the control web historically, or via `rugiar` — has been
confirmed by direct observation to walk**, including checkpoints whose reward
curve looked good (`stable_home_made`: reward 14.5, episode_length 578.7 at just
290 iterations). Pushed a command all the way forward on one of these and it took
zero steps and didn't even look crouched on a crouch-targeted one — that's the
symptom that surfaced this.

**Leading hypothesis, not yet confirmed by watching a checkpoint in sim:**
iteration count. Every self-trained checkpoint here stopped at 290-970 iterations;
the two known-good ones ran the full 10000 (`unitree_rl_gym`'s own documented
target — see "Scale matters" below). 290 iterations may only be enough to reach a
"survive without falling" local optimum (scores fine on `alive`/`orientation`/
`base_height`) without ever learning confident, commanded directed locomotion —
but this is inference from indirect evidence (`iterations_done` correlating with
"works"), not a confirmed root cause. **Do not treat "more iterations" as a proven
fix until a fork-trained checkpoint has actually been watched walking** (`play.py
--viewer=viser`, per the Troubleshooting entry below) — reaching 5000-10000
iterations from scratch at `num_envs=4096` costs roughly 7-14 hours of GPU compute
(see "Scale matters"), a large fraction of Kaggle's free ~30h/week quota, so verify
the hypothesis on a cheaper/shorter run before committing that budget blind again.

**If you're picking this up fresh:** don't trust ANY reward-curve/episode_length
number from this repo's own training as proof of a working gait, no matter how
good it looks — watch it. This exact warning already existed in the repo's own
README ("Reward-curve summaries aren't enough to trust a checkpoint — watch it")
before tonight, and got ignored anyway; don't repeat that.

---

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

# train on Kaggle's free GPU instead of this machine — num_envs=4096 is the
# community/upstream standard; 1500 here is enough to prove the flags/pipeline
# work end-to-end, NOT enough to trust the result walks — see the "Known open
# problem" banner and "Scale matters" below before treating any output as done
rugiar train --task g1 --name cloud_walk --backend kaggle --num_envs 4096 --max_iterations 1500
```

Ctrl-C during a run terminates the training subprocess and leaves the policy
**unregistered** (nothing gets written to `./policies/`) — safe to interrupt.

## Scale matters: `num_envs` and iteration budget aren't cosmetic knobs

**A real lineage in this repo failed specifically because of this**, so it's called out
before anything else. Two runs with the "same" reward function and roughly the same
flags can produce wildly different policies depending on `num_envs` and how many
iterations you actually give it:

| Run | Backend | `num_envs` | Iterations | Reward/episode_length | Actually confirmed to walk? |
|---|---|---|---|---|---|
| `kaggle_g1` (external, `unitree_rl_gym` stock) | Kaggle GPU | 4096 | **10000** (full) | not recorded | **Yes** |
| `kaggle_g1_mid5000` (external, same run) | Kaggle GPU | 4096 | 5000 (half) | not recorded | **Yes** |
| `stable_home_made` (this repo's own pipeline, historical) | Kaggle GPU | 4096 | 290 | reward 14.5, episode_length 578 | **No** — looked good on paper, never confirmed by watching it, later reported not to actually step forward |
| `scratch_walk_base` (this repo, 2026-08-09) | local CPU | 64 | 3000 (10x more than above!) | reward 4.97, episode_length 285 | **No** |

Two separate lessons live in this table, don't conflate them:

1. **`num_envs` matters per-iteration**: `scratch_walk_base` got 10x more iterations
   than `stable_home_made` at `num_envs=64` instead of `4096` and still scored worse
   — each iteration at 4096 envs collects ~64x more environment experience per PPO
   update, so iteration count alone isn't a fair comparison across `num_envs`.
   `num_envs=4096` is the documented community standard for Isaac Gym / rsl_rl
   humanoid locomotion (confirmed against
   [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) and
   [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym),
   the exact upstream this fork descends from).
2. **Even at the right `num_envs`, 290 iterations is not enough** — see the
   "Known open problem" banner at the top of this file. The two rows that are
   *actually confirmed* to walk both ran the FULL 10000-iteration target
   (`unitree_rl_gym`'s and this repo's own `G1RoughCfgPPO.runner.max_iterations`).
   A good-looking reward curve at a few hundred iterations is not evidence of a
   working gait — it wasn't for `stable_home_made`, don't assume it will be for
   the next one either.

**Practical rule: `--num_envs 64` (rugiar's local default) is for smoke-testing the
CLI/flags/reward wiring cheaply on a laptop CPU, never for a policy you actually
want to be good.** For anything meant to walk/balance for real, use
`--backend kaggle` with `--num_envs 4096` and budget iterations toward the
**thousands (5000+), not hundreds** — see "Running a long training job
unattended" below for how to fit that through this environment's execution-time
limits, and its GPU-hour cost, which is real: at this repo's own measured
~5.1s/iteration at `num_envs=4096`, 5000 iterations is **~7 hours** of GPU
compute and 10000 is **~14 hours** — a large fraction of Kaggle's free ~30h/week
quota for one policy. Budget accordingly; don't launch a 10000-iteration run
without accounting for this.

## Configuring the display order (for a downstream control layer)

`rugiar order` sets a catalog-wide display order for local policies —
separate from training/fine-tuning, and separate from any one policy's own
files. It exists so a **different app** — one that only *selects* among
already-trained policies (e.g. a runtime control layer switching what's
active on the robot) rather than training them — has something authoritative
to read instead of inventing its own "newest first" or alphabetical opinion.
`legged_gym.control.training.TrainingManager.get_policy_order()` /
`.set_policy_order()` are the same calls in Python, for that app to use
directly instead of shelling out.

```bash
rugiar order --show                                   # current order
rugiar order --set stable_home_made_4 crouch_walk      # pin these first
rugiar train --list_policies                           # now printed in that order
```

- `--set` only needs to name the policies you want to *pin* — anything left
  out keeps appearing afterward, alphabetically, so a freshly trained policy
  is never silently hidden from the order.
- Stored at `./policies/.policy_order.json` — a flat JSON list of names,
  not inside any one `policies/<name>/` folder (order is a property of the
  catalog, not of a policy) and not merged into `meta.json`.
- `--set` with a name that isn't a real local policy fails fast with
  `unknown local policy: <name>` — check spelling against
  `rugiar train --list_policies` first.

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

## Running a long training job unattended (background execution has a time ceiling)

Confirmed empirically (repeatedly, the hard way): when `rugiar train` runs as a
background process under an agent/CLI orchestrator (e.g. launched via a coding
agent's background-bash tool), **the process gets killed after roughly 28-30
minutes of wall-clock time — even while actively producing output**, with no error
from `rugiar`/Python itself (external SIGKILL, not a crash). This is a property of
the orchestrating environment, not of `rugiar`, Genesis, or Kaggle — it applies
identically to local CPU runs and to `--backend kaggle` runs (the LOCAL `rugiar`
process still has to stay alive the whole time to poll Kaggle and download/finalize
the result, even though the actual GPU compute happens remotely).

**The fix: chunk any run expected to take longer than ~20-25 minutes into multiple
`rugiar train` calls chained with `--from_policy`, each sized to finish comfortably
under that ceiling**, e.g.:

```bash
# chunk 1: from scratch
rugiar train --task g1 --name walk_c1 --max_iterations 1000 ...

# chunk 2: continues from chunk 1's checkpoint — iteration count in the log/meta.json
# is cumulative (counted from wherever --from_policy resumed), so "1000" here means
# +1000 more, landing at 2000 total
rugiar train --task g1 --name walk_c2 --from_policy walk_c1 --max_iterations 1000 ...

# final chunk — reuse the name you actually want to keep
rugiar train --task g1 --name walk --from_policy walk_c2 --max_iterations 1000 ...
```

- Pick a chunk size empirically from the first chunk's `elapsed_s` in its
  `meta.json` — if it took 830s for 1000 iterations, ~1000-1200 iterations per
  chunk leaves comfortable margin under the ceiling.
- If a chunk itself gets killed mid-run (it will happen — the exact ceiling seems
  to have some jitter), it leaves **nothing registered** (same as Ctrl-C) — just
  retry the exact same command; nothing was corrupted, you only lose that chunk's
  partial progress since its last internal checkpoint save.
- Delete the intermediate `_c1`/`_c2`-style policies once a later chunk
  successfully continues from them — the final one's `train_checkpoint.pt`
  already contains that lineage; this repo's own history does the same
  (`stable_home_made_step1_backup` was dropped once `stable_home_made` didn't
  need it anymore).
- `caffeinate -is <command>` (macOS) is worth wrapping around a local run left
  overnight so the machine itself doesn't sleep mid-chunk — unrelated to the
  ~28-30min ceiling above, but a real second way to lose an unattended run.
- This ceiling is exactly why the previous section says budget iterations in the
  thousands via **chunking**, not by asking for one giant `--max_iterations 10000`
  call and expecting it to run overnight unattended in one shot — it won't.

## Training a crouched-but-mobile policy

There is **no dedicated crouch task** — a `g1_crouch` task existed briefly with an
open-ended "as low as it can sustain" reward term, but it was removed (dead/orphaned
code, never worth the extra task class). Everything below is achieved on the plain
`g1` task purely through `rugiar train`'s existing flags — `--base_height_target`,
`--from_policy`, iteration budget. No reward-term surgery needed.

**A real failure, so you don't repeat it:** fine-tuning `--base_height_target` down
AND (re-)enabling the full command/push envelope, in the SAME short fine-tune run,
crashed a real lineage here — reward and `Mean episode length` both collapsed
(491 → 150 steps) and never recovered in the few minutes/iterations given. A
following short "fix" run that boosted `--reward_scale tracking_lin_vel` recovered
the *reward number* (falls make short bursts of high per-step reward look fine in
the log) without recovering actual stability — `Mean episode length` stayed low.
**Reward went up while the robot was still falling constantly — always read reward
and episode length together, never reward alone**, and watch it directly with
`play.py --viewer=viser` before trusting a checkpoint (see repo README §4, "Reward-
curve summaries aren't enough to trust a checkpoint").

What actually works:

```bash
# fine-tune an already-good WALKING base (not a static/zero-velocity one) toward a
# slightly lower stance, keeping the full command envelope it already knows
rugiar train --task g1 --name walk_crouch --from_policy <a policy that already walks well> \
    --max_iterations 800 \
    --base_height_target 0.72 \
    --entropy_coef 0.001
```

- **Base choice matters more than any flag — and `episode_length` alone is NOT
  enough evidence a base "already walks well."** A checkpoint here scored
  episode_length 576.83 (near the best externally-confirmed reference) and was
  still reported not to take a single step under a full-forward command — see the
  "Known open problem" banner at the top of this file. Confirm a base actually
  walks by watching it (`play.py --viewer=viser`, see Troubleshooting) before
  spending a fine-tune run on top of it, not just by reading its `meta.json`.
- **Move the target gently, not aggressively** — a few % off whatever height the
  base already trained at (see its `meta.json` command, or the task's own default
  via `--list_reward_scales`'s `base_height` note isn't shown per-value, but
  `policy_info()`/`meta.json` records what a specific checkpoint used), not a
  guessed round number far away from it.
- **Don't change the command/push envelope in the same run** as the height-target
  change — inherit whatever the base already trained under (leave `--cmd_*_range`/
  `--push_robots` unset) unless you're deliberately budgeting extra iterations for
  BOTH adjustments to converge.
- **Give it real iterations** — the 800 above is illustrative of the FLAGS, not a
  proven-sufficient budget; per "Scale matters," treat a few hundred iterations as
  "enough to prove the pipeline runs," not "enough to trust the result."
- **Don't reach for `--reward_scale tracking_lin_vel` as a first move** — a spiking
  reward with flat/low episode length is a sign to look at stability, not turn up
  velocity-tracking pressure further.

### Training the WALKING base itself from scratch (not fine-tuning a height target)

If there's no existing good base to fine-tune from and you're training from random
init, **don't stage it as "learn to walk WITHOUT pushes first, then add pushes in a
separate later fine-tune."** A real attempt at that here got stuck — after the
initial push-free stage looked fine (episode_length 285), turning `--push_robots on`
in a follow-up fine-tune collapsed it (episode_length 285 → ~90) and it did NOT
recover even after 3x more iterations at that stage, plateauing around 90-130.

This matches upstream: `unitree_rl_gym`'s own `G1RoughCfg.domain_rand` config has
`push_robots = True` **from iteration 0**, not introduced later — the policy learns
balance and push-recovery jointly across the whole run instead of specializing on a
push-free task first and then having to unlearn that specialization under a sudden
distribution shift. Prefer:

```bash
# --push_robots on (or just leave it — it's the task's own default already, see
# G1RoughCfg.domain_rand.push_robots = True) from the very first chunk, not added
# later. 1500 here is one CHUNK (see "Running a long training job unattended") —
# chain several of these via --from_policy toward 5000+ total before expecting a
# real gait, per the "Known open problem" banner at the top of this file.
rugiar train --task g1 --name walk_c1 --backend kaggle --num_envs 4096 \
    --max_iterations 1500 --entropy_coef 0.002
```

General curriculum-learning literature (see Sources) does support *gradually*
ramping up domain-randomization difficulty (terrain roughness, friction range) to
avoid early-learning collapse — but for the specific push perturbation on this
specific robot, the proven reference config is "on from the start," not a staged
curriculum. Follow the reference config over generic theory when they disagree.

## Picking up policies trained outside the web (no restart needed)

A policy `rugiar` just finished training **won't appear in a running control web**
until you either restart the server or hit its **Refresh button** (circular-arrow
icon, top of the Policies panel) — this is expected, not a bug, and the underlying
mechanics are worth understanding if it seems to not be working:

- `swap_experiment.py` (the process behind the control web) scans `./policies/`
  **once, at its own startup**. A policy trained via the web's own "Create Policy"
  panel appears live afterward because that training job runs *inside the same
  process* (`drain_finished_training()`, polled every sim tick) — but `rugiar` is a
  **separate OS process** with its own `TrainingManager`; it writes
  `policies/<name>/` to disk same as always, but the running server has no way to
  know that happened until told.
- The Refresh button calls `ControlService.refresh_local_policies()`
  (`legged_gym/control/service.py`), which re-runs the same disk scan
  (`TrainingManager.discover_local_policies()`) filtered to names not already
  loaded, loads each new one into the running sim, and registers it as a
  Clone-from source too — same effect as a restart, without dropping the live
  viewer/sim connection. Safe to click any time; a policy for a different task
  than the running server (obs/action-space mismatch) is skipped, not loaded
  broken.
- Still needs a full restart for anything Refresh can't do: picking up **code**
  changes (this repo's own `.py` files), or a policy whose task the server wasn't
  launched with `--policy`/`--task` awareness of at all.

## Common recipes

The `--max_iterations` values below (800-1200) illustrate WHICH FLAGS to combine
for each goal, not a proven-sufficient training budget — per the "Known open
problem" banner and "Scale matters" above, treat every one of these as a starting
chunk to chain further (via `--from_policy`) and verify by watching, not a
finished recipe to trust as-is.

```bash
# crouch instead of walk: zero velocity commands, target a lower base height
# (a policy with zero-velocity commands is a much easier target than full
# locomotion — this one may need proportionally fewer iterations to look right,
# but "look right" still means watched, not assumed, per the banner above)
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

- **Before declaring ANY checkpoint good, watch it** — this is the single most
  important entry in this list, see the "Known open problem" banner at the top.
  `rugiar`/the web UI never do this for you. For a checkpoint that still has its
  raw training run under `logs/<task>/<run>/` (not yet cleaned up):
  ```bash
  python legged_gym/scripts/play.py --task=g1 --load_run=<run> --ckpt=<N> \
      --viewer=viser --viser_port=9006
  ```
  `<run>` is the `Aug09_...` -style directory name under `logs/g1/` (the raw
  checkpoint's source; a policy's `meta.json` doesn't store this path directly —
  it's whatever `logs/<task>/` directory has a timestamp matching when that
  policy finished, or check `job.command`/`source_log_dir` in an older meta.json
  that still has one). `--ckpt=-1` (or omit) plays the latest/final checkpoint in
  that run instead of a specific numbered one. This opens a live browser view —
  actually look at it walk (or not) before trusting the reward number next to it.
  For an already-`policies/<name>/`-registered checkpoint whose raw `logs/`
  directory got cleaned up, there's currently no direct "view this policies/
  folder" path in `play.py` — load it into a running control web instead
  (`--policy <name>:policies/<name>/checkpoint.pt` on `swap_experiment.py`, or the
  Refresh button per "Picking up policies trained outside the web") and drive it
  with an actual velocity command.
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
- **`RuntimeError: Attempting to deserialize object on a CUDA device but
  torch.cuda.is_available() is False`** on a **local** `--from_policy` run →
  the base policy's `train_checkpoint.pt` was saved on a CUDA device (check
  `policies/<name>/meta.json`'s `"simulator"` — `isaacgym` almost always
  means it was trained via `--backend kaggle`, GPU), and this machine has no
  GPU. `torch.load()` can't remap that on its own. Fix: either pick a base
  policy that was itself trained `--backend local` on THIS machine (`meta.json`
  has `"simulator": null`/`"genesis"`, `num_envs` in the tens not thousands),
  or fine-tune with `--backend kaggle` too (uploads the base checkpoint and
  continues training on a GPU, avoiding the CPU reload entirely) — see
  `rugiar train --list_policies` and check each candidate's `meta.json`
  before picking a base for a local run.
- **`no Kaggle credentials found at ~/.kaggle/kaggle.json`** → follow
  "Setting up Kaggle for cloud training" above.
- Job failed with a subprocess exit code → the error message names the exact
  log file (`logs/_web_training/<job_id>.log`) — `rugiar` already streamed it
  live, but it's still there to re-read.
- **`episode_length` stuck flat/low across several fine-tune chunks in a row**
  (checked its trend across ≥2-3 chunks' `meta.json`, not just one) → don't push
  further down the same recipe (e.g. don't proceed to a height-target change on top
  of an already-unstable push-adapted policy) — it's usually an undersized
  `num_envs`/iteration budget for the distribution shift just introduced, not
  something more iterations at the same tiny scale will fix. Move to
  `--backend kaggle --num_envs 4096` instead of throwing more low-`num_envs`
  iterations at it.

## Sources

Community/upstream references consulted while writing the guidance above (num_envs,
iteration budget, push-curriculum timing) — re-check these if upstream configs
change:

- [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) — the
  original ETH Zurich RSL project this fork and `rsl_rl` descend from.
- [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
  — the direct upstream for this repo's G1 task; its
  [`g1_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py)
  is the source for `push_robots = True` from the start and
  `max_iterations = 10000`.
- General curriculum-domain-randomization practice (gradual difficulty ramp to
  avoid early-learning collapse, contrasted above with the G1-specific reference
  config's "pushes on from the start") — see the curriculum-learning literature
  survey results for legged robots (searched August 2026); no single canonical
  source, treat as background context rather than a specific citation.
