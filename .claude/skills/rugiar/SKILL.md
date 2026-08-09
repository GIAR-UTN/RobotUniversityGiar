---
name: rugiar
description: Front door to the RobotUniversityGiar (RUgiar) system from the command line — training/fine-tuning policies with the `rugiar` CLI, AND running/controlling a robot (sim today, real G1 once wired up) with `swap_experiment.py` — policy switching, pause/restart, E-STOP, manual velocity commands, over a WebSocket control protocol any client (the built-in web UI, a home-made joystick controller) can speak. Use whenever the user wants to train/fine-tune a policy, discover tasks/reward scales/local policies, connect to or drive a robot (sim or `--real`), understand/build a controller against the control protocol, or anything else about using this system day to day.
allowed-tools: Bash(rugiar:*) Bash(.venv/bin/rugiar:*) Bash(pip install:*) Bash(python3 -c:*) Bash(mkdir -p ~/.kaggle:*) Bash(chmod 600 ~/.kaggle/kaggle.json) Bash(mv:*) Bash(export SIMULATOR=*) Bash(python legged_gym/scripts/swap_experiment.py:*) Bash(.venv/bin/python legged_gym/scripts/swap_experiment.py:*) Bash(python legged_gym/scripts/play.py:*) Bash(.venv/bin/python legged_gym/scripts/play.py:*)
---

# RUgiar system — training CLI (`rugiar`) and running/control (`swap_experiment.py`)

This skill covers the two command-line ways someone actually touches this
system day to day: **training** a policy (`rugiar`, this file's original
scope) and **running/driving a robot with one** (`swap_experiment.py`,
covered in the new section right below). If a user's ask is "connect to the
robot," "switch policies live," "let me drive it with a gamepad," or
anything about the WebSocket control protocol, that's the second section —
don't assume it's a training question.

## swap_experiment.py — running / controlling a robot (sim today, real G1 once wired up)

This is the process behind the control web: it loads one or more trained
policies, exposes policy-switching/pause/restart/E-STOP/velocity commands
over a WebSocket, and drives either the Genesis simulator or (with `--real`)
an actual robot over DDS. Full walkthrough with diagrams: **docs/index.html
§12 "Switching policies live"** (architecture) and **§13 "Talking to the
robot: the control protocol"** (the wire protocol, for building clients).
§9 "Onto the real robot" explains the physical DDS/remote-control gating
sequence `--real` drives through.

**The live, authoritative flag reference is `python legged_gym/scripts/
swap_experiment.py --help`** (needs `SIMULATOR` set first, same as `rugiar`
— see "Prerequisite" below). Snapshot as of this writing, so you don't have
to run it just to see what exists:

```
usage: swap_experiment.py [-h] --policy POLICY_SPECS [--active ACTIVE]
                          [--ramp_ticks RAMP_TICKS] [--headless]
                          [--viser_port VISER_PORT] [--speed SPEED]
                          [--control_port CONTROL_PORT] [--ball] [--real]
                          [--net_interface NET_INTERFACE]
                          [--robot_config ROBOT_CONFIG] [--token TOKEN]

--policy POLICY_SPECS   name:/path/to/policy.pt — repeatable, first one is the default active
--active ACTIVE         which --policy name starts active (default: first one given)
--ramp_ticks N          control ticks to cross-fade over on a switch
--headless              no viewer — runs a scripted smoke test (switch once, then exit).
                        Mutually exclusive with --real (see below).
--viser_port PORT       raw 3D viewer port (sim only — Genesis's native viewer has a
                        rendering bug on Mac/this asset combo, so viser is what's used)
--speed FLOAT           sim playback speed multiplier (1.0 = real-time 50Hz). Ignored
                        with --real — the real control loop paces itself off the
                        robot's own control_dt.
--control_port PORT     starts a networked ControlServer (JSON-over-WebSocket at /ws)
                        on this port. Unless --headless, also serves the unified
                        control web (policies/pause/restart/E-STOP/velocity panel +
                        Docs tab) at http://localhost:<control_port>/.
--ball                  spawn a physics ball prop next to the robot (Genesis only)
--real                  drive an actual robot over DDS (deploy_real/real_adapter.py::
                        RealAdapter) instead of Genesis. No sim env, no viser.
                        Incompatible with --headless — a real robot's reset() blocks
                        on a human at the physical remote, no unattended smoke test.
--net_interface IFACE   DDS network interface on the robot's onboard computer
                        (e.g. 'eth0', 'enp3s0') — required with --real.
--robot_config PATH     a deploy_real/configs/*.yaml (see g1.yaml) — required with --real.
--token SECRET          shared secret required on every /ws connection
                        (?token=... query param, including the web UI, which forwards
                        its own page's own ?token=...). Strongly recommended whenever
                        --control_port is reachable from more than localhost — which
                        --real always is (the robot's own WiFi/LAN).
```

### Quick start — sim, with the control web

```bash
export SIMULATOR=genesis
python legged_gym/scripts/swap_experiment.py \
    --policy stable:policies/stable.pt \
    --policy crouch:policies/crouch/checkpoint.pt \
    --active stable --control_port 9013
# open http://localhost:9013 — switch policies, pause/restart, E-STOP,
# drive velocity commands live; :9006 is the raw 3D view (printed at startup)
```

### Connecting to a real robot

```bash
python legged_gym/scripts/swap_experiment.py \
    --policy stable:policies/stable.pt \
    --control_port 9013 --token <a-shared-secret> \
    --real --net_interface eth0 --robot_config deploy_real/configs/g1.yaml
```
Runs on the robot's own onboard computer (needs `unitree_sdk2py` installed —
this is untested in this dev environment, no physical robot/SDK here — see
`deploy_real/real_adapter.py`'s module docstring for exactly what's been
verified vs. what still needs re-checking against real hardware before
trusting it). `--token` is what stands between "anyone on the robot's WiFi"
and "can send it commands" — always set it for `--real`. Share
`http://<robot-ip>:9013/?token=<secret>` with whoever needs either the web
UI or to build their own controller against the same robot — see below.

### The control protocol, for building a client (home-made controller, automation, etc.)

Full spec: **docs/index.html §13**. The short version: connect to
`ws://<host>:<port>/ws` (append `?token=...` if the server has one), send
`{"method": "set_command", "params": {"vx": 0.4, "vy": 0.0, "yaw": 0.0}, "id": 1}`
to drive a walking velocity (clamped server-side to the active policy's
trained envelope — send whatever, it won't ask for something unsafe), and
either poll `status` or just listen — the server pushes a `status` message
to every connected client at ~10Hz unprompted, with `backend` ("sim"/"real"),
`capabilities`, `command`, and per-field-labeled `telemetry`. A complete,
minimal reference client (connects, authenticates, streams `set_command`
from a gamepad or a `--demo` scripted loop) is `examples/joystick_controller.py`
— read it before writing a new client from scratch, the connect/send loop
doesn't need to change, only where the (vx, vy, yaw) numbers come from.

### Reviewing a specific checkpoint before trusting it

`play.py` opens any single checkpoint live in the browser (not the control
web — no policy switching, just watch one policy):
```bash
python legged_gym/scripts/play.py --task=g1 --load_run=<run> --ckpt=<N> \
    --viewer=viser --viser_port=9006
```
See "Troubleshooting" below (this is the single most important habit in
this whole skill — a good reward curve is not evidence a policy walks).

---

# rugiar — CLI for creating RobotUniversityGiar policies

## ✅ 2026-08-09 update: this repo's own pipeline CAN produce a walking policy — `walk_gpu_c4`

The "known open problem" below (no self-trained policy ever confirmed to walk) is
now **resolved for the base case** (plain `g1` task, velocity command, no
crouch/push-robustness yet). `walk_gpu_c4` — trained entirely through this repo's
own `rugiar` CLI, `--backend kaggle`, `num_envs=4096`, chunked via `--from_policy`
(`walk_gpu_c1` → `c2` → `c3` → `c4`, ~300 iterations/chunk) to **1200 iterations
total** — was watched directly in the viser viewer under a forward velocity
command and confirmed to walk. Gait is visibly ugly ("bunny hops," not a clean
alternating stride), but it is commanded, directed, repeated forward locomotion,
not a fall or a stand-still — a categorically different result from every prior
self-trained checkpoint in this repo.

This confirms the "Scale matters" hypothesis below **only partially**: `num_envs`
being correct (4096, matching the reference) was necessary, but **10000, or even
5000, iterations was never actually the floor for *some* gait to emerge** — 1200
was enough here, i.e. ~2-3 chunks' worth of GPU time (~1h), not 7-14h. Don't
over-read this either: 1200 iterations produced a working-but-ugly gait, not
confirmed to match the smoothness/robustness of the fully-converged
`kaggle_g1`/`kaggle_g1_mid5000` reference — more iterations from here likely still
improve gait quality, but "walking at all" no longer requires the full budget.
All intermediate chunks (`walk_gpu_c1`..`walk_gpu_c4`) were deliberately kept
un-deleted this run for lineage/comparison — see their individual `meta.json`.

**Still unconfirmed / next open questions, not yet answered by this data point:**
does the gait hold up under push perturbation, direction/speed changes, or a
longer walk (episode_length was measured only via reward-curve numbers, not
watched end-to-end for endurance)? Does continuing past 1200 iterations
*smooth out* the bunny-hop, or is that a local optimum this recipe gets stuck in?
Treat "it walks" as proven; treat "it walks *well*" as still open.

---

## Historical context: the problem as it stood before the above finding

As of 2026-08-09 (superseded above), **every G1 policy in `./policies/` known to
genuinely walk when tested (`stable`, `kaggle_g1`, `kaggle_g1_mid5000`) was
imported from an EXTERNAL, already-fully-converged run of stock `unitree_rl_gym`**
(10000/10000 iterations, trained outside this repo entirely — see their
`meta.json`'s `"trained_via": "kaggle-imported"` / `"source"` fields). Nothing
this repo's own training pipeline had produced — via the control web historically,
or via `rugiar` — had been confirmed by direct observation to walk, including
checkpoints whose reward curve looked good (`stable_home_made`: reward 14.5,
episode_length 578.7 at just 290 iterations). Pushed a command all the way forward
on one of these and it took zero steps and didn't even look crouched on a
crouch-targeted one — that's the symptom that surfaced this.

The leading hypothesis at the time (iteration count, needing the full 5000-10000)
turned out to be **directionally right but overstated** — see the update above:
`num_envs=4096` correctness plus roughly 1200 iterations (chunked) was enough to
get a real, if ugly, walking gait, well short of the full 10000-iteration budget
this section originally assumed was necessary.

**If you're picking this up fresh:** don't trust ANY reward-curve/episode_length
number from this repo's own training as proof of a working gait, no matter how
good it looks — watch it. This exact warning already existed in the repo's own
README ("Reward-curve summaries aren't enough to trust a checkpoint — watch it")
before this was first written, and got ignored anyway; don't repeat that. `walk_gpu_c4`
being confirmed by direct observation, not by its reward number, is the reason it's
trusted above and `stable_home_made` (reward 14.5 at 290 iterations) still isn't.

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
| `scratch_walk_robust_3` (this repo, 2026-08-09) | local CPU | 64 | 750+ | — | **No** |
| `walk_gpu_c1` (this repo, 2026-08-09) | Kaggle GPU | 4096 | 300 | reward 14.58, episode_length 576.8 | **No** |
| `walk_crouch_gpu` (this repo, 2026-08-09, fine-tune of `walk_gpu_c1`) | Kaggle GPU | 4096 | 610 (cumulative) | reward 22.24, episode_length 853.8 | **No** |
| `walk_gpu_c2` (this repo, 2026-08-09, chunked continuation of `walk_gpu_c1`) | Kaggle GPU | 4096 | 600 (cumulative) | reward 16.78, episode_length 947.5 | not watched independently — onset window not yet narrowed below here |
| `walk_gpu_c3` (this repo, 2026-08-09, chunked continuation of `walk_gpu_c2`) | Kaggle GPU | 4096 | 900 (cumulative) | reward 8.21, episode_length 510.1 | **Yes** — confirmed by direct viser observation |
| `walk_gpu_c4` (this repo, 2026-08-09, chunked continuation of `walk_gpu_c3`) | Kaggle GPU | 4096 | **1200** (cumulative, `c1`→`c4`) | reward 24.45, episode_length 984.8 | **Yes, better gait than `c3`** — confirmed by direct viser observation, still an ugly "bunny hop" but genuinely commanded/directed forward locomotion |

**Walking-onset window, narrowed by direct observation: somewhere between 600
iterations (`walk_gpu_c2`, unwatched) and 900 (`walk_gpu_c3`, confirmed walking)**
— NOT between 900 and 1200 as first assumed from `c4`'s own reward curve shape.
Notably `walk_gpu_c3` has a *lower* reward (8.21) and episode_length (510.1) than
`walk_gpu_c1` (14.58 / 576.8, confirmed NOT walking) — reinforcing yet again that
these numbers do not rank-order gait quality or even walking-vs-not. `walk_gpu_c2`
remains unwatched; narrowing the window further (600 vs 900) would need watching
that specific checkpoint, still free since it's already on disk.

**`walk_crouch_gpu` vs `walk_gpu_c4` is the cleanest proof in this repo that the
reward/episode_length numbers cannot be used to infer walking, full stop**:
`walk_crouch_gpu` (610 iterations, reward 22.24, episode_length 853.8) scores
*close to* `walk_gpu_c4` (1200 iterations, reward 24.45, episode_length 984.8) —
same order of magnitude, same general shape — yet one walks and the other
doesn't. Do not use curve plateauing/climbing-speed within a single chunk to
infer "training could have stopped earlier and gotten the same result" without
watching the specific earlier checkpoint — `walk_gpu_c1` (576.8 episode_length,
comparable to `walk_gpu_c3`'s 510.1) already looked like a reasonable curve and
does NOT walk, so a plateaued-looking metric is not evidence the underlying gait
had already emerged by that point either.

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
2. **At the right `num_envs`, 290-310 iterations is not enough, but ~1200 chunked
   iterations was** — see the "2026-08-09 update" banner at the top of this file.
   `walk_gpu_c1`→`walk_gpu_c4` (all `num_envs=4096`, chunked to 1200 total) is the
   first *self-trained* row confirmed to walk by direct observation — well short
   of the 10000-iteration full target the two external reference rows used. Don't
   over-correct the other way either: 290-310 iterations at the *same* `num_envs`
   (`walk_gpu_c1`, `walk_crouch_gpu`) did NOT walk, so somewhere between ~300 and
   ~1200 iterations (at 4096 envs) is where commanded locomotion starts to emerge
   for this recipe — this hasn't been narrowed further. A good-looking reward
   curve alone is still not evidence either way — `walk_gpu_c4` was trusted
   because it was watched, not because reward 24.45 looked better than the
   others'.

**Practical rule: `--num_envs 64` (rugiar's local default) is for smoke-testing the
CLI/flags/reward wiring cheaply on a laptop CPU, never for a policy you actually
want to be good.** For anything meant to walk/balance for real, use
`--backend kaggle` with `--num_envs 4096`. Budgeting iterations: ~300 is
confirmed NOT enough, ~1200 (chunked) is confirmed enough for *some* gait to
emerge (ugly, per `walk_gpu_c4`) — treat 1200-2000 as a realistic first checkpoint
to watch before deciding whether to push toward the full 5000-10000 for gait
*quality*, not "walks at all." Iteration cost is real regardless of the target:
at this repo's own measured ~5.1s/iteration at `num_envs=4096`, 1200 iterations
is ~1-1.5 hours, 5000 is **~7 hours**, and 10000 is **~14 hours** of GPU compute —
a meaningful fraction of Kaggle's free ~30h/week quota. Budget accordingly.

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

## What to actually expect as iteration count grows (don't re-derive this each time)

This section exists so nobody has to re-investigate community references from
scratch every time a `g1` run looks rough — confirmed 2026-08-09 by comparing
this repo's `g1_config.py` directly against `unitreerobotics/unitree_rl_gym`'s
and researching community reports:

- **~300 iterations (`num_envs=4096`): no directed locomotion at all is normal**
  — confirmed here (`walk_gpu_c1`, `walk_crouch_gpu`). Don't read anything into
  a checkpoint this young not walking.
- **~600-900 iterations: walking can start to emerge**, per the confirmed window
  in "Scale matters" above (`walk_gpu_c2` unwatched, `walk_gpu_c3` at 900
  confirmed walking).
- **~900-1500ish iterations: an ugly, "bunny hop"-style gait (repeated hops
  instead of an alternating stride) is an EXPECTED intermediate stage, not a
  sign of a broken config or bad reward shaping.** This is a documented pattern
  in bipedal-locomotion RL literature — a temporary imbalance between
  tracking reward and air-time/efficiency terms that a well-designed reward
  (this repo's scales already match the `unitree_rl_gym` reference) typically
  resolves with more training, not a redesign.
- **Clean, non-hopping gait: budget toward the full reference target.**
  `unitree_rl_gym` itself targets 10000 iterations for this exact task; a
  comparable IsaacLab G1 project (different, larger MLP+curriculum setup) used
  16000 starting from an already-functional flat-ground gait. Nobody in the
  community reports a clean gait this early — don't treat "still ugly at
  1200-2000 iterations" as evidence something needs fixing before continuing
  the same recipe further via `--from_policy` chunks.

**On `entropy_coef` specifically: this repo's own `g1_config.py` and
`legged_robot_config.py` already default to `entropy_coef=0.01`, matching the
`unitree_rl_gym` reference exactly** — passing nothing (or explicitly `0.01`)
keeps you at the community-standard exploration level. Only `walk_gpu_c1` and
`walk_crouch_gpu` used a lower value (0.001-0.002, an experiment from before
this was confirmed) — every chunk since (`walk_gpu_c2` onward) already ran at
the default 0.01. If a gait plateaus in gait *quality* (not just reward number,
which plateaus for unrelated reasons per "Measuring actual velocity" below)
across several chunks, going *above* 0.01 (e.g. 0.02) is a reasonable next
experiment — but do it as a separate named branch off the last good checkpoint
(`--name X_hient --from_policy X`), not by overwriting it, since higher entropy
can just as easily make a gait more erratic ("epilepsia," per a real attempt in
this repo) as it can help it escape a local optimum — watch it before trusting
either outcome, same rule as always.

### ✅ 2026-08-09 confirmed: bumping `entropy_coef` above the reference default gave a big gait improvement

`walk_gpu_c4_hient` (`--from_policy walk_gpu_c4 --entropy_coef 0.02`, one 290-iteration
chunk, `num_envs=4096`) was watched directly and confirmed a **large, qualitative
improvement** over `walk_gpu_c4` itself — described by direct observation as "casi
puede correr" (almost able to run), a clear step up from the "bunny hop," though
still visibly asymmetric left/right. Reward (19.59) and episode_length (973.02)
were near-identical to `c4`'s (24.45 / 984.8) — **once again the numbers didn't
predict this jump, only watching it did.**

This confirms `entropy_coef` above the reference 0.01 is a real, effective lever
for this specific plateau (not just a theoretical "might help, might not" from the
literature search) — worth treating as the first thing to try when a checkpoint's
gait quality has stopped visibly improving across chunks, ahead of any reward
redesign. Don't stop at 0.02 either — it's one point on a curve, not a proven
ceiling; try progressively higher values from the same last-good checkpoint (each
as its own separate branch, e.g. `_hient`, `_hient2`, `_hient_x2`) to map out where
the improvement plateaus or turns into instability ("epilepsia" — erratic,
non-productive high-frequency motion, the failure mode on the other side of this
lever). A prior attempt at high entropy on this repo's local Mac/CPU/`num_envs=64`
setup produced exactly that kind of chaotic behavior — but that attempt confounded
two variables at once (high entropy AND too-few envs/iterations per "Scale
matters" above), so it was never a clean test of entropy alone; the correct way to
test entropy is at the reference `num_envs=4096`, isolating it as the only changed
variable from a known-good base, exactly as done here.

**Follow-up, same day: `0.02` looks like a sweet spot for this lineage, `0.04`
did not help further.** Two more chunks branched from `walk_gpu_c4_hient`:
`walk_gpu_c4_hient2` (same `entropy_coef=0.02`, +290 more iterations —
continuing to improve: reward 20.77, episode_length 981.6, up from `hient`'s
19.59/973.0) vs. `walk_gpu_c4_hient_x2` (`entropy_coef=0.04` instead, +280
iterations from the same `hient` base — reward dropped to 6.66, episode_length
to 831.1, `noise_std` up to 1.12 vs `hient2`'s 0.71). This time the numbers and
the direct observation **agreed**: watched in viser, `hient_x2` did not look
better than `hient`/`hient2` — doubling entropy again past the point that
already fixed the bunny-hop bought nothing further (and the reward/episode_length
drop plus rising noise_std are at least consistent with it starting to erode
rather than help, though a single data point isn't enough to call that a firm
ceiling).

**Practical takeaway: don't assume "more entropy = more improvement" scales
linearly.** The jump that mattered was `0.01 → 0.02` (bunny-hop → much better
gait); `0.02 → 0.04` was flat-to-worse in this lineage. If replicating this
recipe elsewhere, try `0.02` before reaching for anything higher, and treat
higher values as needing their own confirmation rather than an automatic
"more is better" continuation of this same result.

## Measuring actual velocity, not just the tracking-reward proxy

`Mean episode rew_tracking_lin_vel` (and the other `rew_*` lines) measure an
exponential-similarity reward term — how close actual velocity got to the
commanded one — not the raw actual velocity. Two checkpoints with near-identical
`rew_tracking_lin_vel` can have completely different real gaits (see the
`walk_gpu_c1` vs `walk_gpu_c3` case in "Scale matters" above: `c3` walks with a
*lower* score than `c1`, which doesn't walk at all).

As of 2026-08-09, `legged_gym/envs/base/legged_robot.py`'s `compute_reward()` /
`_prepare_reward_function()` also accumulate a **diagnostic-only** term,
`actual_lin_vel_x` — the real time-averaged forward velocity in m/s
(`self.simulator.base_lin_vel[:, 0]`, uniform across all three simulator
backends), reusing the same `episode_sums`/`extras["episode"]` plumbing as every
other reward term (hence it shows up as `Mean episode rew_actual_lin_vel_x` in
the log and in a checkpoint's `meta.json` metrics series / the control web's
chart, automatically, no parser changes needed) — but it is **never added to
`rew_buf`**, so it has zero effect on training, purely observational.

This is still not a substitute for watching a checkpoint directly — a nonzero
average forward velocity could come from a fall-and-slide as easily as a real
gait — but it's a much stronger signal than `rew_tracking_lin_vel` alone, and
worth checking before spending a `play.py`/viser session on a checkpoint that
never moved forward at all.

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
