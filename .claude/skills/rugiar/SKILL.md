---
name: rugiar
description: Front door to the RobotUniversityGiar (RUgiar) system from the command line — training/fine-tuning policies with the `rugiar` CLI, fusing/merging already-trained policies' weights with `rugiar fuse`, AND running/controlling a robot (sim today, real G1 once wired up) with `rugiar_driver.py` — policy switching, pause/restart, E-STOP, manual velocity commands, over a WebSocket control protocol any client (the built-in web UI, a home-made joystick controller) can speak. Use whenever the user wants to train/fine-tune a policy, fuse/merge policies, discover tasks/reward scales/local policies, connect to or drive a robot (sim or `--real`), understand/build a controller against the control protocol, or anything else about using this system day to day.
allowed-tools: Bash(rugiar:*) Bash(.venv/bin/rugiar:*) Bash(pip install:*) Bash(python3 -c:*) Bash(mkdir -p ~/.kaggle:*) Bash(chmod 600 ~/.kaggle/kaggle.json) Bash(mv:*) Bash(export SIMULATOR=*) Bash(python legged_gym/scripts/rugiar_driver.py:*) Bash(.venv/bin/python legged_gym/scripts/rugiar_driver.py:*) Bash(python legged_gym/scripts/rugiar_driver_gaze.py:*) Bash(.venv/bin/python legged_gym/scripts/rugiar_driver_gaze.py:*) Bash(python legged_gym/scripts/play.py:*) Bash(.venv/bin/python legged_gym/scripts/play.py:*)
---

# RUgiar system — training CLI (`rugiar`) and running/control (`rugiar_driver.py`)

This skill covers the two command-line ways someone actually touches this
system day to day: **training** a policy (`rugiar`, this file's original
scope) and **running/driving a robot with one** (`rugiar_driver.py`,
covered in the section right below). If a user's ask is "connect to the
robot," "switch policies live," "let me drive it with a gamepad," or
anything about the WebSocket control protocol, that's the second section —
don't assume it's a training question.

## rugiar_driver.py — running / controlling a robot (sim today, real G1 once wired up)

This is the process behind the control web: it loads one or more trained
policies, exposes policy-switching/pause/restart/E-STOP/velocity commands
over a WebSocket, and drives either the Genesis simulator or (with `--real`)
an actual robot over DDS. Full walkthrough with diagrams: **docs/index.html
§12 "Switching policies live"** (architecture) and **§13 "Talking to the
robot: the control protocol"** (the wire protocol, for building clients).
§9 "Onto the real robot" explains the physical DDS/remote-control gating
sequence `--real` drives through.

**The live, authoritative flag reference is `python legged_gym/scripts/
rugiar_driver.py --help`** (needs `SIMULATOR` set first, same as `rugiar`
— see "Prerequisite" below). Snapshot as of this writing, so you don't have
to run it just to see what exists:

```
usage: rugiar_driver.py [-h] --policy POLICY_SPECS [--active ACTIVE]
                          [--ramp_ticks RAMP_TICKS] [--headless]
                          [--viser_port VISER_PORT] [--speed SPEED]
                          [--control_port CONTROL_PORT] [--ball] [--real]
                          [--net_interface NET_INTERFACE]
                          [--robot_config ROBOT_CONFIG] [--token TOKEN]

--policy POLICY_SPECS   name:/path/to/policy.pt — repeatable, optional: any local
                        policies/<name>/ folder trained for --task is auto-discovered
                        regardless (this is only for policies not registered that way).
--task TASK             registered task this server's scene is built for (default: 'g1').
                        All --policy specs and auto-discovered ones must be for this task.
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
--camera                stream a robot-POV RGB camera feed to the control web
                        (Genesis only, needs cfg.sensor.add_rgb_camera support)
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

### Two driver scripts, one per task family — and the Family panel

`rugiar_driver.py` (this section) drives the **"g1" walking family** — every
`g1`-task policy (`stable_home_made_*`, `walk_gpu_c4*`, etc.). A separate,
largely-duplicated sibling script, `rugiar_driver_gaze.py`, drives the
**"target-aware" family** (`g1_gaze` and future siblings whose config sets
`cfg.rewards.target_aware = True`) — same flags/behavior, plus a per-tick
step that feeds the live `--ball` position into the running task's obs.
Each registered task is treated as its own **experiment**, deliberately kept
architecturally independent rather than unified into one policy — see
`legged_gym/scripts/rugiar_driver.py`'s module docstring for the reasoning.

The control web's **Family** panel (above Policies) lets an operator switch
which task/driver is running without a terminal: it calls
`ControlService.switch_family(task)`, which self-relaunches the correct
script for that task's family (picking `rugiar_driver.py` vs
`rugiar_driver_gaze.py` via `_script_for_task()`) on the same port — Genesis
can't rebuild its scene in-process (see `training.py`'s module docstring), so
this is a ~15-20s process handoff, not an instant switch; the browser
reconnects on its own. Only tasks with at least one local trained policy are
offered. Switching families is for changing which *experiment* you're
looking at, not a live in-operation mode change — see the "Family selector"
plan for the fuller reasoning.

### Quick start — sim, with the control web

```bash
export SIMULATOR=genesis
python legged_gym/scripts/rugiar_driver.py \
    --policy <policy_a>:policies/<policy_a>/checkpoint.pt \
    --policy <policy_b>:policies/<policy_b>/checkpoint.pt \
    --active <policy_a> --control_port 9013
# open http://localhost:9013 — switch policies, pause/restart, E-STOP,
# drive velocity commands live; :9006 is the raw 3D view (printed at startup)
```

### Connecting to a real robot

```bash
python legged_gym/scripts/rugiar_driver.py \
    --policy <policy_a>:policies/<policy_a>/checkpoint.pt \
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
See "How to know if a checkpoint actually walks" below — this is the single
most important habit in this whole skill: a good reward curve is not
evidence a policy walks.

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
part `--help` can't tell you.

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
# community/upstream standard; see "Scale matters" below before treating
# any short run's output as a trustworthy result
rugiar train --task g1 --name cloud_walk --backend kaggle --num_envs 4096 --max_iterations 1500
```

Ctrl-C during a run terminates the training subprocess and leaves the policy
**unregistered** (nothing gets written to `./policies/`) — safe to interrupt.

## Fusing policies (`rugiar fuse`)

Merges 2+ already-trained local policies' weights into a new policy — no
further training. Same engine as the control web's "⚛ Fuse policies…" panel
(right under "+ New policy…" — `legged_gym/control/fusion.py` +
`TrainingManager.fuse_policies()`), driven from the CLI:

```bash
rugiar fuse --list_fusion_methods                          # every method this build knows about
rugiar fuse --list_policies                                # same list as `train`'s —
                                                             # fine-tunable == fusable

# uniform 2-way weighted average
rugiar fuse --policies stable_home_made_3 stable_home_made_4 --name blended

# weighted 3-way merge, favoring the first source 2:1:1
rugiar fuse --policies base_a base_b base_c --weights 2 1 1 --name blended_v2

# permutation-aligned merge instead of naive averaging (works for LSTM/GRU too)
rugiar fuse --policies base_a base_b --method git_rebasin --name blended_rebasin
```

`--list_fusion_methods` output shape (one line per method):

```
Fusion methods:
  weighted_average (available): Weighted average — Elementwise weighted sum of matching weights across every source policy...
  git_rebasin (available): Git Re-Basin (permutation alignment) — Solves for the hidden-unit permutation...
```

The result is registered as a normal `./policies/<name>/` — fine-tunable via
`train --from_policy` and fusable again, same as anything trained through
this UI. Every source needs a `train_checkpoint.pt` (same requirement
`--from_policy` has — `--list_policies`'s `fine-tunable=yes/no` doubles as
`fusable=yes/no`), and all sources must be architecturally compatible (same
obs/action dims, hidden dims, recurrent-or-not) — checked directly from each
`train_checkpoint.pt`'s tensor shapes, no live sim/task config needed. A
mismatched **task** across sources is only a warning printed to stderr, not
a hard stop, since two different tasks can share an identical network shape.

**Method: `weighted_average`** (default) — a.k.a. model soup / SWA-style
interpolation, an elementwise weighted sum of matching weights. Reasonable
for closely related checkpoints (a fine-tune lineage, or same-seed
variants) — no guarantee for independently-trained policies, since their
hidden units aren't necessarily aligned (permutation symmetry: two networks
trained from different random inits can converge to functionally-equivalent
but internally *permuted* representations, and naively averaging permuted
weights usually lands between the two minima rather than near either).

**Method: `git_rebasin`** (`--method git_rebasin`) — solves for the
hidden-unit permutation that best aligns every non-reference source to the
first one *before* averaging (Ainsworth et al., 2022's weight-matching
algorithm, `fusion.rebasin_align()`), so the merge lands inside rather than
between the sources' loss basins. Works for both plain and recurrent
(LSTM/GRU) actor/critic policies — an RNN's own per-gate hidden-unit
permutation symmetry is aligned too (all gates of a layer share one
permutation, since it's the same cell state being gated), then chained into
the downstream MLP's own alignment. Always double-check `fusion.py`'s
`FUSION_METHODS` registry (or `--list_fusion_methods`) for the current
`available`/scope state before telling a user what's supported — this is a
snapshot, not a guarantee it stays this way forever.

## How to know if a checkpoint actually walks (don't trust the numbers)

**The single most important rule in this skill: `Mean reward` and `Mean
episode length` are not evidence a policy walks, no matter how good they
look.** This has been proven wrong here more than once, in both directions:

- A checkpoint can post a strong reward/episode_length and still take zero
  steps under a full forward command, or fall repeatedly in a way the
  per-iteration average doesn't make obvious (a spiking reward from short
  high-value bursts before each fall can look identical to steady progress).
- Two checkpoints can score in the same range — one confirmed to walk, the
  other confirmed not to — with no way to tell which is which from the
  numbers alone. In one confirmed case here, the checkpoint that actually
  walked had a *lower* reward and episode_length than an earlier checkpoint
  from the same lineage that took no steps at all.
- A reward curve plateauing partway through a training chunk is not
  evidence that the underlying gait had already reached its final quality
  at that point — the metric and the real behavior can move independently.

**The only reliable check is watching it directly** under a commanded
velocity — `play.py --viewer=viser` for a single checkpoint (see
"Reviewing a specific checkpoint" above), or load it into `rugiar_driver.py`
and drive it with an actual velocity command. Budget for this before trusting
any checkpoint, especially before deleting the `logs/<task>/<run>/` directory
it came from.

**A cheaper pre-filter, not a replacement for watching:** `legged_robot.py`
also logs a diagnostic-only metric, `actual_lin_vel_x` — the real
time-averaged forward velocity in m/s (not the `tracking_lin_vel` reward,
which is a similarity score against the commanded velocity, not the raw
value). It shows up automatically as `Mean episode rew_actual_lin_vel_x` in
the training log and in a checkpoint's `meta.json` metrics / the control
web's chart — no parser changes needed, and it has zero effect on training
(never added to the actual reward total). A near-zero average forward
velocity is a strong signal a checkpoint never moved forward at all — cheap
to check before spending a viewer session on it. But a nonzero value can
also come from a fall-and-slide, not just real locomotion — it narrows down
what's worth watching, it doesn't replace watching it.

## Scale matters: `num_envs` and iteration budget aren't cosmetic knobs

Two runs with the "same" reward function and roughly the same flags can
produce wildly different policies depending on `num_envs` and how many
iterations you actually give it. Two separate lessons, don't conflate them:

1. **`num_envs` changes what an "iteration" means, it isn't just a speed
   knob.** `num_envs=4096` is the documented community standard for Isaac
   Gym / rsl_rl humanoid locomotion (confirmed against
   [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)
   and [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym),
   the exact upstream this fork descends from) — each iteration at 4096
   envs collects ~64x more environment experience per PPO update than at
   64 envs, so iteration counts are not comparable across different
   `num_envs`. A run here with 10x more iterations than another, but at
   `num_envs=64` instead of `4096`, still scored worse.
2. **Even at the right `num_envs`, there's a real floor below which nothing
   walks, and it's higher than a quick smoke test.** Confirmed here at
   `num_envs=4096`: ~300 iterations is not enough for any directed
   locomotion under command; commanded walking started to appear somewhere
   between ~600 and ~900 iterations for one lineage; by ~1200 iterations a
   real (if visually rough) walking gait was consistently present. This
   doesn't mean 1200 is a universal floor — treat it as "several hundred is
   confirmed not enough, low thousands is where it becomes plausible,"
   not an exact number to target blindly.

**Practical rule: `--num_envs 64` (rugiar's local default) is for
smoke-testing the CLI/flags/reward wiring cheaply on a laptop CPU, never for
a policy you actually want to be good.** For anything meant to walk/balance
for real, use `--backend kaggle` with `--num_envs 4096`. Budgeting
iterations: treat a few hundred as enough only to prove the pipeline runs,
low thousands (1200-2000) as a realistic first checkpoint worth watching,
and the full 5000-10000 (see "What to actually expect" below) as the target
for gait *quality*, not "walks at all." Iteration cost is real regardless of
target: at this repo's own measured ~5s/iteration at `num_envs=4096`, 1200
iterations is ~1-1.5 hours, 5000 is **~7 hours**, and 10000 is **~14 hours**
of GPU compute — a meaningful fraction of Kaggle's free ~30h/week quota.
Budget accordingly.

## What to actually expect as iteration count grows

This section exists so nobody has to re-investigate community references
from scratch every time a `g1` run looks rough — confirmed by comparing this
repo's `g1_config.py` directly against `unitreerobotics/unitree_rl_gym`'s and
researching community reports:

- **~300 iterations (`num_envs=4096`): no directed locomotion at all is
  normal.** Don't read anything into a checkpoint this young not walking.
- **~600-900 iterations: walking can start to emerge**, per the confirmed
  window above.
- **~900-1500ish iterations: an ugly, "bunny hop"-style gait (repeated hops
  instead of an alternating stride) is an EXPECTED intermediate stage, not a
  sign of a broken config or bad reward shaping.** This is a documented
  pattern in bipedal-locomotion RL literature — a temporary imbalance
  between tracking reward and air-time/efficiency terms that a well-designed
  reward (this repo's scales already match the `unitree_rl_gym` reference)
  typically resolves with more training, not a redesign.
- **Clean, non-hopping gait: budget toward the full reference target.**
  `unitree_rl_gym` itself targets 10000 iterations for this exact task; a
  comparable IsaacLab G1 project (different, larger MLP+curriculum setup)
  used 16000 starting from an already-functional flat-ground gait. Nobody in
  the community reports a clean gait this early — don't treat "still ugly at
  1200-2000 iterations" as evidence something needs fixing before continuing
  the same recipe further via `--from_policy` chunks.

## `entropy_coef`: a real lever when gait quality plateaus

This repo's own `g1_config.py` / `legged_robot_config.py` already default
`entropy_coef` to `0.01`, matching the `unitree_rl_gym` community reference
exactly — passing nothing keeps you at the community-standard exploration
level. Don't assume you need to lower it; a lower value here was tried once
early on and is not the reference setting.

**If a checkpoint's gait *quality* plateaus across several chunks** (not
just the reward number, which per "How to know if a checkpoint actually
walks" above can plateau or move for unrelated reasons), **going above the
0.01 default is a reasonable first experiment, ahead of any reward
redesign.** Confirmed here: bumping to `0.02` from a plateaued, "bunny hop"
checkpoint produced a large, directly-observed gait improvement — described
on watching it as "almost able to run," a clear step up, though still
visibly asymmetric. The reward/episode_length numbers for that checkpoint
were unremarkable compared to the one before it — once again, only watching
it revealed the jump.

**Don't assume "more entropy = more improvement" scales linearly, though.**
A follow-up test doubling again to `0.04` from the same improved checkpoint
did not help further — this time the reward/episode_length numbers *and*
direct observation agreed it was flat-to-worse (lower reward, lower
episode_length, higher `noise_std`, no visible further improvement). One
data point isn't enough to call `0.04` a firm ceiling for every lineage, but
it's a caution against assuming a bigger jump automatically means a bigger
improvement.

Practical guidance for trying this:
- **Always branch to a new name** (`--from_policy <base> --name <base>_hient`),
  never overwrite the last good checkpoint — higher entropy can just as
  easily make a gait more erratic ("epilepsia": non-productive, jittery
  high-frequency motion) as it can help it escape a plateau.
- **Watch the result before trusting either outcome**, same rule as always.
- **Isolate the variable.** A prior attempt at high entropy on this repo's
  local Mac/CPU/`num_envs=64` setup produced exactly that kind of chaotic
  behavior — but that attempt changed entropy AND used too-few envs/
  iterations at the same time (see "Scale matters" above), so it was never a
  clean test of entropy alone. Test entropy changes at the reference
  `num_envs=4096`, changing nothing else from a known-good base.

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
rugiar order --set <policy_a> <policy_b>               # pin these first
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
- Once a later chunk successfully continues from an intermediate `_c1`/`_c2`-style
  policy, that intermediate is safe to delete — the final one's `train_checkpoint.pt`
  already contains that lineage. Worth keeping them around instead if you want to
  compare checkpoints across the chunk boundaries later (e.g. to find where a
  behavior first appeared) — that comparison is free since the files are already
  on disk, it just costs some local storage.
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
`play.py --viewer=viser` before trusting a checkpoint.

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
  enough evidence a base "already walks well."** A checkpoint here scored near
  the best externally-confirmed reference on `episode_length` alone and was
  still reported not to take a single step under a full-forward command — see
  "How to know if a checkpoint actually walks" above. Confirm a base actually
  walks by watching it (`play.py --viewer=viser`) before spending a fine-tune
  run on top of it, not just by reading its `meta.json`.
- **Move the target gently, not aggressively** — a few % off whatever height the
  base already trained at (check its `meta.json` command, or the task's own
  default), not a guessed round number far away from it.
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
# chain several of these via --from_policy toward the thousands before expecting a
# real gait, per "Scale matters" and "What to actually expect" above.
rugiar train --task g1 --name walk_c1 --backend kaggle --num_envs 4096 \
    --max_iterations 1500 --entropy_coef 0.01
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

- `rugiar_driver.py` (the process behind the control web) scans `./policies/`
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
- If a server is already running the exact process you want (same task, same
  policies you're already comparing), **restart that one instead of starting a
  second server on a different port** — a fresh parallel instance won't have
  whatever was already loaded into the running one, and now there are two
  processes to keep track of instead of one.

## Common recipes

The `--max_iterations` values below (500-1200) illustrate WHICH FLAGS to combine
for each goal, not a proven-sufficient training budget — per "Scale matters"
above, treat every one of these as a starting chunk to chain further (via
`--from_policy`) and verify by watching, not a finished recipe to trust as-is.

```bash
# crouch instead of walk: zero velocity commands, target a lower base height
# (a policy with zero-velocity commands is a much easier target than full
# locomotion — this one may need proportionally fewer iterations to look right,
# but "look right" still means watched, not assumed)
rugiar train --task g1 --name crouch --max_iterations 1000 \
    --cmd_vx_range 0 0 --cmd_vy_range 0 0 --cmd_yaw_range 0 0 \
    --base_height_target 0.45

# cautious gait: penalize torque/joint-velocity harder from an existing base
rugiar train --task g1 --name cautious --from_policy <existing_policy> \
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
  important entry in this list, see "How to know if a checkpoint actually walks"
  above. `rugiar`/the web UI never do this for you. For a checkpoint that still
  has its raw training run under `logs/<task>/<run>/` (not yet cleaned up):
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
  (`--policy <name>:policies/<name>/checkpoint.pt` on `rugiar_driver.py`, or the
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
- **Before starting any new `rugiar_driver.py`/web server, check what's
  already running** (e.g. `las ports audit` if this environment uses the Local
  Agent Society port registry) and restart an existing relevant session instead
  of launching a parallel one on a different port — see "Picking up policies
  trained outside the web" above.

## Sources

Community/upstream references consulted while writing the guidance above (num_envs,
iteration budget, push-curriculum timing, entropy_coef) — re-check these if
upstream configs change:

- [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) — the
  original ETH Zurich RSL project this fork and `rsl_rl` descend from.
- [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
  — the direct upstream for this repo's G1 task; its
  [`g1_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py)
  and
  [`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py)
  are the source for `push_robots = True` from the start, `max_iterations = 10000`,
  and `entropy_coef = 0.01`.
- General curriculum-domain-randomization practice (gradual difficulty ramp to
  avoid early-learning collapse, contrasted above with the G1-specific reference
  config's "pushes on from the start") — see the curriculum-learning literature
  survey results for legged robots; no single canonical source, treat as
  background context rather than a specific citation.
- Bipedal-locomotion RL literature on the "bunny hop" intermediate-stage pattern
  (temporary tracking-vs-air-time reward imbalance) and comparable IsaacLab G1
  training budgets — background context, not a single canonical source.
