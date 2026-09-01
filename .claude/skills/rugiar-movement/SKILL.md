---
name: rugiar-movement
description: Running/driving a RUgiar robot (sim today, real G1 once wired up) with `rugiar_driver.py` — policy switching, pause/restart, E-STOP, manual velocity commands, over a WebSocket control protocol any client (the built-in web UI, a home-made joystick controller, or the `rugiar_mcp` MCP server) can speak. Also covers reviewing a single checkpoint live with `play.py` and confirming a policy actually walks before trusting it. Use whenever the user wants to connect to or drive a robot (sim or `--real`), switch policies live, understand/build a controller against the control protocol, or watch a checkpoint to judge gait quality. Does NOT cover training a policy (see rugiar-training) or starting/managing the rugiar_mcp server or control web as infrastructure (see rugiar-management) — this skill is the driving/watching half only.
allowed-tools: Bash(export SIMULATOR=*) Bash(python legged_gym/scripts/rugiar_driver.py:*) Bash(.venv/bin/python legged_gym/scripts/rugiar_driver.py:*) Bash(python legged_gym/scripts/rugiar_driver_target.py:*) Bash(.venv/bin/python legged_gym/scripts/rugiar_driver_target.py:*) Bash(python legged_gym/scripts/play.py:*) Bash(.venv/bin/python legged_gym/scripts/play.py:*)
---

# RUgiar movement — driving a robot with `rugiar_driver.py`

This skill is one of three optional splits of the original `rugiar` skill
(the other two: `rugiar-training` for `rugiar train`/`fuse`/`distill`, and
`rugiar-management` for `rugiar_mcp`/control-web/server-lifecycle concerns).
The original `rugiar` skill still exists and covers all three combined — use
this narrower one when the task is specifically about driving/watching a
robot, not training or server management.

For the system-wide picture beyond this skill's driver scope — how Training,
Policy Operations, Control, the Web UI, the CLI, the Robot Driver, and
Third-Party Integrations fit together — see
**`legged_gym/control/ARCHITECTURE.md`**, not this file.

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
rugiar_driver.py --help`** (needs `SIMULATOR` set first — see "Prerequisite"
below). Snapshot as of this writing, so you don't have to run it just to see
what exists:

```
usage: rugiar_driver.py [-h] --policy POLICY_SPECS [--active ACTIVE]
                          [--ramp_ticks RAMP_TICKS] [--headless]
                          [--viser_port VISER_PORT] [--speed SPEED]
                          [--control_port CONTROL_PORT] [--scenario {ball,default,race}]
                          [--scenario-option KEY=VALUE] [--real]
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
--scenario {ball,default,race}
                        which named scenario's props/web-UI config to use (Genesis only).
                        Defaults to 'default' (full admin: no props, every web-UI control
                        visible). 'ball': a physics ball prop. 'race': a start/finish line
                        and crash-mat track (see legged_gym/utils/scenarios.py).
--scenario-option KEY=VALUE
                        override one of the scenario's default options (repeatable),
                        e.g. --scenario-option track_length=10.
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

## Prerequisite: SIMULATOR must be set

`rugiar_driver.py` imports `legged_gym`, which **refuses to import at all**
unless `SIMULATOR` is set:

```bash
export SIMULATOR=genesis    # or isaaclab
```

### Two driver scripts, one per task family — and the Family panel

`rugiar_driver.py` (this section) drives the **"g1" walking family** — every
`g1`-task policy (`stable_home_made_*`, `walk_gpu_c4*`, etc.). A separate,
largely-duplicated sibling script, `rugiar_driver_target.py`, drives the
**"target-aware" family** (`g1_target` and future siblings whose config sets
`cfg.rewards.target_aware = True`) — same flags/behavior, plus a per-tick
step that feeds the live `--scenario ball` position into the running task's obs.
Each registered task is treated as its own **experiment**, deliberately kept
architecturally independent rather than unified into one policy — see
`legged_gym/scripts/rugiar_driver.py`'s module docstring for the reasoning.

The control web's **Family** panel (above Policies) lets an operator switch
which task/driver is running without a terminal: it calls
`ControlService.switch_family(task)`, which self-relaunches the correct
script for that task's family (picking `rugiar_driver.py` vs
`rugiar_driver_target.py` via `_script_for_task()`) on the same port — Genesis
can't rebuild its scene in-process, so this is a ~15-20s process handoff, not
an instant switch; the browser reconnects on its own. Only tasks with at
least one local trained policy are offered.

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
UI or to build their own controller against the same robot.

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

For controlling the robot via MCP tools instead of raw WebSocket (typed
tools like `set_velocity`/`switch_policy`/`get_status`), see the
`rugiar-management` skill's `rugiar_mcp` section — it's a thin client on top
of this same protocol and needs a driver like the one above already running.

## Reviewing a specific checkpoint before trusting it

`play.py` opens any single checkpoint live in the browser (not the control
web — no policy switching, just watch one policy):
```bash
python legged_gym/scripts/play.py --task=g1 --load_run=<run> --ckpt=<N> \
    --viewer=viser --viser_port=9006
```
`<run>` is the `Aug09_...`-style directory name under `logs/<task>/` (the raw
checkpoint's source — a `policies/<name>/` folder that already got cleaned
up has no direct "view this" path in `play.py`; load it into a running
`rugiar_driver.py` instead and drive it with an actual velocity command).
`--ckpt=-1` (or omit) plays the latest/final checkpoint in that run.

## How to know if a checkpoint actually walks (don't trust the numbers)

**The single most important rule: `Mean reward` and `Mean episode length` are
not evidence a policy walks, no matter how good they look.** This has been
proven wrong here more than once, in both directions:

- A checkpoint can post a strong reward/episode_length and still take zero
  steps under a full forward command, or fall repeatedly in a way the
  per-iteration average doesn't make obvious (a spiking reward from short
  high-value bursts before each fall can look identical to steady progress).
- Two checkpoints can score in the same range — one confirmed to walk, the
  other confirmed not to — with no way to tell which is which from the
  numbers alone. In one confirmed case here, the checkpoint that actually
  walked had a *lower* reward and episode_length than an earlier checkpoint
  from the same lineage that took no steps at all.

**The only reliable check is watching it directly** under a commanded
velocity — `play.py --viewer=viser` for a single checkpoint, or load it into
`rugiar_driver.py` and drive it with an actual velocity command. Budget for
this before trusting any checkpoint, especially before deleting the
`logs/<task>/<run>/` directory it came from.

**A cheaper pre-filter, not a replacement for watching:** `legged_robot.py`
also logs a diagnostic-only metric, `actual_lin_vel_x` — the real
time-averaged forward velocity in m/s. It shows up as `Mean episode
rew_actual_lin_vel_x` in the training log and a checkpoint's `meta.json`
metrics. A near-zero average forward velocity is a strong signal a
checkpoint never moved forward at all — cheap to check before spending a
viewer session on it. But a nonzero value can also come from a
fall-and-slide, not just real locomotion — it narrows down what's worth
watching, it doesn't replace watching it.
