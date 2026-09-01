---
name: rugiar-management
description: Standing up and managing RUgiar's server-side infrastructure — running the `rugiar_mcp` MCP server on top of an already-running `rugiar_driver.py`, port hygiene (checking what's already running before starting a new server), and picking up policies a training job wrote to disk into an already-running control web (Refresh vs. full restart). Use whenever the user wants to start/stop/restart the control web or MCP server, avoid launching a duplicate server on a new port, or make a freshly trained policy show up in a live session. Does NOT cover training a policy (see rugiar-training) or driving/watching a robot once connected (see rugiar-movement) — this skill is the "keep the site/servers running correctly" half.
allowed-tools: Bash(pip install:*) Bash(python -m rugiar_mcp.server:*) Bash(cp rugiar_mcp/.env.example rugiar_mcp/.env) Bash(las ports audit) Bash(las ports free) Bash(las ports claim:*) Bash(ps aux:*)
---

# RUgiar management — control web / MCP server lifecycle and port hygiene

This skill is one of three optional splits of the original `rugiar` skill
(the other two: `rugiar-training` for `rugiar train`/`fuse`/`distill`, and
`rugiar-movement` for driving/watching a robot with `rugiar_driver.py`). The
original `rugiar` skill still exists and covers all three combined — use
this narrower one when the task is specifically about standing up or
managing the server-side pieces (the control web, the MCP server, which
process is already listening on which port), not training or driving.

## rugiar_mcp — controlling the robot via MCP instead of raw WebSocket

`rugiar_mcp/` (repo root) wraps a *running* `rugiar_driver.py`'s ControlServer
as an MCP server, so an MCP client (this assistant, or any other agent) can
call typed tools instead of hand-rolling the WebSocket protocol. It's a thin
client on top of the same `/ws` connection a hand-rolled controller uses —
**it does not replace `rugiar_driver.py`, it needs one already running** with
`--control_port` reachable at the host/port this MCP server is pointed at
(driving/starting that process is covered by the `rugiar-movement` skill).

### Prerequisite: a `rugiar_driver.py` already running with `--control_port`

```bash
python legged_gym/scripts/rugiar_driver.py --policy ... --control_port 9017
```

`rugiar_mcp` is only a client — check `las ports audit` (or `ps aux | grep
rugiar_driver`) for an existing instance before assuming you need to start
one; reuse it rather than launching a second driver on a different port.

### Running the MCP server

```bash
cd rugiar_mcp/.. # repo root
pip install -e ".[mcp]"        # one-time, if not already installed
cp rugiar_mcp/.env.example rugiar_mcp/.env   # then edit CONTROL_HOST/PORT/TOKEN
python -m rugiar_mcp.server
```

Env vars (`rugiar_mcp/.env` or exported directly) — **`CONTROL_PORT` must
match the target driver's own `--control_port`**, not necessarily the 9013
default:

| Variable | Default | Meaning |
|---|---|---|
| `CONTROL_HOST` | `localhost` | host of the running `rugiar_driver.py`'s ControlServer |
| `CONTROL_PORT` | `9013` | its `--control_port` |
| `CONTROL_TOKEN` | `""` | must match the driver's `--token` if it set one |
| `MCP_TRANSPORT` | `stdio` | `stdio` (local subprocess client) or `streamable-http` (remote clients, e.g. Hermes) |
| `MCP_PORT` | `9014` | port for `streamable-http`/`sse` — **claim it via `las ports claim` before treating it as permanent**, same port hygiene as any other server in this repo |
| `CAMERA_CACHE_MS` | `100` | camera-frame cache TTL |

For `streamable-http`, point a client at `http://<host>:<MCP_PORT>/mcp`
(single endpoint — **not** `/sse`, that's the legacy transport).

### Tools (verified working against a live driver, 2026-08-21)

| Tool | Verified behavior |
|---|---|
| `list_policies` | Returns `active`/`pending`/`ramping`/full `policies` list/`safety_tripped`. Works. |
| `get_status` | Full snapshot incl. `telemetry` (base height, gravity vector, ang/lin vel). Works. |
| `get_telemetry` | Just the `telemetry` sub-object of `get_status`. Works. |
| `get_odometry` | `{"available": true, "distance_traveled", "time_elapsed", "average_speed"}` in sim. Works. |
| `get_command_limits` | Trained vs. effective (speed-limit-scaled) command ranges + current command. Works. |
| `set_velocity(vx, vy, yaw, accel?)` | Immediate mode confirmed (`accel` omitted); `accel` set spawns an async ramp task that cancels any prior ramp. Works. |
| `switch_policy(name)` | An unknown `name` comes back as a tool **error result** (not a protocol-level exception) — client code should check for that rather than assuming success. |
| `get_camera_frame_base64` | **Was broken as shipped — fixed 2026-08-21.** `/camera.mjpg` is an unbounded MJPEG stream that never closes; the original code called `httpx.get()` waiting for a complete response body, which hung forever. Fixed in `rugiar_mcp/server.py` to stream and cut as soon as one full JPEG is seen — now returns in well under a second. If this tool ever hangs again after a future edit, suspect a regression back to non-streaming reads on this endpoint first. |

## Picking up policies trained outside the web (no restart needed)

A policy `rugiar train` just finished training **won't appear in a running
control web** until you either restart the server or hit its **Refresh
button** (circular-arrow icon, top of the Policies panel) — this is
expected, not a bug, and the underlying mechanics are worth understanding if
it seems to not be working:

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

## Port hygiene — before starting any new server

Before starting any new `rugiar_driver.py`/`rugiar_mcp` server, check what's
already running and reuse/restart it instead of launching a parallel
instance on a different port:

```bash
las ports audit             # check conflicts against the registry
ps aux | grep rugiar_driver # or, without LAS, check directly
las ports free               # get a free port, if you really do need a new one
las ports claim "<description>" --port <PORT>
```

This matters doubly for `rugiar_mcp`'s `MCP_PORT` (streamable-http mode) and
for `rugiar_driver.py`'s `--control_port`/`--viser_port` — a stray second
instance on a different port silently has none of the state (loaded
policies, live sim, connected clients) the "real" one already has, and now
there are two processes to keep track of instead of one.

To stop a driver/server cleanly and free memory, send it a graceful
`SIGTERM` (`kill -TERM <pid>`) rather than `-9` — let it shut down its sim
loop and any open WebSocket connections instead of being killed mid-tick.
