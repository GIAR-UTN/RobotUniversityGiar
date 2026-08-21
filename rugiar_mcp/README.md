# RobotUniversityGiar MCP Server

MCP server that talks to `rugiar_driver.py`'s ControlServer via WebSocket and HTTP.

## Install

```bash
# optional mcp extras
pip install -e ".[mcp]"
```

Copy `.env.example` to `.env` and set `CONTROL_HOST`, `CONTROL_PORT`, `CONTROL_TOKEN`.

## Run

```bash
python -m rugiar_mcp.server
```

## Transport

The server supports three transports:

- **`stdio`** (default) — for local MCP clients that launch the server as a subprocess
- **`streamable-http`** — for remote/network clients (e.g., Hermes). Uses a single `/mcp` endpoint.
- **`sse`** — legacy two-endpoint transport (not recommended for new integrations)

To use **streamable-http** (recommended for Hermes and other external agents):

```bash
export MCP_TRANSPORT=streamable-http
export MCP_PORT=9014
python -m rugiar_mcp.server
```

In Docker Compose:
```yaml
environment:
  - MCP_TRANSPORT=streamable-http
  - MCP_PORT=9014
```

### Hermes Configuration

Point Hermes to the **streamable-http** endpoint:

```json
{
  "mcpServers": {
    "rugiar": {
      "type": "streamable-http",
      "url": "http://<host>:9014/mcp"
    }
  }
}
```

**Do not use `/sse`** — the streamable-http transport uses a single `/mcp` endpoint for both requests and responses.

## Tools

### Policy Management

| Tool | Description |
|------|-------------|
| `list_policies` | List every loaded policy and show which one is currently active |
| `switch_policy(name)` | Change the robot's active behavior policy by name. Must match a known loaded policy |

### Status & Telemetry

| Tool | Description |
|------|-------------|
| `get_status` | Full system status snapshot. Use this for an overview of what the robot is doing right now |
| `get_telemetry` | Raw sensor-like readings: height, gravity vector, angular velocity, linear velocity. No timestamp — call repeatedly to track changes over time |
| `get_odometry` | Cumulative distance traveled and elapsed time since tracking started. Automatically resets when the robot teleports or is manually restarted. Use this to answer "how far have I moved?" or "for how long?" |
| `get_command_limits` | Maximum speeds the current policy allows, plus the current operator speed limit. Returns trained ranges, effective (scaled) ranges, and the current commanded velocity |

### Control

| Tool | Description |
|------|-------------|
| `set_velocity(vx, vy, yaw, accel?)` | Tell the robot to move at a specific forward, sideways, and turning speed. If `accel` is provided, the command ramps smoothly from the current speed to the target over time = \|delta\| / accel. If omitted, the command takes effect immediately |

### Vision

| Tool | Description |
|------|-------------|
| `get_camera_frame_base64` | Grab a single camera frame as a base64 JPEG, cached 100 ms to avoid hammering the stream |

## Tool Details

### `set_velocity` Acceleration Ramp

```python
set_velocity(vx=0.5, vy=0.0, yaw=0.1, accel=2.0)
```

- `vx`, `vy`: forward and sideways speed in m/s
- `yaw`: turning rate in rad/s
- `accel`: optional acceleration limit in m/s² and rad/s². If omitted, command is immediate.

The ramp runs asynchronously — a new `set_velocity` call cancels any in-progress ramp.

### `get_odometry` for LLMs

This is the **preferred way** for an LLM to track motion over time:

1. Call `get_odometry()` before starting a maneuver → baseline `(0, 0, 0)`
2. Call it again afterward → read `distance_traveled` and `time_elapsed`
3. If the robot falls/resets, odometry auto-resets to `(0, 0, 0)`

Returns `{"available": False}` on real hardware where simulator ground-truth position is unavailable.

## Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTROL_HOST` | `localhost` | Hostname of the ControlServer (the main sim service) |
| `CONTROL_PORT` | `9013` | Port of the ControlServer |
| `CONTROL_TOKEN` | `""` | Authentication token for WebSocket connection |
| `MCP_TRANSPORT` | `stdio` | Transport type: `stdio`, `streamable-http`, or `sse` |
| `MCP_PORT` | `9014` | Port for HTTP transports |
| `CAMERA_CACHE_MS` | `100` | Camera frame cache TTL in milliseconds |
