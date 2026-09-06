"""MCP server for RobotUniversityGiar control.

Tools:
- list_policies
- switch_policy
- set_velocity
- get_status
- get_telemetry
- get_odometry
- get_command_limits
 - get_camera_frame_base64
 - get_depth_camera_frame_base64
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
from mcp.server import MCPServer

from .control_client import ControlClient, _is_closed

mcp = MCPServer("rugiar-control")
client = ControlClient()

# Camera caches (separate so RGB and depth fetches don't evict each other)
_CAMERA_CACHE_TTL_MS = 100
_camera_cache: dict = {"ts": 0.0, "data": None}
_depth_camera_cache: dict = {"ts": 0.0, "data": None}

# Ramp tasks
_ramp_tasks: dict = {}

DEFAULT_ACCEL = 1.0  # m/s^2 and rad/s^2

async def _ensure_connected():
    if _is_closed(client._ws):
        await client.connect()

async def _fetch_config():
    host = os.getenv("CONTROL_HOST", "localhost")
    port = os.getenv("CONTROL_PORT", "9013")
    url = f"http://{host}:{port}/config"
    async with httpx.AsyncClient(timeout=5.0) as http:
        r = await http.get(url)
        r.raise_for_status()
        return r.json()

@mcp.tool()
async def list_policies() -> dict:
    """List every loaded policy and show which one is currently active."""
    logging.info("[MCP] list_policies called")
    await _ensure_connected()
    status = await client.get_status()
    if not status:
        status = await client.call("status", {})
    policies = status.get("policies", [])
    return {
        "active": status.get("active"),
        "pending": status.get("pending"),
        "ramping": status.get("ramping"),
        "policies": policies,
        "safety_tripped": status.get("safety_tripped"),
    }

@mcp.tool()
async def switch_policy(name: str) -> dict:
    """Change the robot's active behavior policy by name. Must match a known loaded policy."""
    logging.info(f"[MCP] switch_policy called with name={name}")
    await _ensure_connected()
    status = await client.get_status()
    if not status:
        status = await client.call("status", {})
    policies = status.get("policies", [])
    if name not in policies and name != status.get("active"):
        raise ValueError(f"Unknown policy '{name}'. Loaded: {policies}")
    res = await client.call("request_switch", {"name": name})
    return {"requested": True, "name": name, "result": res}

@mcp.tool()
async def get_status() -> dict:
    """Full system status snapshot. Use this for an overview of what the robot is doing right now."""
    logging.info("[MCP] get_status called")
    await _ensure_connected()
    status = await client.get_status()
    if not status:
        status = await client.call("status", {})
    return status

@mcp.tool()
async def get_telemetry() -> dict:
    """Raw sensor-like readings: height, gravity vector, angular velocity, linear velocity.
    No timestamp — call repeatedly to track changes over time."""
    logging.info("[MCP] get_telemetry called")
    status = await client.get_status()
    if not status:
        status = await client.call("status", {})
    return status.get("telemetry", {})

@mcp.tool()
async def get_odometry() -> dict:
    """Cumulative distance traveled and elapsed time since tracking started.
    Automatically resets when the robot teleports or is manually restarted.
    Use this to answer "how far have I moved?" or "for how long?"."""
    logging.info("[MCP] get_odometry called")
    await _ensure_connected()
    odometry = await client.call("get_odometry", {})
    if odometry is None:
        return {"available": False, "reason": "Odometry requires simulator ground-truth position (not available on real hardware)."}
    return {"available": True, **odometry}

@mcp.tool()
async def get_command_limits() -> dict:
    """Maximum speeds the current policy allows, plus the current operator speed limit.
    Returns trained ranges, effective (scaled) ranges, and the current commanded velocity."""
    logging.info("[MCP] get_command_limits called")
    await _ensure_connected()
    status = await client.get_status()
    if not status:
        status = await client.call("status", {})
    cfg = await _fetch_config()
    ranges = cfg.get("command_ranges", {})
    limit = status.get("operator_speed_limit", 1.0)
    def scale(r):
        return [r[0]*limit, r[1]*limit]
    effective = {k: scale(v) for k, v in ranges.items()}
    return {
        "trained_ranges": ranges,
        "operator_speed_limit": limit,
        "effective_ranges": effective,
        "current_command": status.get("command"),
        "active_policy": status.get("active"),
        "backend": status.get("backend"),
    }

@mcp.tool()
async def set_velocity(vx: float, vy: float, yaw: float, accel: Optional[float] = None) -> dict:
    """Tell the robot to move at a specific forward, sideways, and turning speed.
    If accel is provided, the command ramps smoothly from the current speed to the target
    over time = |delta| / accel. If accel is omitted, the command takes effect immediately."""
    logging.info(f"[MCP] set_velocity called with vx={vx}, vy={vy}, yaw={yaw}, accel={accel}")
    await _ensure_connected()
    status = await client.get_status()
    if not status:
        status = await client.call("status", {})
    cur = status.get("command", {"vx":0.0,"vy":0.0,"yaw":0.0})
    cur_vx, cur_vy, cur_yaw = float(cur.get("vx",0)), float(cur.get("vy",0)), float(cur.get("yaw",0))
    target_vx, target_vy, target_yaw = float(vx), float(vy), float(yaw)
    if accel is None:
        await client.call("set_command", {"vx": target_vx, "vy": target_vy, "yaw": target_yaw})
        return {"mode":"immediate","target":{"vx":target_vx,"vy":target_vy,"yaw":target_yaw}}
    accel = float(accel)
    if accel <= 0:
        raise ValueError("accel must be > 0")
    # Cancel existing ramp
    key = "velocity_ramp"
    if key in _ramp_tasks:
        _ramp_tasks[key].cancel()
    async def ramp():
        dt = 0.02
        cur_x, cur_y, cur_z = cur_vx, cur_vy, cur_yaw
        while True:
            dx = target_vx - cur_x
            dy = target_vy - cur_y
            dz = target_yaw - cur_z
            if abs(dx) < 1e-3 and abs(dy) < 1e-3 and abs(dz) < 1e-3:
                await client.call("set_command", {"vx": target_vx, "vy": target_vy, "yaw": target_yaw})
                break
            step_x = max(-accel*dt, min(accel*dt, dx))
            step_y = max(-accel*dt, min(accel*dt, dy))
            step_z = max(-accel*dt, min(accel*dt, dz))
            cur_x += step_x
            cur_y += step_y
            cur_z += step_z
            await client.call("set_command", {"vx": cur_x, "vy": cur_y, "yaw": cur_z})
            await asyncio.sleep(dt)
    task = asyncio.create_task(ramp())
    _ramp_tasks[key] = task
    est = max(abs(target_vx-cur_vx), abs(target_vy-cur_vy), abs(target_yaw-cur_yaw))/accel
    return {"mode":"ramped","target":{"vx":target_vx,"vy":target_vy,"yaw":target_yaw},"accel":accel,"time_estimate_s": est}

@mcp.tool()
async def get_camera_frame_base64() -> dict:
    """Grab a single camera frame as a base64 JPEG, cached 100 ms to avoid hammering the stream."""
    logging.info("[MCP] get_camera_frame_base64 called")
    global _camera_cache
    now = time.monotonic()*1000
    if _camera_cache["data"] and now - _camera_cache["ts"] < _CAMERA_CACHE_TTL_MS:
        return {"base64": _camera_cache["data"], "cached": True}
    host = os.getenv("CONTROL_HOST", "localhost")
    port = os.getenv("CONTROL_PORT", "9013")
    url = f"http://{host}:{port}/camera.mjpg"
    # /camera.mjpg is an unbounded MJPEG stream (server never closes the
    # connection) — httpx.get() would block forever waiting for a body that
    # never ends. Stream instead and stop as soon as one full JPEG is seen.
    async with httpx.AsyncClient(timeout=5.0) as http:
        buf = b""
        async with http.stream("GET", url) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start) if start != -1 else -1
                if start != -1 and end != -1:
                    jpeg = buf[start:end + 2]
                    break
                if len(buf) > 5_000_000:
                    raise RuntimeError("No JPEG found in first 5MB of camera stream")
            else:
                raise RuntimeError("Camera stream ended before a full JPEG frame arrived")
        b64 = base64.b64encode(jpeg).decode("ascii")
        _camera_cache["data"] = b64
        _camera_cache["ts"] = now
        return {"base64": b64, "cached": False, "size": len(jpeg)}

@mcp.tool()
async def get_depth_camera_frame_base64() -> dict:
    """Grab a single depth-camera frame as a base64 JPEG, cached 100 ms to avoid hammering the stream."""
    logging.info("[MCP] get_depth_camera_frame_base64 called")
    global _depth_camera_cache
    now = time.monotonic()*1000
    if _depth_camera_cache["data"] and now - _depth_camera_cache["ts"] < _CAMERA_CACHE_TTL_MS:
        return {"base64": _depth_camera_cache["data"], "cached": True}
    host = os.getenv("CONTROL_HOST", "localhost")
    port = os.getenv("CONTROL_PORT", "9013")
    url = f"http://{host}:{port}/depth.mjpg"
    # /depth.mjpg is an unbounded MJPEG stream — same fetch pattern as /camera.mjpg.
    async with httpx.AsyncClient(timeout=5.0) as http:
        buf = b""
        async with http.stream("GET", url) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start) if start != -1 else -1
                if start != -1 and end != -1:
                    jpeg = buf[start:end + 2]
                    break
                if len(buf) > 5_000_000:
                    raise RuntimeError("No JPEG found in first 5MB of depth camera stream")
            else:
                raise RuntimeError("Depth camera stream ended before a full JPEG frame arrived")
        b64 = base64.b64encode(jpeg).decode("ascii")
        _depth_camera_cache["data"] = b64
        _depth_camera_cache["ts"] = now
        return {"base64": b64, "cached": False, "size": len(jpeg)}

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    port = int(os.getenv("MCP_PORT", "9014"))
    if transport in ("sse", "streamable-http"):
        mcp.run(transport=transport, host="0.0.0.0", port=port)
    else:
        mcp.run()
