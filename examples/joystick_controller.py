#!/usr/bin/env python3
"""Reference home-made controller for the control protocol described in
docs/index.html §13 ("Talking to the robot: the control protocol").

This talks to legged_gym/control/transport.py::ControlServer the exact same
way the web UI does — a plain WebSocket at /ws. Two kinds of traffic go over
that socket:

  - A steady set_command stream (vx, vy, yaw) at --hz, driven by the stick.
  - Discrete, edge-triggered control messages — request_switch, pause,
    resume, restart, estop — fired once per button press, not every tick.
    These map to the exact same server-side calls
    (legged_gym/control/service.py) the web UI's policy panel/E-STOP button
    use; the server doesn't care what kind of client sent them.

It works unmodified against swap_experiment.py running either the Genesis
simulator or (once wired up, --real) an actual G1 — same protocol either
way, only what set_command/estop actually *do* underneath differs.

Usage:
    python examples/joystick_controller.py ws://localhost:9013 --demo
    python examples/joystick_controller.py ws://<robot-ip>:9013 --token <secret>

Default gamepad button mapping (Xbox-style; override with the --btn-* flags
if your controller numbers these differently):
    A (0)       estop                     — trips safety immediately
    B (1)       pause / resume toggle
    X (2)       restart                   — reset to standing pose
    right bumper (5)   switch to next policy (cycles server's policy list)
    left bumper  (4)   switch to previous policy

Requires: pip install websockets   (pygame only if you want real gamepad
input instead of --demo — see make_gamepad_command()/poll_buttons() below).

Building your OWN home-made controller (a different gamepad library, a
phone app, a custom microcontroller bridge, ...)? The connect/send/receive
loop in main()/run() stays identical — replace get_command() and
poll_buttons() with whatever reads your input device.
"""
import argparse
import asyncio
import itertools
import json
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("This example needs the 'websockets' package: pip install websockets")


async def send_control(ws, msg_id, method: str, **params) -> None:
    """Fires one discrete control message (request_switch/pause/resume/
    restart/estop) — these are one-shot commands, unlike set_command's
    steady stream, so they're sent once per button edge, not on a timer."""
    await ws.send(json.dumps({"method": method, "params": params, "id": next(msg_id)}))
    print(f"-> {method}({params})" if params else f"-> {method}")


async def stream_commands(ws, msg_id, get_command, hz: float) -> None:
    while True:
        vx, vy, yaw = get_command()
        await ws.send(json.dumps({
            "method": "set_command",
            "params": {"vx": vx, "vy": vy, "yaw": yaw},
            "id": next(msg_id),
        }))
        await asyncio.sleep(1.0 / hz)


async def watch_status(ws) -> None:
    """The server pushes an unprompted status message to every connected
    client at ~10Hz (see docs/index.html §13) — this just prints the bits
    an operator cares about when they press a button, so switch/estop/pause
    presses get visible confirmation instead of firing blind."""
    last = None
    async for raw in ws:
        msg = json.loads(raw)
        if msg.get("method") != "status":
            if "error" in msg:
                print(f"!! server error: {msg['error']}")
            continue
        result = msg.get("result", {})
        summary = (result.get("active"), result.get("pending"), result.get("ramping"),
                   result.get("paused"), result.get("safety_tripped"))
        if summary != last:
            last = summary
            active, pending, ramping, paused, tripped = summary
            state = f"active={active}"
            if ramping:
                state += f" (switching -> {pending})"
            if paused:
                state += " PAUSED"
            if tripped:
                state += " ESTOPPED"
            print(f"[status] {state}")


async def poll_buttons(stick, ws, msg_id, policies, btn_map, poll_hz: float = 20.0) -> None:
    """Edge-detects gamepad button presses and fires the matching control
    message once per press (not once per tick, or a held button would spam
    request_switch/restart). `policies` is the ordered list this client
    cycles through for the bumper buttons — pass --policies to override the
    guessed order; the server is the source of truth for what's actually
    loaded (see the printed [status] active=... line)."""
    import pygame
    held = {name: False for name in btn_map}
    policy_idx = 0
    while True:
        pygame.event.pump()
        for name, btn in btn_map.items():
            pressed = stick.get_button(btn)
            if pressed and not held[name]:
                if name == "estop":
                    await send_control(ws, msg_id, "estop")
                elif name == "pause_resume":
                    # Client-side toggle guess; [status] output is the real
                    # source of truth if it ever drifts (e.g. two clients).
                    await send_control(ws, msg_id, "resume" if held.get("_paused") else "pause")
                    held["_paused"] = not held.get("_paused", False)
                elif name == "restart":
                    await send_control(ws, msg_id, "restart")
                elif name == "next_policy" and policies:
                    policy_idx = (policy_idx + 1) % len(policies)
                    await send_control(ws, msg_id, "request_switch", name=policies[policy_idx])
                elif name == "prev_policy" and policies:
                    policy_idx = (policy_idx - 1) % len(policies)
                    await send_control(ws, msg_id, "request_switch", name=policies[policy_idx])
            held[name] = pressed
        await asyncio.sleep(1.0 / poll_hz)


async def run(url: str, token: str, get_command, stick, policies, btn_map, hz: float) -> None:
    full_url = f"{url}/ws" + (f"?token={token}" if token else "")
    async with websockets.connect(full_url) as ws:
        print(f"Connected to {full_url}")
        msg_id = itertools.count(1)
        tasks = [
            asyncio.create_task(stream_commands(ws, msg_id, get_command, hz)),
            asyncio.create_task(watch_status(ws)),
        ]
        if stick is not None:
            tasks.append(asyncio.create_task(poll_buttons(ws=ws, msg_id=msg_id, stick=stick,
                                                            policies=policies, btn_map=btn_map)))
        await asyncio.gather(*tasks)


def demo_command() -> tuple:
    """No gamepad needed — walks forward for 4s, then turns in place for 4s,
    on a loop. Good for proving the connection/auth/protocol work end to end
    before wiring up real input hardware. (Button controls are skipped in
    --demo since there's no gamepad to poll — use a real controller to
    exercise switch/pause/estop.)"""
    phase = time.monotonic() % 8.0
    if phase < 4.0:
        return (0.3, 0.0, 0.0)
    return (0.0, 0.0, 0.5)


def make_gamepad():
    """Real joystick input via pygame — left stick = vx/vy, right stick x =
    yaw, buttons = switch/pause/restart/estop (see poll_buttons above and
    the --btn-* flags). Swap the axis reads out for your own input source;
    the clamp values here are just a client-side courtesy (the server
    clamps to the active policy's trained envelope regardless — see
    set_command in the control protocol docs)."""
    import pygame
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        sys.exit("No gamepad detected — plug one in, or run with --demo")
    stick = pygame.joystick.Joystick(0)
    stick.init()
    print(f"Using gamepad: {stick.get_name()}")

    MAX_VX, MAX_VY, MAX_YAW = 0.8, 0.5, 1.0

    def get_command() -> tuple:
        pygame.event.pump()
        vx = -stick.get_axis(1) * MAX_VX  # stick "up" is a negative axis value
        vy = -stick.get_axis(0) * MAX_VY
        yaw = -stick.get_axis(3) * MAX_YAW
        return (vx, vy, yaw)

    return get_command, stick


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="control server base URL, e.g. ws://192.168.1.50:9013")
    parser.add_argument("--token", default=None, help="shared secret, if the server was started with --token")
    parser.add_argument("--hz", type=float, default=10.0, help="how often to send set_command")
    parser.add_argument("--demo", action="store_true", help="scripted forward/turn loop — no gamepad required")
    parser.add_argument("--policies", nargs="*", default=None,
                         help="ordered policy names to cycle through with the bumpers "
                              "(default: whatever order the server reports at connect time "
                              "is NOT auto-discovered here — pass explicitly, or check the "
                              "web UI / server startup log for --policy names)")
    parser.add_argument("--btn-estop", type=int, default=0)
    parser.add_argument("--btn-pause", type=int, default=1)
    parser.add_argument("--btn-restart", type=int, default=2)
    parser.add_argument("--btn-next-policy", type=int, default=5)
    parser.add_argument("--btn-prev-policy", type=int, default=4)
    args = parser.parse_args()

    url = args.url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")

    if args.demo:
        get_command, stick = demo_command, None
    else:
        get_command, stick = make_gamepad()

    btn_map = {
        "estop": args.btn_estop,
        "pause_resume": args.btn_pause,
        "restart": args.btn_restart,
        "next_policy": args.btn_next_policy,
        "prev_policy": args.btn_prev_policy,
    }

    try:
        asyncio.run(run(url, args.token, get_command, stick, args.policies, btn_map, args.hz))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
