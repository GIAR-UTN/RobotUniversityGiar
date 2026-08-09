#!/usr/bin/env python3
"""Reference home-made controller for the control protocol described in
docs/index.html §13 ("Talking to the robot: the control protocol").

This talks to legged_gym/control/transport.py::ControlServer the exact same
way the web UI does — a plain WebSocket at /ws, sending
{"method": "set_command", "params": {"vx": ..., "vy": ..., "yaw": ...}, "id": N}
at a steady rate. It works unmodified against swap_experiment.py running
either the Genesis simulator or (once wired up, --real) an actual G1 — the
server doesn't care what kind of client is on the other end of the socket.

Usage:
    python examples/joystick_controller.py ws://localhost:9013 --demo
    python examples/joystick_controller.py ws://<robot-ip>:9013 --token <secret>

Requires: pip install websockets   (pygame only if you want real gamepad
input instead of --demo — see make_gamepad_command() below).

Building your OWN home-made controller (a different gamepad library, a
phone app, a custom microcontroller bridge, ...)? Everything below
send_commands() is the part to replace — the connect/send loop stays
identical, because the protocol doesn't care where the (vx, vy, yaw)
numbers came from.
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


async def send_commands(url: str, token: str, get_command, hz: float = 10.0) -> None:
    full_url = f"{url}/ws" + (f"?token={token}" if token else "")
    async with websockets.connect(full_url) as ws:
        print(f"Connected to {full_url}")
        msg_id = itertools.count(1)
        while True:
            vx, vy, yaw = get_command()
            await ws.send(json.dumps({
                "method": "set_command",
                "params": {"vx": vx, "vy": vy, "yaw": yaw},
                "id": next(msg_id),
            }))
            await asyncio.sleep(1.0 / hz)


def demo_command() -> tuple:
    """No gamepad needed — walks forward for 4s, then turns in place for 4s,
    on a loop. Good for proving the connection/auth/protocol work end to end
    before wiring up real input hardware."""
    phase = time.monotonic() % 8.0
    if phase < 4.0:
        return (0.3, 0.0, 0.0)
    return (0.0, 0.0, 0.5)


def make_gamepad_command():
    """Real joystick input via pygame — left stick = vx/vy, right stick x =
    yaw. Swap this out for your own input source; the clamp values here are
    just a client-side courtesy (the server clamps to the active policy's
    trained envelope regardless — see set_command in the control protocol
    docs)."""
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

    return get_command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="control server base URL, e.g. ws://192.168.1.50:9013")
    parser.add_argument("--token", default=None, help="shared secret, if the server was started with --token")
    parser.add_argument("--hz", type=float, default=10.0, help="how often to send set_command")
    parser.add_argument("--demo", action="store_true", help="scripted forward/turn loop — no gamepad required")
    args = parser.parse_args()

    url = args.url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    get_command = demo_command if args.demo else make_gamepad_command()

    try:
        asyncio.run(send_commands(url, args.token, get_command, hz=args.hz))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
