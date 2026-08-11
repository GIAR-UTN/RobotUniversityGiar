#!/usr/bin/env python3
"""
Headless smoke test for legged_gym/control/transport.py — mirrors the spirit
of the existing --headless smoke test in rugiar_driver.py (no browser, no
sim/Genesis dependency): connect a client, switch policies, estop, assert
status transitions, all over a real WebSocket round trip.

Uses a tiny fake in place of ControlService so this test has zero dependency
on Genesis/torch/the sim stack — it is exercising the transport, not the
control logic (that's covered by testing ControlService itself).

Run directly: python tests/test_control_transport.py
"""
import asyncio
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets

from legged_gym.control.transport import ControlServer


class FakeControlService:
    """Mimics ControlService's five-method surface without any real
    adapter/supervisor/safety machinery — see module docstring."""

    def __init__(self):
        self.active = "stable"
        self.paused = False
        self.tripped = False

    def request_switch(self, name: str) -> bool:
        if self.tripped or name == self.active:
            return False
        self.active = name
        return True

    def status(self) -> dict:
        return {"active": self.active, "paused": self.paused, "safety_tripped": self.tripped}

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def estop(self) -> None:
        self.tripped = True


class ControlServerTest(unittest.TestCase):
    # Each test starts its own ControlServer thread (there is no teardown
    # hook to stop one, and the daemon threads outlive their test) — give
    # each test method a distinct port so they never collide.
    _next_port = 9614

    def setUp(self):
        self.service = FakeControlService()
        ControlServerTest._next_port += 1
        self.server = ControlServer(self.service, port=ControlServerTest._next_port)
        self.server.serve_in_thread()
        time.sleep(0.2)  # let the event loop actually bind before connecting

    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_request_switch_and_status_round_trip(self):
        async def scenario():
            async with websockets.connect(f"ws://localhost:{self.server.port}/ws") as ws:
                await ws.send(json.dumps({"method": "request_switch", "params": {"name": "cautious"}, "id": 1}))
                # Command is queued on the socket thread; must be drained on
                # the "sim" thread before a reply is sent — mirrors the real
                # sim loop calling drain_commands() once per tick.
                for _ in range(50):
                    if not self.server.commands.empty():
                        self.server.drain_commands()
                        break
                    await asyncio.sleep(0.02)
                reply = json.loads(await ws.recv())
                return reply

        reply = self._run(scenario())
        self.assertEqual(reply["id"], 1)
        self.assertTrue(reply["result"])
        self.assertEqual(self.service.active, "cautious")

    def test_estop_is_prioritized_within_a_batch(self):
        async def scenario():
            async with websockets.connect(f"ws://localhost:{self.server.port}/ws") as ws:
                # Queue a request_switch (to a policy that would otherwise
                # succeed), then estop, before any drain — estop must run
                # first within the batch despite arriving second.
                await ws.send(json.dumps({"method": "request_switch", "params": {"name": "cautious"}, "id": 2}))
                await ws.send(json.dumps({"method": "estop", "params": {}, "id": 3}))
                await asyncio.sleep(0.1)
                self.server.drain_commands()
                r1 = json.loads(await ws.recv())
                r2 = json.loads(await ws.recv())
                return r1, r2

        r1, r2 = self._run(scenario())
        # Reply order mirrors execution order: estop (id 3) must be
        # dispatched, and therefore replied to, before the switch (id 2).
        self.assertEqual(r1["id"], 3)
        self.assertEqual(r2["id"], 2)
        self.assertTrue(self.service.tripped)
        # The switch ran AFTER estop tripped safety, so FakeControlService
        # (mirroring the real safety-gate invariant) must have refused it.
        self.assertFalse(r2["result"])
        self.assertEqual(self.service.active, "stable")

    def test_unknown_method_is_rejected(self):
        async def scenario():
            async with websockets.connect(f"ws://localhost:{self.server.port}/ws") as ws:
                await ws.send(json.dumps({"method": "delete_everything", "params": {}, "id": 4}))
                await asyncio.sleep(0.1)
                self.server.drain_commands()
                return json.loads(await ws.recv())

        reply = self._run(scenario())
        self.assertIn("error", reply)
        self.assertNotIn("result", reply)

    def test_non_dict_message_is_rejected_without_reaching_the_sim_thread(self):
        """A client sending a bare JSON scalar/array must never be queued —
        drain_commands()/_dispatch() assume dict .get() access, so an
        un-rejected non-dict payload would raise on the sim thread and take
        the whole sim loop down over a client typo."""
        async def scenario():
            async with websockets.connect(f"ws://localhost:{self.server.port}/ws") as ws:
                await ws.send(json.dumps([1, 2, 3]))
                return json.loads(await ws.recv())

        reply = self._run(scenario())
        self.assertIn("error", reply)
        self.assertTrue(self.server.commands.empty())

    def test_status_broadcast_reaches_client(self):
        async def scenario():
            async with websockets.connect(f"ws://localhost:{self.server.port}/ws") as ws:
                self.server.publish_status(self.service.status())
                # Broadcast loop ticks at ~10 Hz; give it a couple of cycles.
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                return json.loads(raw)

        msg = self._run(scenario())
        self.assertEqual(msg["method"], "status")
        self.assertIn("active", msg["result"])

    def test_serve_in_thread_raises_on_port_already_in_use(self):
        """A second ControlServer on the same port must fail loudly and
        synchronously from serve_in_thread(), not print 'listening' and
        silently drop every reply (the bug the Fable review flagged)."""
        other = ControlServer(FakeControlService(), port=self.server.port)
        with self.assertRaises(RuntimeError):
            other.serve_in_thread()


if __name__ == "__main__":
    unittest.main()
