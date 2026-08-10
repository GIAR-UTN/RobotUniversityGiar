#!/usr/bin/env python3
"""
Regression test for a real bug: ControlService.fuse_policies() existed and
worked, but the WebSocket transport (legged_gym/control/transport.py)
carries its own explicit `METHODS` allowlist, separate from ControlService
itself — adding a new ControlService method does NOT automatically make it
callable over the wire, and nothing enforced that they stay in sync. The
control web's Fuse policies panel called fuse_policies() and got back
"unknown method 'fuse_policies'" until METHODS was updated by hand.

This asserts every public, RPC-shaped ControlService method (i.e. everything
except tick(), which takes a raw obs tensor and is only ever called from the
sim loop itself, never over the wire) is present in transport.METHODS — so
the next method that's added and forgotten here fails a test instead of
failing silently at the exact moment someone tries to use it live.

Unlike most other tests in this file's neighborhood, this one needs the
REAL ControlService class (to introspect its actual method list), not a
fake — so it needs SIMULATOR set and a real simulator backend importable,
same as any test importing legged_gym.control.service directly.

Run directly: SIMULATOR=genesis python tests/test_control_service_methods_registered.py
"""
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legged_gym.control.service import ControlService
from legged_gym.control.transport import METHODS

# Not RPC-shaped: called once per sim tick with a raw obs tensor, which isn't
# JSON — this is the control loop's own internal call, never a wire method.
NOT_RPC_METHODS = {"tick"}


class TestControlServiceMethodsRegistered(unittest.TestCase):
    def test_every_public_method_is_in_the_transport_allowlist(self):
        public_methods = {
            name for name, _ in inspect.getmembers(ControlService, predicate=inspect.isfunction)
            if not name.startswith("_")
        }
        expected = public_methods - NOT_RPC_METHODS
        missing = expected - METHODS
        self.assertEqual(missing, set(),
                          f"ControlService method(s) not reachable over the wire — add to "
                          f"transport.METHODS: {sorted(missing)}")

    def test_no_stale_entries_pointing_at_a_removed_method(self):
        public_methods = {
            name for name, _ in inspect.getmembers(ControlService, predicate=inspect.isfunction)
            if not name.startswith("_")
        }
        stale = METHODS - public_methods
        self.assertEqual(stale, set(),
                          f"transport.METHODS name(s) with no matching ControlService method "
                          f"(rename or removal not reflected here): {sorted(stale)}")


if __name__ == "__main__":
    unittest.main()
