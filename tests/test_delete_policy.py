#!/usr/bin/env python3
"""
Unit tests for deleting a policy — PolicySupervisor.remove_policy()
(legged_gym/control/supervisor.py) and TrainingManager.forget_source()
(legged_gym/control/training.py), the two halves ControlService.
delete_policy() composes. Same package-stub trick as
tests/test_push_direction.py: legged_gym/__init__.py unconditionally
imports a physics backend as a side effect of package import, even though
nothing this test touches actually needs one.

Run directly: python tests/test_delete_policy.py
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_package(dotted_name: str):
    if dotted_name in sys.modules:
        return
    module = types.ModuleType(dotted_name)
    module.__path__ = [str(ROOT / Path(*dotted_name.split(".")))]
    module.__package__ = dotted_name
    sys.modules[dotted_name] = module


for _pkg in ("legged_gym", "legged_gym.control"):
    _stub_package(_pkg)

from legged_gym.control.supervisor import PolicySupervisor
from legged_gym.control.training import TrainingManager


class _FakeBackend:
    def reset(self):
        pass


class _FakePolicy:
    def __init__(self, name):
        self.name = name
        self.backend = _FakeBackend()


def _supervisor(names, active):
    return PolicySupervisor({n: _FakePolicy(n) for n in names}, active=active)


class TestRemovePolicy(unittest.TestCase):
    def test_removes_an_inactive_policy(self):
        sup = _supervisor(["stable", "croucher", "damping"], active="stable")
        sup.remove_policy("croucher")
        self.assertNotIn("croucher", sup.policies)

    def test_refuses_to_remove_active_policy(self):
        sup = _supervisor(["stable", "croucher", "damping"], active="croucher")
        with self.assertRaises(ValueError):
            sup.remove_policy("croucher")
        self.assertIn("croucher", sup.policies)

    def test_refuses_to_remove_damping(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        with self.assertRaises(ValueError):
            sup.remove_policy("damping")
        self.assertIn("damping", sup.policies)

    def test_refuses_to_remove_pending_switch_target(self):
        sup = _supervisor(["stable", "croucher", "damping"], active="stable")
        sup.request_switch("croucher")
        with self.assertRaises(ValueError):
            sup.remove_policy("croucher")
        self.assertIn("croucher", sup.policies)

    def test_unknown_policy_raises_keyerror(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        with self.assertRaises(KeyError):
            sup.remove_policy("nonexistent")


class TestForgetSource(unittest.TestCase):
    def test_removes_from_catalog_and_deletes_checkpoint_file(self):
        mgr = TrainingManager.__new__(TrainingManager)
        mgr.policy_sources = {}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "exported" / "policy_lstm_1.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fake")
            mgr.register_source("croucher", task="g1", checkpoint=str(checkpoint))
            self.assertTrue(checkpoint.exists())

            mgr.forget_source("croucher")

            self.assertNotIn("croucher", mgr.policy_sources)
            self.assertFalse(checkpoint.exists())

    def test_unknown_source_is_a_no_op(self):
        mgr = TrainingManager.__new__(TrainingManager)
        mgr.policy_sources = {}
        mgr.forget_source("nonexistent")  # must not raise
        self.assertEqual(mgr.policy_sources, {})

    def test_source_with_no_checkpoint_on_disk_is_still_forgotten(self):
        mgr = TrainingManager.__new__(TrainingManager)
        mgr.policy_sources = {}
        mgr.register_source("stable", task="g1", checkpoint=None)
        mgr.forget_source("stable")  # must not raise even though there's no file
        self.assertNotIn("stable", mgr.policy_sources)


if __name__ == "__main__":
    unittest.main()
