#!/usr/bin/env python3
"""
Unit tests for renaming a policy — PolicySupervisor.rename_policy()
(legged_gym/control/supervisor.py), TrainingManager.rename_policy()
(legged_gym/control/training.py), and ControlService.rename_policy(),
which composes both. Same package-stub trick as test_delete_policy.py.

Run directly: python tests/test_rename_policy.py
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

import legged_gym.control.training as training_mod
from legged_gym.control.service import ControlService
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


class TestSupervisorRenamePolicy(unittest.TestCase):
    def test_renames_an_inactive_policy(self):
        sup = _supervisor(["stable", "croucher", "damping"], active="stable")
        sup.rename_policy("croucher", "crouch_v2")
        self.assertNotIn("croucher", sup.policies)
        self.assertIn("crouch_v2", sup.policies)
        self.assertEqual(sup.policies["crouch_v2"].name, "crouch_v2")

    def test_renaming_the_active_policy_updates_active_name(self):
        sup = _supervisor(["stable", "croucher", "damping"], active="croucher")
        sup.rename_policy("croucher", "crouch_v2")
        self.assertEqual(sup.active_name, "crouch_v2")
        self.assertEqual(sup.active.name, "crouch_v2")

    def test_renaming_a_pending_switch_target_updates_pending_name(self):
        sup = _supervisor(["stable", "croucher", "damping"], active="stable")
        sup.request_switch("croucher")
        sup.rename_policy("croucher", "crouch_v2")
        self.assertEqual(sup.pending_name, "crouch_v2")

    def test_refuses_to_rename_damping(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        with self.assertRaises(ValueError):
            sup.rename_policy("damping", "damping2")

    def test_refuses_collision_with_existing_name(self):
        sup = _supervisor(["stable", "croucher", "damping"], active="stable")
        with self.assertRaises(ValueError):
            sup.rename_policy("croucher", "stable")
        self.assertIn("croucher", sup.policies)

    def test_unknown_policy_raises_keyerror(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        with self.assertRaises(KeyError):
            sup.rename_policy("nonexistent", "whatever")


class TestTrainingManagerRenamePolicy(unittest.TestCase):
    def setUp(self):
        self._orig_policies_dir = training_mod.POLICIES_DIR
        self._tmp = tempfile.TemporaryDirectory()
        training_mod.POLICIES_DIR = Path(self._tmp.name)

    def tearDown(self):
        training_mod.POLICIES_DIR = self._orig_policies_dir
        self._tmp.cleanup()

    def _mgr(self):
        mgr = TrainingManager.__new__(TrainingManager)
        mgr.policy_sources = {}
        return mgr

    def test_renames_folder_and_repoints_catalog_paths(self):
        mgr = self._mgr()
        old_dir = training_mod.POLICIES_DIR / "croucher"
        old_dir.mkdir(parents=True)
        (old_dir / "checkpoint.pt").write_bytes(b"fake")
        (old_dir / "train_checkpoint.pt").write_bytes(b"fake")
        mgr.register_source(
            "croucher", task="g1", checkpoint=str(old_dir / "checkpoint.pt"),
            train_checkpoint=str(old_dir / "train_checkpoint.pt"),
        )

        mgr.rename_policy("croucher", "crouch_v2")

        new_dir = training_mod.POLICIES_DIR / "crouch_v2"
        self.assertFalse(old_dir.exists())
        self.assertTrue((new_dir / "checkpoint.pt").is_file())
        self.assertTrue((new_dir / "train_checkpoint.pt").is_file())
        self.assertNotIn("croucher", mgr.policy_sources)
        source = mgr.policy_sources["crouch_v2"]
        self.assertEqual(source["checkpoint"], str(new_dir / "checkpoint.pt"))
        self.assertEqual(source["train_checkpoint"], str(new_dir / "train_checkpoint.pt"))

    def test_refuses_to_rename_a_policy_with_no_dedicated_folder(self):
        mgr = self._mgr()
        mgr.register_source("stable", task="g1", checkpoint="/some/external/path.pt")
        with self.assertRaises(ValueError):
            mgr.rename_policy("stable", "stable2")

    def test_refuses_collision_with_an_existing_folder(self):
        mgr = self._mgr()
        (training_mod.POLICIES_DIR / "croucher").mkdir(parents=True)
        (training_mod.POLICIES_DIR / "taken").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            mgr.rename_policy("croucher", "taken")


class TestControlServiceRenamePolicy(unittest.TestCase):
    def setUp(self):
        self._orig_policies_dir = training_mod.POLICIES_DIR
        self._tmp = tempfile.TemporaryDirectory()
        training_mod.POLICIES_DIR = Path(self._tmp.name)

        self.supervisor = _supervisor(["stable", "croucher", "damping"], active="stable")
        self.training = TrainingManager.__new__(TrainingManager)
        self.training.policy_sources = {}
        old_dir = training_mod.POLICIES_DIR / "croucher"
        old_dir.mkdir(parents=True)
        (old_dir / "checkpoint.pt").write_bytes(b"fake")
        self.training.register_source("croucher", task="g1", checkpoint=str(old_dir / "checkpoint.pt"))

        self.service = ControlService.__new__(ControlService)
        self.service.supervisor = self.supervisor
        self.service.training = self.training

    def tearDown(self):
        training_mod.POLICIES_DIR = self._orig_policies_dir
        self._tmp.cleanup()

    def test_renames_across_supervisor_and_training(self):
        self.service.rename_policy("croucher", "crouch_v2")
        self.assertIn("crouch_v2", self.supervisor.policies)
        self.assertIn("crouch_v2", self.training.policy_sources)
        self.assertFalse((training_mod.POLICIES_DIR / "croucher").exists())

    def test_same_name_is_a_no_op(self):
        self.service.rename_policy("croucher", "croucher")
        self.assertIn("croucher", self.supervisor.policies)

    def test_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            self.service.rename_policy("croucher", "   ")

    def test_rejects_invalid_characters(self):
        with self.assertRaises(ValueError):
            self.service.rename_policy("croucher", "not/a valid name!")

    def test_rejects_reserved_damping_name(self):
        with self.assertRaises(ValueError):
            self.service.rename_policy("croucher", "damping")


if __name__ == "__main__":
    unittest.main()
