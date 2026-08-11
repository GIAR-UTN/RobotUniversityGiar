#!/usr/bin/env python3
"""
Unit tests for SimAdapter.set_fall_termination()/fall_termination
(legged_gym/control/adapter.py).

Regression test for the "robot takes a step then just stops responding, no
red safety-tripped dot" bug reported against the control web:
legged_robot.py's own fall/contact-force termination (check_termination()'s
fail_buf) reads the SAME cfg.env.max_projected_gravity /
fail_to_terminal_time_s the TRAINING config sets for curriculum purposes
(short episodes, fast resets) — left at those values during live control, a
policy that's merely stumbling (nowhere near SafetyGovernor's own much
stricter trip threshold) gets silently teleported back to its default pose
mid-env.step(), with no signal to ControlService/the web UI at all.
set_fall_termination() lets a live control session relax (or revert) those
two thresholds independently of whatever a training run elsewhere uses,
since each backs its own separate env instance.

Like test_viser_camera_tracking.py/test_push_direction.py, this stubs out
legged_gym's package hierarchy so importing adapter.py doesn't eagerly pull
in a real physics backend (Genesis/IsaacGym/IsaacLab) via
legged_gym/__init__.py — this module only touches plain attributes and a
numpy stand-in for the fail_buf tensor.

Run directly: python tests/test_fall_termination.py
"""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_package(dotted_name: str):
    if dotted_name in sys.modules:
        return
    module = types.ModuleType(dotted_name)
    module.__path__ = [str(ROOT / Path(*dotted_name.split(".")))]
    module.__package__ = dotted_name
    sys.modules[dotted_name] = module


for _pkg in ("legged_gym", "legged_gym.utils"):
    _stub_package(_pkg)
# The real legged_gym/__init__.py normally defines this; adapter.py's own
# import chain (legged_gym.utils.math_utils) resolves through it.
sys.modules["legged_gym"].LEGGED_GYM_ROOT_DIR = str(ROOT)

from legged_gym.control.adapter import SimAdapter

TRAINING_MAX_PROJECTED_GRAVITY = -0.1
TRAINING_FAIL_TO_TERMINAL_TIME_S = 0.1


class _EnvCfg:
    class env:
        max_projected_gravity = TRAINING_MAX_PROJECTED_GRAVITY
        fail_to_terminal_time_s = TRAINING_FAIL_TO_TERMINAL_TIME_S


class _FakeEnv:
    def __init__(self):
        self.cfg = _EnvCfg()
        self.fail_buf = np.array([3, 0])  # nonzero -- a live change must clear it


def _make_adapter():
    """A bare SimAdapter with only what set_fall_termination()/
    fall_termination touch -- bypasses __init__ (real env, torch tensors,
    random-events wiring) entirely, same pattern as
    test_viser_camera_tracking.py's _make_viewer()."""
    adapter = object.__new__(SimAdapter)
    adapter.env = _FakeEnv()
    adapter._training_max_projected_gravity = TRAINING_MAX_PROJECTED_GRAVITY
    adapter._training_fail_to_terminal_time_s = TRAINING_FAIL_TO_TERMINAL_TIME_S
    adapter._fall_max_projected_gravity = None
    adapter._fall_to_terminal_time_s = None
    return adapter


class TestFallTermination(unittest.TestCase):
    def test_none_none_matches_training_config(self):
        adapter = _make_adapter()
        adapter.set_fall_termination(None, None)
        self.assertEqual(adapter.env.cfg.env.max_projected_gravity, TRAINING_MAX_PROJECTED_GRAVITY)
        self.assertEqual(adapter.env.cfg.env.fail_to_terminal_time_s, TRAINING_FAIL_TO_TERMINAL_TIME_S)

    def test_override_relaxes_both_thresholds(self):
        adapter = _make_adapter()
        adapter.set_fall_termination(0.5, 1.0)
        self.assertEqual(adapter.env.cfg.env.max_projected_gravity, 0.5)
        self.assertEqual(adapter.env.cfg.env.fail_to_terminal_time_s, 1.0)

    def test_partial_override_leaves_the_other_at_training_value(self):
        adapter = _make_adapter()
        adapter.set_fall_termination(0.5, None)
        self.assertEqual(adapter.env.cfg.env.max_projected_gravity, 0.5)
        self.assertEqual(adapter.env.cfg.env.fail_to_terminal_time_s, TRAINING_FAIL_TO_TERMINAL_TIME_S)

    def test_reverting_to_none_none_restores_training_values(self):
        adapter = _make_adapter()
        adapter.set_fall_termination(0.5, 1.0)
        adapter.set_fall_termination(None, None)
        self.assertEqual(adapter.env.cfg.env.max_projected_gravity, TRAINING_MAX_PROJECTED_GRAVITY)
        self.assertEqual(adapter.env.cfg.env.fail_to_terminal_time_s, TRAINING_FAIL_TO_TERMINAL_TIME_S)

    def test_nonpositive_hold_time_rejected(self):
        adapter = _make_adapter()
        with self.assertRaises(ValueError):
            adapter.set_fall_termination(None, 0.0)
        with self.assertRaises(ValueError):
            adapter.set_fall_termination(None, -1.0)

    def test_fail_buf_cleared_on_any_change(self):
        """A live threshold change must not inherit a partial fail_buf count
        built up under whatever (possibly stricter) setting was previously
        in effect."""
        adapter = _make_adapter()
        self.assertTrue((adapter.env.fail_buf != 0).any())  # seeded nonzero above
        adapter.set_fall_termination(0.5, None)
        self.assertTrue((adapter.env.fail_buf == 0).all())

    def test_fall_termination_property_reports_effective_and_training_values(self):
        adapter = _make_adapter()
        adapter.set_fall_termination(0.5, None)
        ft = adapter.fall_termination
        self.assertEqual(ft, {
            "max_projected_gravity": 0.5,
            "fail_to_terminal_time_s": TRAINING_FAIL_TO_TERMINAL_TIME_S,
            "training_max_projected_gravity": TRAINING_MAX_PROJECTED_GRAVITY,
            "training_fail_to_terminal_time_s": TRAINING_FAIL_TO_TERMINAL_TIME_S,
        })


if __name__ == "__main__":
    unittest.main()
