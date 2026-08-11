#!/usr/bin/env python3
"""
Unit tests for SimAdapter.set_command()/set_operator_speed_limit()/
_clamp_command() (legged_gym/control/adapter.py).

Regression coverage for the "press and hold a movement key, it moves
briefly then stops responding" bug: held-key cruise used to ramp toward
the RAW edge of the trained command envelope, where a borderline/
undertrained checkpoint is more likely to stumble than walk — the fix
moved the actual enforced cap to the SERVER (here), as `operator_speed_limit`,
shared by every client (web UI, examples/joystick_controller.py, a future
robot) instead of each one reimplementing its own local scaling constant.
Also covers the explicit follow-up ask: allow deliberately going PAST the
trained envelope (values > 1.0, up to OPERATOR_SPEED_LIMIT_MAX) for
out-of-distribution experimentation in sim.

This stubs out legged_gym's package hierarchy so importing adapter.py
doesn't eagerly pull in a real physics backend via legged_gym/__init__.py.

Run directly: python tests/test_operator_speed_limit.py
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
sys.modules["legged_gym"].LEGGED_GYM_ROOT_DIR = str(ROOT)

from legged_gym.control.adapter import SimAdapter, OPERATOR_SPEED_LIMIT_MAX


class _Ranges:
    lin_vel_x = [-1.0, 1.0]
    lin_vel_y = [-1.0, 1.0]
    ang_vel_yaw = [-1.0, 1.0]


class _Commands:
    ranges = _Ranges()
    heading_command = True


class _Cfg:
    commands = _Commands()


class _FakeEnv:
    def __init__(self):
        self.cfg = _Cfg()
        self.commands = np.zeros((1, 3))


def _make_adapter():
    """A bare SimAdapter with only what set_command()/
    set_operator_speed_limit()/_clamp_command() touch -- bypasses __init__
    (real env, random-events/episode-timeout wiring) entirely."""
    adapter = object.__new__(SimAdapter)
    adapter.env = _FakeEnv()
    adapter._orig_heading_command = adapter.env.cfg.commands.heading_command
    adapter._auto_commands = True
    adapter._manual_command = (0.0, 0.0, 0.0)
    adapter._operator_speed_limit = 1.0
    return adapter


class TestOperatorSpeedLimit(unittest.TestCase):
    def test_set_command_clamps_to_trained_range_by_default(self):
        adapter = _make_adapter()
        adapter.set_command(5.0, -5.0, 5.0)
        self.assertEqual(adapter._manual_command, (1.0, -1.0, 1.0))

    def test_limit_below_one_scales_the_clamp_down(self):
        adapter = _make_adapter()
        adapter.set_operator_speed_limit(0.5)
        adapter.set_command(1.0, 1.0, 1.0)
        self.assertEqual(adapter._manual_command, (0.5, 0.5, 0.5))

    def test_limit_above_one_is_allowed_up_to_max(self):
        adapter = _make_adapter()
        adapter.set_operator_speed_limit(2.0)
        adapter.set_command(5.0, -5.0, 5.0)
        self.assertEqual(adapter._manual_command, (2.0, -2.0, 2.0))

    def test_limit_above_max_rejected(self):
        adapter = _make_adapter()
        with self.assertRaises(ValueError):
            adapter.set_operator_speed_limit(OPERATOR_SPEED_LIMIT_MAX + 0.1)

    def test_limit_at_max_is_accepted(self):
        adapter = _make_adapter()
        adapter.set_operator_speed_limit(OPERATOR_SPEED_LIMIT_MAX)
        self.assertEqual(adapter.operator_speed_limit, OPERATOR_SPEED_LIMIT_MAX)

    def test_limit_zero_or_negative_rejected(self):
        adapter = _make_adapter()
        with self.assertRaises(ValueError):
            adapter.set_operator_speed_limit(0.0)
        with self.assertRaises(ValueError):
            adapter.set_operator_speed_limit(-0.5)

    def test_lowering_limit_reclamps_an_already_active_manual_command(self):
        """Dialing the limit down while already cruising at speed must take
        effect immediately, not just on the next set_command call."""
        adapter = _make_adapter()
        adapter.set_command(1.0, 1.0, 1.0)
        adapter.set_operator_speed_limit(0.3)
        self.assertEqual(adapter._manual_command, (0.3, 0.3, 0.3))
        self.assertEqual(list(adapter.env.commands[0]), [0.3, 0.3, 0.3])

    def test_adjusting_limit_alone_does_not_force_manual_mode(self):
        """Adjusting the limit is not the same gesture as issuing a
        command -- must not silently hijack control away from auto mode."""
        adapter = _make_adapter()
        self.assertTrue(adapter._auto_commands)
        adapter.set_operator_speed_limit(0.5)
        self.assertTrue(adapter._auto_commands)
        self.assertTrue(adapter.env.cfg.commands.heading_command)  # untouched

    def test_set_command_disables_auto_mode_and_heading_command(self):
        adapter = _make_adapter()
        adapter.set_command(0.2, 0.0, 0.0)
        self.assertFalse(adapter._auto_commands)
        self.assertFalse(adapter.env.cfg.commands.heading_command)


if __name__ == "__main__":
    unittest.main()
