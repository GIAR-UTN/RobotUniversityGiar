#!/usr/bin/env python3
"""
Unit tests for the three reward functions in
legged_gym/envs/base/legged_robot.py that got a configurable target added
(alongside the pre-existing _reward_base_height): _reward_lin_vel_z,
_reward_ang_vel_xy, _reward_orientation. Pure tensor math — no physics
backend needed — but legged_robot.py itself can't be imported without one
(LeggedRobot's BaseTask import chain pulls in every simulator backend, same
issue test_push_direction.py's own docstring describes for Simulator).

So this extracts just those three method bodies straight from the real
source via `ast` (not hand-copied into this file, which would silently
drift from the actual implementation) and execs them into a minimal
stand-in object exposing exactly the attributes they read
(self.simulator.*, self.cfg.rewards.*) — same spirit as
test_push_direction.py's _StubSimulator, one level more minimal since these
three reward methods don't need a real Simulator at all.

Run directly: python tests/test_reward_targets.py
"""
import ast
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402 - after sys.path setup, matching sibling tests' style

LEGGED_ROBOT_PY = ROOT / "legged_gym" / "envs" / "base" / "legged_robot.py"
TARGET_METHODS = ("_reward_lin_vel_z", "_reward_ang_vel_xy", "_reward_orientation")


def _extract_methods():
    source = LEGGED_ROBOT_PY.read_text()
    tree = ast.parse(source, filename=str(LEGGED_ROBOT_PY))
    class_node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "LeggedRobot")
    # legged_robot.py compiles under `from __future__ import annotations`
    # (annotations kept as strings, never evaluated) — this segment is
    # compiled standalone without that future import active, so the "->
    # Reward" return annotation IS evaluated eagerly; just needs to resolve
    # to something, its actual value is irrelevant here.
    ns = {"torch": torch, "Reward": torch.Tensor}
    found = {}
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_METHODS:
            segment = textwrap.dedent(ast.get_source_segment(source, node))
            exec(compile(segment, filename=str(LEGGED_ROBOT_PY), mode="exec"), ns)  # noqa: S102
            found[node.name] = ns[node.name]
    missing = set(TARGET_METHODS) - found.keys()
    assert not missing, f"reward method(s) not found in legged_robot.py: {missing}"
    return found


REWARD_METHODS = _extract_methods()


class _Obj:
    """Plain attribute bag — object() itself doesn't allow attribute assignment."""


def _make_self(lin_vel_z=0.0, ang_vel_xy=(0.0, 0.0), grav_xy=(0.0, 0.0), **targets):
    self_obj = _Obj()
    self_obj.simulator = _Obj()
    self_obj.simulator.base_lin_vel = torch.tensor([[0.0, 0.0, lin_vel_z]])
    self_obj.simulator.base_ang_vel = torch.tensor([[ang_vel_xy[0], ang_vel_xy[1], 0.0]])
    self_obj.simulator.projected_gravity = torch.tensor([[grav_xy[0], grav_xy[1], -1.0]])
    self_obj.cfg = _Obj()
    self_obj.cfg.rewards = _Obj()
    for key, value in targets.items():
        setattr(self_obj.cfg.rewards, key, value)
    return self_obj


class TestNoTargetConfiguredMatchesOldAlwaysZeroBehavior(unittest.TestCase):
    """Whether or not a task's cfg.rewards even defines the new *_target
    attribute (getattr(..., 0.0) covers both), the reward must reproduce
    the exact pre-existing sum-of-squares-toward-zero math — the entire
    point of defaulting to 0.0 was zero behavior change for every task that
    doesn't opt in to a non-default target."""

    def test_lin_vel_z_attribute_missing_entirely(self):
        self_obj = _make_self(lin_vel_z=0.7)  # no lin_vel_z_target set at all
        rew = REWARD_METHODS["_reward_lin_vel_z"](self_obj)
        self.assertAlmostEqual(rew.item(), 0.7 ** 2, places=5)

    def test_lin_vel_z_target_explicitly_zero(self):
        self_obj = _make_self(lin_vel_z=0.7, lin_vel_z_target=0.0)
        rew = REWARD_METHODS["_reward_lin_vel_z"](self_obj)
        self.assertAlmostEqual(rew.item(), 0.7 ** 2, places=5)

    def test_ang_vel_xy_attribute_missing_entirely(self):
        self_obj = _make_self(ang_vel_xy=(0.3, -0.4))
        rew = REWARD_METHODS["_reward_ang_vel_xy"](self_obj)
        self.assertAlmostEqual(rew.item(), 0.3 ** 2 + 0.4 ** 2, places=5)

    def test_orientation_attribute_missing_entirely(self):
        self_obj = _make_self(grav_xy=(0.1, 0.2))
        rew = REWARD_METHODS["_reward_orientation"](self_obj)
        self.assertAlmostEqual(rew.item(), 0.1 ** 2 + 0.2 ** 2, places=5)


class TestConfiguredTargetIsHonored(unittest.TestCase):
    def test_lin_vel_z_matching_target_gives_zero_reward(self):
        self_obj = _make_self(lin_vel_z=0.5, lin_vel_z_target=0.5)
        rew = REWARD_METHODS["_reward_lin_vel_z"](self_obj)
        self.assertAlmostEqual(rew.item(), 0.0, places=5)

    def test_lin_vel_z_penalizes_deviation_from_nonzero_target(self):
        self_obj = _make_self(lin_vel_z=0.5, lin_vel_z_target=0.2)
        rew = REWARD_METHODS["_reward_lin_vel_z"](self_obj)
        self.assertAlmostEqual(rew.item(), (0.5 - 0.2) ** 2, places=5)

    def test_ang_vel_xy_target_is_a_magnitude_not_a_per_axis_vector(self):
        # |[0.3, 0.4]| == 0.5 regardless of the split between the two axes —
        # a magnitude target, matching what Live Telemetry shows as one
        # number, not a per-axis one (see VARIABLE_REGISTRY's "Roll/pitch
        # rate" entry in training.py).
        self_obj = _make_self(ang_vel_xy=(0.3, 0.4), ang_vel_xy_target=0.5)
        rew = REWARD_METHODS["_reward_ang_vel_xy"](self_obj)
        self.assertAlmostEqual(rew.item(), 0.0, places=5)

    def test_orientation_target_is_a_magnitude_not_a_per_axis_vector(self):
        self_obj = _make_self(grav_xy=(0.0, 0.3), orientation_tilt_target=0.3)
        rew = REWARD_METHODS["_reward_orientation"](self_obj)
        self.assertAlmostEqual(rew.item(), 0.0, places=5)

    def test_orientation_penalizes_deviation_from_nonzero_tilt_target(self):
        # magnitude sqrt(0.3^2+0.4^2) = 0.5, target 0.2 -> (0.5-0.2)^2
        self_obj = _make_self(grav_xy=(0.3, 0.4), orientation_tilt_target=0.2)
        rew = REWARD_METHODS["_reward_orientation"](self_obj)
        self.assertAlmostEqual(rew.item(), 0.3 ** 2, places=5)


if __name__ == "__main__":
    unittest.main()
