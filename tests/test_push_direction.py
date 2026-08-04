#!/usr/bin/env python3
"""
Unit test for Simulator.sample_push_vel_xy() (legged_gym/simulator/simulator.py)
— pure tensor math, no physics backend needed.

legged_gym/__init__.py and legged_gym/{envs,simulator,utils}/__init__.py all
eagerly import a real physics backend (Genesis/IsaacGym/IsaacLab, plus PIL,
warp, onnxruntime...) as a side effect of package import, even though the
handful of leaf modules this test actually needs (legged_robot_config.py,
base_config.py, math_utils.py, simulator.py) have none of those dependencies
themselves. So we install empty namespace-package stand-ins for the
intermediate packages (pointing at the real directories via __path__) and let
Python's normal file-based import resolve the leaf modules through them —
this runs without any physics backend installed.

Run directly: python tests/test_push_direction.py
"""
import math
import sys
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


for _pkg in ("legged_gym", "legged_gym.envs", "legged_gym.envs.base",
             "legged_gym.simulator", "legged_gym.utils"):
    _stub_package(_pkg)

import torch

from legged_gym.simulator.simulator import PUSH_DIR_BIAS_DEG, Simulator
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class _StubSimulator(Simulator):
    """Exercises sample_push_vel_xy() without going through Simulator.__init__
    (which drives a real physics backend). Simulator declares many more
    @abstractmethod hooks than this test needs (torque computation, terrain,
    domain randomization, ...) — clearing __abstractmethods__ below lets us
    instantiate without stubbing all of them out."""

    def __init__(self, num_envs: int, max_push_vel_xy: float, push_dir, yaw: float):
        self._num_envs = num_envs
        self._device = "cpu"
        cfg = LeggedRobotCfg()
        cfg.domain_rand.max_push_vel_xy = max_push_vel_xy
        cfg.domain_rand.push_dir = push_dir
        self._cfg = cfg
        half = yaw / 2.0
        # (num_envs, 4) xyzw quaternion for a pure yaw rotation.
        self._base_quat = torch.tensor(
            [[0.0, 0.0, math.sin(half), math.cos(half)]] * num_envs
        )

    # Unused abstract methods — not exercised by this test.
    def step(self): ...
    def post_physics_step(self): ...
    def reset_idx(self, env_ids): ...
    def reset_dofs(self, env_ids, dof_pos, dof_vel): ...
    def reset_root_states(self, env_ids, base_pos, base_quat, base_lin_vel_w, base_ang_vel_w): ...
    def update_terrain_curriculum(self, env_ids, move_up, move_down): ...
    def push_robots(self): ...
    def push_links(self): ...
    def draw_debug_vis(self): ...
    def set_viewer_camera(self, eye, target): ...


_StubSimulator.__abstractmethods__ = frozenset()


class TestPushDirection(unittest.TestCase):
    NUM_ENVS = 256
    MAX_PUSH_VEL = 1.5

    def _sample(self, push_dir, yaw=0.0):
        sim = _StubSimulator(self.NUM_ENVS, self.MAX_PUSH_VEL, push_dir, yaw)
        return sim.sample_push_vel_xy()

    def test_unset_push_dir_matches_original_isotropic_sampling(self):
        vel = self._sample(None)
        self.assertEqual(vel.shape, (self.NUM_ENVS, 2))
        self.assertTrue(torch.all(vel >= -self.MAX_PUSH_VEL))
        self.assertTrue(torch.all(vel <= self.MAX_PUSH_VEL))
        # Isotropic sampling covers both signs on each axis given enough envs.
        self.assertTrue((vel[:, 0] < 0).any() and (vel[:, 0] > 0).any())
        self.assertTrue((vel[:, 1] < 0).any() and (vel[:, 1] > 0).any())

    def test_unknown_push_dir_raises(self):
        with self.assertRaises(ValueError):
            self._sample("sideways")

    def _assert_biased_toward(self, vel, expected_local_angle_rad, yaw):
        # World-frame angle of each sampled push vector.
        angles = torch.atan2(vel[:, 1], vel[:, 0])
        expected_world_angle = expected_local_angle_rad + yaw
        # Wrap difference to [-pi, pi] and check it stays within the sampling
        # cone (PUSH_DIR_SPREAD_DEG / 2) plus a small numerical margin.
        diff = (angles - expected_world_angle + math.pi) % (2 * math.pi) - math.pi
        max_allowed = math.radians(60.0) / 2.0 + 1e-3
        self.assertTrue(torch.all(diff.abs() <= max_allowed))

    def test_directions_at_zero_yaw(self):
        for name, bias_deg in PUSH_DIR_BIAS_DEG.items():
            vel = self._sample(name, yaw=0.0)
            self._assert_biased_toward(vel, math.radians(bias_deg), yaw=0.0)

    def test_directions_respect_robot_heading(self):
        # Facing +90 degrees yaw: 'behind' should now push along world +y,
        # not world +x — confirms the cone rotates with the robot instead of
        # staying fixed in world frame.
        yaw = math.pi / 2
        for name, bias_deg in PUSH_DIR_BIAS_DEG.items():
            vel = self._sample(name, yaw=yaw)
            self._assert_biased_toward(vel, math.radians(bias_deg), yaw=yaw)


if __name__ == "__main__":
    unittest.main()
