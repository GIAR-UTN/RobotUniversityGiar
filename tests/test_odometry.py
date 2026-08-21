#!/usr/bin/env python3
"""ControlService.get_odometry() — merged in from the (previously unmerged,
stale) `mcp-base` branch's rugiar_mcp/ work, done surgically rather than a
raw `git merge` because that branch predates this session's mjlab work
entirely: RobotState gained a new required field (`base_pos_xy`), so every
constructor of it needed a matching update, including
legged_gym/control/mjlab_adapter.py, which doesn't exist in mcp-base's
history at all and would have been silently left broken (missing a
required dataclass field, no merge conflict to flag it) by a literal merge.

Covers: distance/time/speed accumulation over ticks, teleport/reset
re-baselining, and `None` (unavailable) on a backend with no
base_pos_xy (e.g. RealAdapter).

Run:
    SIMULATOR=genesis .venv/bin/python -m pytest tests/test_odometry.py -q
    SIMULATOR=mjlab CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest tests/test_odometry.py -q
"""
import unittest
from unittest import mock

import torch

from legged_gym.control import service as service_mod
from legged_gym.control.adapter import RobotState, Lifecycle
from legged_gym.control.service import ControlService


class FakeAdapter:
    """Only get_state() matters to get_odometry() -- returns whatever
    RobotState this test queues up next via `states`."""

    def __init__(self, states):
        self._states = list(states)

    def get_state(self):
        return self._states.pop(0) if len(self._states) > 1 else self._states[0]


def _state(x, y, base_pos_xy_available=True):
    pos = torch.tensor([[x, y]]) if base_pos_xy_available else None
    return RobotState(
        dof_pos=torch.zeros(1, 12), dof_vel=torch.zeros(1, 12),
        default_dof_pos=torch.zeros(1, 12), base_quat=torch.zeros(1, 4),
        base_ang_vel=torch.zeros(1, 3), base_lin_vel=torch.zeros(1, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        base_height=torch.zeros(1), base_pos_xy=pos,
        commands=torch.zeros(1, 3), action_scale=0.25, lifecycle=Lifecycle.ACTIVE,
    )


def _service(states):
    return ControlService(FakeAdapter(states), supervisor=object(), safety=object())


class TestGetOdometry(unittest.TestCase):
    """time.time() is mocked with a deterministic 1s-per-call clock
    throughout -- a real wall-clock delta between two calls milliseconds
    apart rounds to 0.000 at get_odometry()'s own 3-decimal precision,
    which is a test-speed artifact, not something the code should hide
    behind (get_odometry() itself is correct either way)."""

    def _clock(self, start=1000.0, step=1.0):
        ticks = iter(start + step * i for i in range(1000))
        return mock.patch.object(service_mod.time, "time", side_effect=lambda: next(ticks))

    def test_first_call_baselines_at_zero(self):
        svc = _service([_state(0.0, 0.0)])
        with self._clock():
            odo = svc.get_odometry()
        self.assertEqual(odo, {"distance_traveled": 0.0, "time_elapsed": 0.0, "average_speed": 0.0})

    def test_accumulates_distance_across_calls(self):
        svc = _service([_state(0.0, 0.0), _state(0.3, 0.4)])  # 3-4-5 triangle -> 0.5 exact
        with self._clock():
            svc.get_odometry()  # baseline
            odo = svc.get_odometry()
        self.assertAlmostEqual(odo["distance_traveled"], 0.5, places=3)
        self.assertGreater(odo["time_elapsed"], 0.0)

    def test_accumulates_over_multiple_steps(self):
        svc = _service([_state(0.0, 0.0), _state(0.1, 0.0), _state(0.2, 0.0), _state(0.3, 0.0)])
        with self._clock():
            svc.get_odometry()
            svc.get_odometry()
            svc.get_odometry()
            odo = svc.get_odometry()
        self.assertAlmostEqual(odo["distance_traveled"], 0.3, places=3)

    def test_large_jump_is_treated_as_reset_not_travel(self):
        """A >1.0m single-tick delta means restart()/family-switch/episode
        reset happened underneath, not real motion -- re-baseline instead
        of counting it."""
        svc = _service([_state(0.0, 0.0), _state(0.05, 0.0), _state(5.0, 5.0)])
        with self._clock():
            svc.get_odometry()
            odo = svc.get_odometry()
            self.assertAlmostEqual(odo["distance_traveled"], 0.05, places=3)
            odo = svc.get_odometry()  # the >1m jump
        self.assertEqual(odo["distance_traveled"], 0.0)  # re-baselined, not 5.0-ish

    def test_none_when_base_pos_xy_unavailable(self):
        """RealAdapter-style backend: base_pos_xy is None -- get_odometry()
        must return None (genuinely unavailable), not a fabricated zero."""
        svc = _service([_state(0.0, 0.0, base_pos_xy_available=False)])
        self.assertIsNone(svc.get_odometry())

    def test_average_speed_is_distance_over_time(self):
        svc = _service([_state(0.0, 0.0), _state(1.0, 0.0)])
        with self._clock():
            svc.get_odometry()
            odo = svc.get_odometry()
        expected = odo["distance_traveled"] / odo["time_elapsed"]
        self.assertAlmostEqual(odo["average_speed"], expected, places=3)


if __name__ == "__main__":
    unittest.main()
