#!/usr/bin/env python3
"""
Unit tests for SafetyGovernor (legged_gym/control/safety.py) — the sole gate
that approves a pending policy switch and the sole thing that forces a
hand-off to the damping fallback on a fall/NaN. These are the invariants
that would fail *silently* if broken: a bad threshold comparison, a trip
that doesn't latch, or a forced damping switch that quietly no-ops because
the fallback policy isn't loaded wouldn't raise anything on their own — the
robot would just do the wrong thing.

Uses the real PolicySupervisor (not a fake) throughout, since the entire
point of these tests is the SafetyGovernor <-> PolicySupervisor interaction
(request_switch + confirm_pending_switch called directly from tick()).
Same package-stub trick as tests/test_delete_policy.py, extended to
legged_gym.utils since safety.py pulls in adapter.py's RobotState, which
imports legged_gym.utils.math_utils — stubbing the parent package with its
real __path__ lets that submodule import directly without running
legged_gym/utils/__init__.py (which drags in rsl_rl and a SIMULATOR-backend
import legged_gym/__init__.py would otherwise require).

Run directly: python tests/test_safety_governor.py
"""
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_package(dotted_name: str):
    if dotted_name in sys.modules:
        return
    module = types.ModuleType(dotted_name)
    module.__path__ = [str(ROOT / Path(*dotted_name.split(".")))]
    module.__package__ = dotted_name
    sys.modules[dotted_name] = module


for _pkg in ("legged_gym", "legged_gym.control", "legged_gym.utils"):
    _stub_package(_pkg)

from legged_gym.control.adapter import Lifecycle, RobotState
from legged_gym.control.safety import SafetyGovernor
from legged_gym.control.supervisor import PolicySupervisor


class _FakeBackend:
    def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self):
        self.reset_calls += 1

    def step(self, obs):
        self.step_calls += 1
        return torch.zeros(1, 3)


class _FakeObsSpec:
    num_obs = 0  # 0 is falsy -> PolicySupervisor._check_obs_spec skips the shape check


class _FakePolicy:
    def __init__(self, name):
        self.name = name
        self.backend = _FakeBackend()
        self.obs_spec = _FakeObsSpec()


def _supervisor(names, active):
    return PolicySupervisor({n: _FakePolicy(n) for n in names}, active=active)


def _state(gravity_z: float, lifecycle=Lifecycle.ACTIVE, num_envs=1,
           dof_pos=None, base_quat=None):
    """Minimal RobotState — only the fields SafetyGovernor actually reads
    (projected_gravity, dof_pos, base_quat, lifecycle) need real values;
    everything else is an unused placeholder."""
    if dof_pos is None:
        dof_pos = torch.zeros(num_envs, 12)
    if base_quat is None:
        base_quat = torch.zeros(num_envs, 4)
    return RobotState(
        dof_pos=dof_pos,
        dof_vel=torch.zeros(num_envs, 12),
        default_dof_pos=torch.zeros(num_envs, 12),
        base_quat=base_quat,
        base_ang_vel=torch.zeros(num_envs, 3),
        base_lin_vel=torch.zeros(num_envs, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, gravity_z]] * num_envs),
        base_height=torch.zeros(num_envs),
        commands=torch.zeros(num_envs, 3),
        action_scale=0.25,
        lifecycle=lifecycle,
    )


class TestIsSafeToSwitch(unittest.TestCase):
    """The sole gate is_safe_to_switch() — what it actually gates on."""

    def test_upright_not_tripped_not_fault_is_safe(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        self.assertTrue(gov.is_safe_to_switch(_state(gravity_z=-1.0)))

    def test_past_threshold_is_unsafe(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        # Comparison is strictly "< threshold" (safety.py:53) — clearly past
        # the threshold must never read as upright.
        self.assertFalse(gov.is_safe_to_switch(_state(gravity_z=0.71)))
        self.assertFalse(gov.is_safe_to_switch(_state(gravity_z=0.9)))
        self.assertTrue(gov.is_safe_to_switch(_state(gravity_z=0.69)))

    def test_tripped_is_unsafe_even_if_upright(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gov.tripped = True
        self.assertFalse(gov.is_safe_to_switch(_state(gravity_z=-1.0)))

    def test_fault_lifecycle_is_unsafe_even_if_upright(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        self.assertFalse(gov.is_safe_to_switch(_state(gravity_z=-1.0, lifecycle=Lifecycle.FAULT)))

    def test_uses_worst_case_env_not_average(self):
        # Multi-env: one fallen env among several upright ones must still
        # read as unsafe — is_safe_to_switch uses .max(), not a mean.
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.9], [0.0, 0.0, -1.0]])
        state = _state(gravity_z=-1.0, num_envs=3)
        state.projected_gravity = gravity
        self.assertFalse(gov.is_safe_to_switch(state))


class TestTickFallAndNanDetection(unittest.TestCase):
    """tick() is the main loop: detect a fall/NaN, latch tripped, and force
    the hand-off to damping every tick thereafter — not just flag it."""

    def test_fall_trips_and_forces_switch_to_damping(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gov.tick(_state(gravity_z=0.9))  # fallen
        self.assertTrue(gov.tripped)
        self.assertEqual(sup.active_name, "damping")

    def test_upright_tick_does_not_trip(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gov.tick(_state(gravity_z=-1.0))
        self.assertFalse(gov.tripped)
        self.assertEqual(sup.active_name, "stable")

    def test_nan_in_dof_pos_trips(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        state = _state(gravity_z=-1.0, dof_pos=torch.tensor([[float("nan")] * 12]))
        gov.tick(state)
        self.assertTrue(gov.tripped)
        self.assertEqual(sup.active_name, "damping")

    def test_nan_in_base_quat_trips(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        state = _state(gravity_z=-1.0, base_quat=torch.tensor([[float("nan")] * 4]))
        gov.tick(state)
        self.assertTrue(gov.tripped)
        self.assertEqual(sup.active_name, "damping")

    def test_trip_latches_across_ticks_even_once_upright_again(self):
        # The whole point of the latch: no silent self-healing. Once
        # tripped, staying tripped is the only thing standing between "we
        # saw a fall" and "we forgot about it three ticks later."
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gov.tick(_state(gravity_z=0.9))  # fall
        self.assertTrue(gov.tripped)
        gov.tick(_state(gravity_z=-1.0))  # upright again, one tick later
        self.assertTrue(gov.tripped)
        self.assertEqual(sup.active_name, "damping")

    def test_reset_clears_trip_and_allows_switching_again(self):
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gov.tick(_state(gravity_z=0.9))
        self.assertTrue(gov.tripped)
        gov.reset()
        self.assertFalse(gov.tripped)
        self.assertTrue(gov.is_safe_to_switch(_state(gravity_z=-1.0)))

    def test_reset_while_still_fallen_retrips_on_next_tick(self):
        # reset() only clears the flag — it does not and cannot un-fall the
        # robot. If reset() is called (e.g. by an operator's Restart command)
        # before the robot is actually upright again, the very next tick
        # must re-trip rather than silently staying "safe".
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gov.tick(_state(gravity_z=0.9))
        gov.reset()
        gov.tick(_state(gravity_z=0.9))  # still fallen
        self.assertTrue(gov.tripped)
        self.assertEqual(sup.active_name, "damping")


class TestTickPendingSwitch(unittest.TestCase):
    """The other half of tick(): approving an already-queued switch."""

    def test_pending_switch_confirmed_when_safe(self):
        sup = _supervisor(["stable", "croucher", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        sup.request_switch("croucher")
        gov.tick(_state(gravity_z=-1.0))
        self.assertEqual(sup.active_name, "croucher")
        self.assertIsNone(sup.pending_name)

    def test_pending_switch_not_confirmed_when_unsafe(self):
        # Fallen AND a switch was already queued: the fall trip takes over
        # (forces damping) and the originally-requested switch must not
        # sneak through as a side effect.
        sup = _supervisor(["stable", "croucher", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        sup.request_switch("croucher")
        gov.tick(_state(gravity_z=0.9))
        self.assertEqual(sup.active_name, "damping")
        self.assertNotEqual(sup.active_name, "croucher")

    def test_pending_switch_held_while_fault_lifecycle_without_a_fall(self):
        # FAULT lifecycle with otherwise-upright telemetry: is_safe_to_switch
        # says no, but this isn't a fall/NaN trip, so tripped never gets
        # set — the pending request should simply stay queued, not get
        # silently dropped or force damping.
        sup = _supervisor(["stable", "croucher", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        sup.request_switch("croucher")
        gov.tick(_state(gravity_z=-1.0, lifecycle=Lifecycle.FAULT))
        self.assertFalse(gov.tripped)
        self.assertEqual(sup.active_name, "stable")
        self.assertEqual(sup.pending_name, "croucher")


class TestTickAlreadyOnDamping(unittest.TestCase):
    def test_no_redundant_switch_once_damping_is_already_active(self):
        # tick()'s forced hand-off is guarded by
        # `active_name != damping_policy_name` — once damping is already
        # active, further ticks while tripped must not keep re-requesting/
        # re-confirming a switch (which would restart the cross-fade ramp
        # every single tick and never let it finish).
        sup = _supervisor(["stable", "damping"], active="stable")
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gov.tick(_state(gravity_z=0.9))  # trips, switches to damping, ramp starts
        self.assertTrue(sup.is_ramping)
        sup._ramp_remaining = 1  # let the ramp actually finish naturally
        sup.step(torch.zeros(1, 3))
        self.assertFalse(sup.is_ramping)
        gov.tick(_state(gravity_z=0.9))  # still tripped, still fallen
        self.assertFalse(sup.is_ramping)  # must NOT have restarted the ramp


class TestMissingDampingPolicy(unittest.TestCase):
    def test_trip_with_no_damping_policy_loaded_does_not_crash_or_switch(self):
        # SafetyGovernor.__init__ never validates that damping_policy_name
        # actually exists in supervisor.policies (see safety.py's own
        # docstring: "if absent, tripping just holds the current policy").
        # Confirm that's really what happens — no crash, no switch — and
        # not, e.g., a KeyError from confirm_pending_switch reaching in for
        # a policy that was never registered.
        sup = _supervisor(["stable"], active="stable")  # no "damping" at all
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        gov.tick(_state(gravity_z=0.9))
        self.assertTrue(gov.tripped)
        self.assertEqual(sup.active_name, "stable")  # held in place, not switched
        self.assertIsNone(sup.pending_name)


class TestFallDuringRamp(unittest.TestCase):
    def test_fall_mid_ramp_abandons_old_ramp_and_ramps_to_damping_instead(self):
        # A fall detected while already mid cross-fade (stable -> croucher):
        # confirm_pending_switch always ramps FROM whatever's currently
        # "active" (which flips immediately on confirm, before the ramp
        # finishes) -- so the forced damping switch should cross-fade from
        # croucher (the mid-flight target), not from the original stable.
        sup = _supervisor(["stable", "croucher", "damping"], active="stable", )
        sup.ramp_ticks = 10
        gov = SafetyGovernor(sup, max_projected_gravity_z=0.7)
        sup.request_switch("croucher")
        gov.tick(_state(gravity_z=-1.0))  # confirms stable -> croucher, ramp starts
        self.assertEqual(sup.active_name, "croucher")
        self.assertTrue(sup.is_ramping)
        self.assertIs(sup._ramp_from, sup.policies["stable"])

        gov.tick(_state(gravity_z=0.9))  # fall mid-ramp
        self.assertEqual(sup.active_name, "damping")
        self.assertIs(sup._ramp_from, sup.policies["croucher"])


if __name__ == "__main__":
    unittest.main()
