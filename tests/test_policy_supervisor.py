#!/usr/bin/env python3
"""
Unit tests for PolicySupervisor's switch/ramp mechanics
(legged_gym/control/supervisor.py) — confirm_pending_switch() (where a
switch actually takes effect) and step()'s cross-fade blending (the thing
standing between "policy switch" and "the PD controller sees an
instantaneous jump"). remove_policy()/rename_policy() already have dedicated
coverage in tests/test_delete_policy.py and tests/test_rename_policy.py;
this file is deliberately scoped to the switch/ramp invariants those don't
touch. Same package-stub trick as those files.

Run directly: python tests/test_policy_supervisor.py
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


for _pkg in ("legged_gym", "legged_gym.control"):
    _stub_package(_pkg)

from legged_gym.control.supervisor import PolicySupervisor


class _FakeObsSpec:
    def __init__(self, num_obs=0):
        self.num_obs = num_obs


class _FakeBackend:
    """Returns a constant action and counts calls, so tests can tell
    exactly which policies stepped, how many times, and verify the blend
    math against known constants."""

    def __init__(self, action_value: float):
        self.action = torch.full((1, 3), action_value)
        self.step_calls = 0
        self.reset_calls = 0

    def step(self, obs):
        self.step_calls += 1
        return self.action

    def reset(self):
        self.reset_calls += 1


class _FakePolicy:
    def __init__(self, name, action_value=0.0, num_obs=0):
        self.name = name
        self.backend = _FakeBackend(action_value)
        self.obs_spec = _FakeObsSpec(num_obs)


def _supervisor(policies_by_name, active, ramp_ticks=4):
    return PolicySupervisor(dict(policies_by_name), active=active, ramp_ticks=ramp_ticks)


class TestConfirmPendingSwitch(unittest.TestCase):
    def test_no_pending_is_a_no_op(self):
        stable = _FakePolicy("stable")
        sup = _supervisor({"stable": stable}, active="stable")
        self.assertFalse(sup.confirm_pending_switch())
        self.assertEqual(sup.active_name, "stable")
        self.assertFalse(sup.is_ramping)

    def test_confirm_switches_active_and_clears_pending(self):
        stable, croucher = _FakePolicy("stable"), _FakePolicy("croucher")
        sup = _supervisor({"stable": stable, "croucher": croucher}, active="stable")
        sup.request_switch("croucher")
        self.assertTrue(sup.confirm_pending_switch())
        self.assertEqual(sup.active_name, "croucher")
        self.assertIsNone(sup.pending_name)

    def test_confirm_resets_the_new_policys_backend(self):
        # Backend reset (e.g. zeroing LSTM hidden state) happens at confirm
        # time, not at request time — a policy sitting idle as a pending
        # target must not be carrying stale hidden state from a previous
        # activation once it actually takes over.
        stable, croucher = _FakePolicy("stable"), _FakePolicy("croucher")
        sup = _supervisor({"stable": stable, "croucher": croucher}, active="stable")
        resets_at_construction = croucher.backend.reset_calls
        sup.request_switch("croucher")
        self.assertEqual(croucher.backend.reset_calls, resets_at_construction)  # not yet
        sup.confirm_pending_switch()
        self.assertEqual(croucher.backend.reset_calls, resets_at_construction + 1)

    def test_confirm_starts_a_ramp_from_the_previously_active_policy(self):
        stable, croucher = _FakePolicy("stable"), _FakePolicy("croucher")
        sup = _supervisor({"stable": stable, "croucher": croucher}, active="stable", ramp_ticks=5)
        sup.request_switch("croucher")
        sup.confirm_pending_switch()
        self.assertTrue(sup.is_ramping)
        self.assertIs(sup._ramp_from, stable)
        self.assertEqual(sup._ramp_remaining, 5)


class TestStepRamping(unittest.TestCase):
    def test_no_ramp_returns_active_policys_action_directly(self):
        stable = _FakePolicy("stable", action_value=1.0)
        sup = _supervisor({"stable": stable}, active="stable")
        action = sup.step(torch.zeros(1, 3))
        self.assertTrue(torch.equal(action, torch.full((1, 3), 1.0)))

    def test_first_ramp_step_is_entirely_old_action(self):
        # alpha = steps_done / (ramp_ticks - 1); on the very first step after
        # confirm(), steps_done == 0, so alpha == 0 and the blend is 100%
        # old / 0% new — the ramp starts exactly at the outgoing policy.
        old, new = _FakePolicy("stable", action_value=0.0), _FakePolicy("croucher", action_value=10.0)
        sup = _supervisor({"stable": old, "croucher": new}, active="stable", ramp_ticks=4)
        sup.request_switch("croucher")
        sup.confirm_pending_switch()
        action = sup.step(torch.zeros(1, 3))
        self.assertTrue(torch.equal(action, torch.full((1, 3), 0.0)))

    def test_blend_is_linear_across_the_ramp(self):
        old, new = _FakePolicy("stable", action_value=0.0), _FakePolicy("croucher", action_value=10.0)
        sup = _supervisor({"stable": old, "croucher": new}, active="stable", ramp_ticks=4)
        sup.request_switch("croucher")
        sup.confirm_pending_switch()
        expected_alphas = [0.0, 1 / 3, 2 / 3, 1.0]  # alpha = steps_done/(ramp_ticks-1), steps_done = 0,1,2,3
        for alpha in expected_alphas:
            action = sup.step(torch.zeros(1, 3))
            expected_value = 10.0 * alpha
            self.assertAlmostEqual(action[0, 0].item(), expected_value, places=5)

    def test_ramp_ends_after_ramp_ticks_steps_and_returns_pure_new_action(self):
        old, new = _FakePolicy("stable", action_value=0.0), _FakePolicy("croucher", action_value=10.0)
        sup = _supervisor({"stable": old, "croucher": new}, active="stable", ramp_ticks=4)
        sup.request_switch("croucher")
        sup.confirm_pending_switch()
        for _ in range(4):
            sup.step(torch.zeros(1, 3))
        self.assertFalse(sup.is_ramping)
        self.assertIsNone(sup._ramp_from)
        action = sup.step(torch.zeros(1, 3))
        self.assertTrue(torch.equal(action, torch.full((1, 3), 10.0)))

    def test_outgoing_policy_keeps_stepping_for_the_full_ramp(self):
        # Documented behavior (supervisor.py's own comment): the outgoing
        # policy's backend keeps advancing (e.g. its LSTM state) for the
        # duration of the ramp rather than being frozen.
        old, new = _FakePolicy("stable", action_value=0.0), _FakePolicy("croucher", action_value=10.0)
        sup = _supervisor({"stable": old, "croucher": new}, active="stable", ramp_ticks=4)
        sup.request_switch("croucher")
        sup.confirm_pending_switch()
        for _ in range(4):
            sup.step(torch.zeros(1, 3))
        self.assertEqual(old.backend.step_calls, 4)
        self.assertEqual(new.backend.step_calls, 4)

    def test_ramp_reaches_pure_new_smoothly_without_a_snap(self):
        # Fixed bug (was: the docstring promises a "0 -> 1" cross-fade, but
        # because ramp_remaining was decremented *after* alpha was computed,
        # the highest alpha ever produced during the ramp was
        # (ramp_ticks-1)/ramp_ticks, not 1.0 — the following call then
        # snapped straight to the pure new action instead of continuing the
        # same linear slope). Now the last in-ramp step itself reaches
        # alpha=1.0, so the call right after the ramp ends returns the exact
        # same value — no discontinuity.
        old, new = _FakePolicy("stable", action_value=0.0), _FakePolicy("croucher", action_value=10.0)
        sup = _supervisor({"stable": old, "croucher": new}, active="stable", ramp_ticks=4)
        sup.request_switch("croucher")
        sup.confirm_pending_switch()
        actions = [sup.step(torch.zeros(1, 3))[0, 0].item() for _ in range(4)]
        self.assertAlmostEqual(actions[-1], 10.0, places=5)  # last in-ramp step already 100% new
        final_action = sup.step(torch.zeros(1, 3))
        self.assertEqual(final_action[0, 0].item(), 10.0)  # same value, no snap

    def test_obs_spec_mismatch_warns_for_both_outgoing_and_incoming_policy(self):
        old = _FakePolicy("stable", action_value=0.0, num_obs=5)
        new = _FakePolicy("croucher", action_value=10.0, num_obs=5)
        sup = _supervisor({"stable": old, "croucher": new}, active="stable", ramp_ticks=2)
        sup.request_switch("croucher")
        sup.confirm_pending_switch()
        with self.assertWarns(UserWarning):
            sup.step(torch.zeros(1, 3))  # obs has 3 dims, both policies expect 5


if __name__ == "__main__":
    unittest.main()
