"""
SafetyGovernor — the only component allowed to decide "yes, act on that
pending policy switch now" or "no, this state looks wrong, fall back to a
safe default." Kept separate from PolicySupervisor on purpose (see Fable's
review in the project history): the swap mechanics and the "is this a safe
moment" judgment are different concerns, and on real hardware the second one
is non-negotiable in a way that shouldn't be buried inside generic swap code.

This is intentionally simple for v1 — a couple of thresholds — because the
interface (`tick()`) is what matters: it's the seam where a fork can later
plug in a real learned recovery policy or a smarter trip condition without
anyone above this layer noticing.
"""
from __future__ import annotations

import torch

from .adapter import RobotState, Lifecycle
from .supervisor import PolicySupervisor


class SafetyGovernor:
    def __init__(
        self,
        supervisor: PolicySupervisor,
        max_projected_gravity_z: float = 0.7,
        damping_policy_name: str = "damping",
    ):
        """
        max_projected_gravity_z: same threshold family legged_robot.py uses
            for its own fall-termination check — projected_gravity[:,2] near
            -1 means upright, near/above this threshold means tipped over.
        damping_policy_name: if present in supervisor.policies, this is what
            the governor switches to (bypassing the normal safe-moment gate,
            since "we already detected a fall" IS the emergency) when
            tripped. If absent, tripping just holds the current policy and
            flags FAULT for the caller to notice.
        """
        self.supervisor = supervisor
        self.max_projected_gravity_z = max_projected_gravity_z
        self.damping_policy_name = damping_policy_name
        self.tripped = False

    def is_safe_to_switch(self, state: RobotState) -> bool:
        """Conservative on purpose: only allow a switch while roughly
        upright and not already mid-emergency. A learned "is this a good
        moment to hand off" classifier is a reasonable v2 here — see
        selector.py for where the equivalent proposal-side logic lives."""
        if self.tripped:
            return False
        if state.lifecycle == Lifecycle.FAULT:
            return False
        upright = state.projected_gravity[:, 2].max().item() < self.max_projected_gravity_z
        return upright

    def tick(self, state: RobotState) -> RobotState:
        """Call once per control tick, after reading state and before asking
        the supervisor to act. Handles: (a) tripping (fall or NaN) and
        forcing a hand-off to the damping fallback — not just flagging it,
        actually switching, since a `tripped` flag nobody acts on is not a
        safety mechanism; (b) approving a pending switch if one is queued
        and the moment is safe. Once tripped, stays tripped (and stays on
        the damping skill) until reset() is called explicitly — see
        legged_gym/scripts/rugiar_driver.py's Restart handler."""
        fallen = state.projected_gravity[:, 2].max().item() >= self.max_projected_gravity_z
        nan_detected = torch.isnan(state.dof_pos).any().item() or torch.isnan(state.base_quat).any().item()

        if (fallen or nan_detected) and not self.tripped:
            self.tripped = True

        if self.tripped:
            # Not just flagging FAULT for someone else to notice — actually
            # force the hand-off to the fallback skill every tick until
            # reset(), regardless of what caused the trip (a real fall, a
            # NaN, or ControlService.estop() setting self.tripped directly).
            if (self.damping_policy_name in self.supervisor.policies
                    and self.supervisor.active_name != self.damping_policy_name):
                self.supervisor.request_switch(self.damping_policy_name)
                self.supervisor.confirm_pending_switch()
            return state

        if self.supervisor.pending_name is not None and self.is_safe_to_switch(state):
            self.supervisor.confirm_pending_switch()

        return state

    def reset(self) -> None:
        self.tripped = False
