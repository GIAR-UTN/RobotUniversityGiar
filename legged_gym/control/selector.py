"""
Selector — proposes policy switches autonomously. This is the "no web UI at
all, the robot decides for itself" half of the control system: it calls
`PolicySupervisor.request_switch()`, the exact same method a human clicking
a button in the browser calls. Same call site, same effect either way —
that's the whole point of routing both through PolicySupervisor instead of
letting the web layer poke the robot directly.

TiltRecoverySelector below is deliberately a simple rule, not a learned
network — the 2025-2026 research direction for this (see README: RPG,
SkillBlender) is a learned gating network that blends/selects among skills
continuously. That's a real v2: drop in a class implementing the same
`propose(state) -> Optional[str]` signature and nothing else has to change.
"""
from __future__ import annotations

from typing import Optional, Protocol

from .adapter import RobotState


class Selector(Protocol):
    def propose(self, state: RobotState) -> Optional[str]:
        """Return the name of the policy that should be active given this
        state, or None to mean "no opinion, leave it as is". Called once per
        tick by whoever's driving the autonomous loop (see
        legged_gym/scripts/rugiar_driver.py for the reference wiring)."""
        ...


class TiltRecoverySelector:
    """The simplest useful autonomous rule: if the robot is tipping past a
    threshold, propose switching to a steadier/cautious skill; otherwise
    propose the normal one. Good enough to prove the "no web needed" path
    end-to-end; not a substitute for a real learned selector.

    Known limitation: this always has an opinion (never returns None), so if
    you wire a live Selector into ControlService alongside a human operator,
    it will re-propose and override a manual switch on the very next tick.
    No hysteresis/deadband is implemented yet — a real deployment wants
    either a "human override wins for N seconds" rule or a smarter selector
    that returns None once its own preferred policy is already active and
    conditions haven't changed."""

    def __init__(self, tilt_threshold: float, recovery_policy: str, normal_policy: str):
        self.tilt_threshold = tilt_threshold
        self.recovery_policy = recovery_policy
        self.normal_policy = normal_policy

    def propose(self, state: RobotState) -> Optional[str]:
        tilt = state.projected_gravity[:, 2].max().item()
        return self.recovery_policy if tilt > self.tilt_threshold else self.normal_policy
