"""
PolicySupervisor — owns the loaded skills and the mechanics of switching
between them. Deliberately does NOT decide *when* it's safe to act on a
switch request — that call belongs to SafetyGovernor (safety.py). This
keeps "can we switch right now" as a single, auditable decision point
instead of being scattered across whichever code happened to request it.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch

from .policy import Policy


class PolicySupervisor:
    def __init__(self, policies: Dict[str, Policy], active: str, ramp_ticks: int = 15):
        if active not in policies:
            raise KeyError(f"'{active}' is not one of the loaded policies: {list(policies)}")
        self.policies = policies
        self.active_name = active
        self.pending_name: Optional[str] = None
        self.ramp_ticks = ramp_ticks
        self._ramp_remaining = 0
        self._ramp_from: Optional[Policy] = None

        for p in policies.values():
            p.backend.reset()

    @property
    def active(self) -> Policy:
        return self.policies[self.active_name]

    @property
    def is_ramping(self) -> bool:
        return self._ramp_remaining > 0

    def request_switch(self, name: str) -> bool:
        """Called by a human (web/GUI button) or an autonomous Selector —
        same method, same effect, regardless of caller. Only records intent;
        does not touch the robot. Returns False if the request is a no-op
        (already active, or already the pending target)."""
        if name not in self.policies:
            raise KeyError(f"Unknown policy '{name}'. Loaded: {list(self.policies)}")
        if name == self.active_name and not self.is_ramping:
            return False
        if name == self.pending_name:
            return False
        self.pending_name = name
        return True

    def cancel_pending_switch(self) -> None:
        self.pending_name = None

    def add_policy(self, policy: Policy) -> None:
        """Registers a newly-trained policy into the running supervisor —
        the hot-load counterpart to loading policies at construction time.
        Does not touch active/pending state; the new policy is simply
        selectable from here on, same as any policy loaded at startup."""
        policy.backend.reset()
        self.policies[policy.name] = policy

    def remove_policy(self, name: str) -> None:
        """Discards a loaded policy — e.g. a training experiment that
        converged to something useless (see ControlService.delete_policy(),
        which also drops it from the clone-from catalog and deletes its
        checkpoint file). Refuses to remove 'damping' (SafetyGovernor
        assumes it always exists), the currently active policy, or one
        with a switch already pending — any of those would leave the
        supervisor pointing at a name that no longer exists. The caller
        has to switch away first."""
        if name == "damping":
            raise ValueError("'damping' is the safety fallback and can't be removed")
        if name not in self.policies:
            raise KeyError(f"Unknown policy '{name}'. Loaded: {list(self.policies)}")
        if name == self.active_name:
            raise ValueError(f"can't remove '{name}' — it's the active policy; switch to another one first")
        if name == self.pending_name:
            raise ValueError(f"can't remove '{name}' — a switch to it is already pending")
        del self.policies[name]

    def confirm_pending_switch(self) -> bool:
        """SafetyGovernor calls this — and ONLY this — once it has decided
        the current instant is safe to switch. Begins a linear cross-fade
        over `ramp_ticks` steps instead of a hard cut, so the PD controller
        sees a gradually-moving target instead of an instantaneous jump."""
        if self.pending_name is None:
            return False
        new_policy = self.policies[self.pending_name]
        new_policy.backend.reset()
        self._ramp_from = self.active
        self.active_name = self.pending_name
        self.pending_name = None
        self._ramp_remaining = self.ramp_ticks
        return True

    def _check_obs_spec(self, policy: Policy, obs: torch.Tensor) -> None:
        expected = policy.obs_spec.num_obs
        if expected and obs.shape[-1] != expected:
            warnings.warn(
                f"Policy '{policy.name}' expects {expected}-dim observations but got "
                f"{obs.shape[-1]}. All loaded policies must currently share one observation "
                f"space — this is exactly the failure mode ObsSpec exists to catch.",
                stacklevel=3,
            )

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        self._check_obs_spec(self.active, obs)
        new_action = self.active.backend.step(obs)
        if self._ramp_remaining <= 0:
            return new_action

        # Blend with the outgoing policy's action for a smooth handoff. The
        # outgoing policy keeps ticking (its LSTM state keeps advancing) for
        # the duration of the ramp — simplest correct behavior; freezing its
        # state instead is a reasonable future refinement, not required for
        # this to be safe.
        self._check_obs_spec(self._ramp_from, obs)
        old_action = self._ramp_from.backend.step(obs)
        alpha = 1.0 - (self._ramp_remaining / self.ramp_ticks)  # 0 -> 1 across the ramp
        action = (1.0 - alpha) * old_action + alpha * new_action

        self._ramp_remaining -= 1
        if self._ramp_remaining == 0:
            self._ramp_from = None
        return action

    @property
    def status(self) -> dict:
        return {
            "active": self.active_name,
            "pending": self.pending_name,
            "ramping": self.is_ramping,
        }
