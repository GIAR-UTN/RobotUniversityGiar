"""
ControlService — the one call surface. A viser button callback calls
`service.request_switch("cautious")`. An autonomous Selector loop calls the
exact same method. A future WebSocket/HTTP bridge for controlling a real,
UI-less robot from an external web app would also just call this method —
it would be a thin transport wrapper around this class, not a parallel
implementation.

Today this class is used in-process (see legged_gym/scripts/swap_experiment.py):
viser's GUI callbacks call straight into it, no network hop needed for a
local sim demo. The moment you want to drive this from a *different*
process or machine (a real robot with no local display, an external web
app), wrap this same class with a tiny JSON-RPC-ish layer over WebSocket —
switch/status/pause/estop is the whole surface, per the architecture
write-up in the README.
"""
from __future__ import annotations

from typing import Optional

import torch

from .adapter import RobotAdapter, Lifecycle
from .safety import SafetyGovernor
from .selector import Selector
from .supervisor import PolicySupervisor


class ControlService:
    def __init__(
        self,
        adapter: RobotAdapter,
        supervisor: PolicySupervisor,
        safety: SafetyGovernor,
        selector: Optional[Selector] = None,
    ):
        self.adapter = adapter
        self.supervisor = supervisor
        self.safety = safety
        self.selector = selector
        self.paused = False

    # ---- the "human or autonomous, same call" surface ----

    def request_switch(self, name: str) -> bool:
        return self.supervisor.request_switch(name)

    def status(self) -> dict:
        s = self.supervisor.status
        s["paused"] = self.paused
        s["safety_tripped"] = self.safety.tripped
        # Every user-selectable policy name — "damping" is the safety
        # fallback skill, not a switch target, so it's excluded the same
        # way swap_experiment.py's viser panel already excludes it.
        s["policies"] = [name for name in self.supervisor.policies if name != "damping"]
        # Adapter-declared, not UI-hardcoded — see SimAdapter/RealAdapter's
        # backend_name/capabilities class attributes. Lets a control web
        # show the same panel for sim and real, graying out what the
        # current backend can't do (e.g. "restart" on real hardware).
        s["backend"] = getattr(self.adapter, "backend_name", "sim")
        s["capabilities"] = getattr(self.adapter, "capabilities", {})
        return s

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def estop(self) -> None:
        """Emergency stop. Trips safety (which forces the damping fallback —
        see SafetyGovernor.tick — every tick from here on, not just once),
        AND calls the adapter's own estop() if it has one. On SimAdapter
        that's just a lifecycle flag; on RealAdapter it's a real, immediate
        zero-torque write over DDS — the one action that must not depend on
        anything in this file working correctly."""
        self.safety.tripped = True
        adapter_estop = getattr(self.adapter, "estop", None)
        if adapter_estop is not None:
            adapter_estop()

    # ---- the per-tick driving loop ----

    def tick(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        """One control step. Returns None if paused (caller should hold
        position / not call adapter.send_action this tick).

        Note: while safety.tripped, this still returns an action — but
        safety.tick() below has already forced the supervisor onto the
        damping (zero-action) skill and refuses to confirm any other
        pending switch, so "still returns an action" means "returns the
        harmless hold-position action," not "keeps running whatever policy
        was active when it tripped." estop()/a NaN/a fall are read this way
        on purpose: freezing entirely (returning None) would leave real
        motors holding their last commanded torque, which is not obviously
        safer than a controlled, zero-target hold."""
        if self.paused:
            return None

        state = self.adapter.get_state()
        state = self.safety.tick(state)

        if not self.safety.tripped and self.selector is not None:
            proposed = self.selector.propose(state)
            if proposed is not None and proposed != self.supervisor.active_name:
                self.supervisor.request_switch(proposed)
                # Autonomous proposals still go through the SAME safety gate
                # as a human's request — being self-driven doesn't grant a
                # shortcut. safety.tick() above already tried to confirm any
                # pending switch this tick if it judged the moment safe.

        action = self.supervisor.step(obs)
        self.adapter.record(obs, action, state)
        return action
