"""
Backend-agnostic control layer for switching between locomotion policies —
in simulation today, on real Unitree hardware later — without rewriting the
switching/safety/UI logic per backend.

See the repo root README.md (sections 3-5) for the full write-up of why this
package exists and how the pieces fit together.

Quick map:
  adapter.py    RobotAdapter protocol + Lifecycle/RobotState + SimAdapter
  supervisor.py PolicySupervisor — owns loaded policies, does the cross-fade swap
  safety.py     SafetyGovernor — the only thing allowed to say "yes, switch now"
  selector.py   Selector — proposes switches (rule-based today, learned later)
  service.py    ControlService — one call surface for a human UI or an
                autonomous process to drive the supervisor identically
"""
from .adapter import Lifecycle, RobotState, RobotAdapter, SimAdapter
from .policy import Policy, ExplicitStatePolicy, InternalStatePolicy, load_policy, damping_policy
from .supervisor import PolicySupervisor
from .safety import SafetyGovernor
from .selector import Selector, TiltRecoverySelector
from .service import ControlService

__all__ = [
    "Lifecycle", "RobotState", "RobotAdapter", "SimAdapter",
    "Policy", "ExplicitStatePolicy", "InternalStatePolicy", "load_policy", "damping_policy",
    "PolicySupervisor",
    "SafetyGovernor",
    "Selector", "TiltRecoverySelector",
    "ControlService",
]
