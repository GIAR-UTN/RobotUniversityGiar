"""
RobotAdapter — the one boundary between "how do I talk to this specific robot
(simulated or real)" and everything else in legged_gym/control/ (supervisor,
safety governor, selector). Nothing above this layer should ever import
Genesis, MuJoCo, or unitree_sdk2 directly.

Two implementations exist:
  - SimAdapter (this file)   — wraps a legged_gym env (Genesis or MuJoCo)
  - RealAdapter (deploy/real_adapter.py) — wraps deploy_real.py's DDS
    LowCmd/LowState channel; lives in a separate package on purpose so this
    module (and everything that imports it) stays installable/importable on
    a machine with no unitree_sdk2 and no real robot attached.
"""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional, Protocol

import torch


class Lifecycle(Enum):
    """Mirrors ros2_control's controller lifecycle naming on purpose — it's a
    widely-recognized vocabulary, and it leaves the door open to an eventual
    ros2_control bridge without renaming anything here."""
    INACTIVE = "inactive"   # not built / not connected yet
    READY = "ready"         # built/connected, holding a safe default pose, not stepping a policy
    ACTIVE = "active"       # a policy is driving the robot
    FAULT = "fault"         # SafetyGovernor tripped — see safety.py; adapter should be holding position


@dataclasses.dataclass
class RobotState:
    """Canonical snapshot of robot state, backend-agnostic. Every adapter
    produces this same shape regardless of whether the numbers came from a
    Genesis tensor or a real LowState DDS message."""
    dof_pos: torch.Tensor          # (num_envs, num_dof)
    dof_vel: torch.Tensor          # (num_envs, num_dof)
    default_dof_pos: torch.Tensor  # (num_envs, num_dof) — the PD "home" pose
    base_quat: torch.Tensor        # (num_envs, 4) — w,x,y,z or x,y,z,w per backend's own convention
    base_ang_vel: torch.Tensor     # (num_envs, 3)
    base_lin_vel: Optional[torch.Tensor]  # (num_envs, 3) — None on real hardware (not directly sensed)
    projected_gravity: torch.Tensor  # (num_envs, 3) — gravity vector in the base frame; the same
                                      # upright/fallen signal legged_robot.py uses for episode
                                      # termination (projected_gravity[:,2] > threshold = fallen)
    commands: torch.Tensor         # (num_envs, 3) — requested lin_x, lin_y, ang_yaw
    action_scale: float
    lifecycle: Lifecycle


class RobotAdapter(Protocol):
    """What PolicySupervisor/SafetyGovernor/Selector are allowed to depend on.
    No policy-specific, no Genesis-specific, no DDS-specific details leak
    past this interface."""

    num_envs: int

    def reset(self) -> RobotState:
        """Sim: teleports to the default pose instantly. Real: NOT a hard
        reset — see RealAdapter, which raises or degrades this to "wait for
        the operator to move through the physical gating sequence"."""
        ...

    def get_state(self) -> RobotState:
        ...

    def send_action(self, action: torch.Tensor) -> RobotState:
        """Applies one policy-tick's action (already scaled/offset by the
        caller — see PolicySupervisor) and returns the resulting state."""
        ...

    def record(self, obs: torch.Tensor, action: torch.Tensor, state: RobotState) -> None:
        """Optional logging hook, no-op by default. Exists so a fork can wire
        up dataset recording (e.g. LeRobot-style episode capture) without
        threading a logger through every call site."""
        ...


class SimAdapter:
    """RobotAdapter over a legged_gym env (works with either the Genesis or
    MuJoCo backend this repo supports — it only touches env.simulator.*
    tensors and env.step()/env.reset(), which both backends populate the
    same way)."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self._lifecycle = Lifecycle.READY

    def reset(self) -> RobotState:
        self.env.reset()
        self._lifecycle = Lifecycle.READY
        return self.get_state()

    def get_state(self) -> RobotState:
        sim = self.env.simulator
        return RobotState(
            dof_pos=sim.dof_pos,
            dof_vel=sim.dof_vel,
            default_dof_pos=sim.default_dof_pos,
            base_quat=sim.base_quat,
            base_ang_vel=sim.base_ang_vel,
            base_lin_vel=sim.base_lin_vel,
            projected_gravity=sim.projected_gravity,
            commands=self.env.commands[:, :3],
            action_scale=self.env.cfg.control.action_scale,
            lifecycle=self._lifecycle,
        )

    def send_action(self, action: torch.Tensor) -> RobotState:
        self._lifecycle = Lifecycle.ACTIVE
        obs_buf, _, rews, dones, infos = self.env.step(action.detach())
        self._last_obs = obs_buf
        return self.get_state()

    def get_observations(self) -> torch.Tensor:
        """Not part of RobotAdapter — legged_gym's own observation vector is
        env-specific (differs per robot/task), so callers that need it ask
        the SimAdapter directly rather than through the generic protocol."""
        return self.env.get_observations()

    def record(self, obs: torch.Tensor, action: torch.Tensor, state: RobotState) -> None:
        pass

    def fault(self) -> None:
        self._lifecycle = Lifecycle.FAULT

    def estop(self) -> None:
        """Sim has no motors to cut power to — the meaningful thing an
        e-stop can do here is flag FAULT so SafetyGovernor/ControlService
        force the damping skill. RealAdapter.estop() does the real,
        immediate zero-torque write this name implies on actual hardware."""
        self.fault()
