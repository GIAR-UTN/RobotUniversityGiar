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
    base_height: Optional[torch.Tensor]  # (num_envs,) — world-frame base z, i.e. what
                                          # rewards.base_height_target tracks (legged_robot.py's
                                          # _reward_base_height). This is SIMULATOR GROUND TRUTH, not
                                          # a sensor reading — no IMU or other real sensor measures
                                          # height directly, so this is None on real hardware (mirrors
                                          # base_lin_vel's same real-hardware caveat above). Still a
                                          # legitimate training-time target: training only ever runs
                                          # in sim, so "ground truth exists" is all that's required
                                          # there — the caveat only matters for what real-robot
                                          # *inference* could ever condition on.
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

    # Read by ControlService.status() (see service.py) so a UI can show
    # which backend is driving the robot and gray out controls the current
    # backend doesn't support — e.g. RealAdapter has no instant "restart".
    backend_name = "sim"
    capabilities = {"restart": True}

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self._lifecycle = Lifecycle.READY

        # Manual velocity-command override (see set_command/set_random_events
        # below). Snapshot the original heading_command setting so it can be
        # restored when returning to "auto" mode — see set_command's docstring
        # for why heading_command has to be off while a manual command is active.
        self._orig_heading_command = env.cfg.commands.heading_command
        self._auto_commands = True
        self._manual_command = (0.0, 0.0, 0.0)

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
            base_height=sim.base_pos[:, 2],
            commands=self.env.commands[:, :3],
            action_scale=self.env.cfg.control.action_scale,
            lifecycle=self._lifecycle,
        )

    def send_action(self, action: torch.Tensor) -> RobotState:
        if not self._auto_commands:
            # Re-assert every tick, right before stepping — the same pattern
            # play.py's own joystick/slider control already uses. Necessary
            # because _post_physics_step_callback (legged_robot.py) resamples
            # commands on its own schedule regardless of who's driving; a
            # manual command has to keep winning, tick after tick, or it gets
            # silently overwritten a few seconds later.
            self._apply_manual_command()
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

    def _apply_manual_command(self) -> None:
        vx, vy, yaw = self._manual_command
        self.env.commands[:, 0] = vx
        self.env.commands[:, 1] = vy
        self.env.commands[:, 2] = yaw

    def set_command(self, vx: float, vy: float, yaw: float) -> None:
        """Directly commands a target walking velocity, overriding whatever
        the environment's own domain-randomization command resampling would
        otherwise pick (see set_random_events). Clamped to the exact ranges
        used during training (cfg.commands.ranges) — a manual command is
        never allowed to ask the policy for something outside the envelope
        it was actually trained across.

        Also switches off cfg.commands.heading_command for as long as a
        manual command is active: G1's default config computes yaw-RATE
        from a heading TARGET every tick (see _post_physics_step_callback in
        legged_robot.py) — with that on, a direct yaw-rate command would be
        silently overwritten within the very same tick it was set."""
        ranges = self.env.cfg.commands.ranges
        vx = max(min(vx, ranges.lin_vel_x[1]), ranges.lin_vel_x[0])
        vy = max(min(vy, ranges.lin_vel_y[1]), ranges.lin_vel_y[0])
        yaw = max(min(yaw, ranges.ang_vel_yaw[1]), ranges.ang_vel_yaw[0])
        self._manual_command = (vx, vy, yaw)
        self._auto_commands = False
        self.env.cfg.commands.heading_command = False
        self._apply_manual_command()

    def set_random_events(self, push_robots: bool, auto_commands: bool,
                           push_dir: Optional[str] = None) -> None:
        """Independently toggles the two domain-randomization stimuli that
        otherwise run unconditionally every tick, in the sim demo just like
        in training (legged_robot.py's _post_physics_step_callback) — random
        shoves, and the velocity command changing on its own every few
        seconds. Turning both off is what lets you drive the robot
        deliberately, the way an operator would, instead of watching it
        react to the same randomized stressors used during training.
        push_dir (None/'behind'/'front'/'left'/'right') biases the shove
        direction the same way training's --push_dir does — read live by
        Simulator.sample_push_vel_xy() on every push, so this takes effect
        on the very next one."""
        self.env.cfg.domain_rand.push_robots = push_robots
        self.env.cfg.domain_rand.push_dir = push_dir
        if auto_commands:
            self._auto_commands = True
            self.env.cfg.commands.heading_command = self._orig_heading_command
        else:
            self._auto_commands = False
            self.env.cfg.commands.heading_command = False
            self._apply_manual_command()  # hold at the last manual value (0,0,0 if never set)

    @property
    def command(self) -> tuple:
        """Current (vx, vy, yaw) — whether it got there via set_command or
        the environment's own auto-resampling."""
        return tuple(float(v) for v in self.env.commands[0, :3].tolist())

    @property
    def random_events(self) -> dict:
        return {
            "push_robots": bool(self.env.cfg.domain_rand.push_robots),
            "auto_commands": self._auto_commands,
            "push_dir": getattr(self.env.cfg.domain_rand, "push_dir", None),
        }

    def estop(self) -> None:
        """Sim has no motors to cut power to — the meaningful thing an
        e-stop can do here is flag FAULT so SafetyGovernor/ControlService
        force the damping skill. RealAdapter.estop() does the real,
        immediate zero-torque write this name implies on actual hardware."""
        self.fault()
