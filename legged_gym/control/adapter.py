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
import math
from enum import Enum
from typing import Optional, Protocol

import torch

from legged_gym.utils.math_utils import quat_rotate_inverse

# See SimAdapter.set_operator_speed_limit's docstring — bounded, not
# unbounded, so a typo (e.g. an extra zero) can't ask for a wildly
# extreme out-of-distribution command; 3x the trained envelope is
# generous enough for genuine "what happens past the edge" experimentation
# in sim without that risk.
OPERATOR_SPEED_LIMIT_MAX = 3.0


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

    def __init__(self, env, operator_speed_limit: float = 1.0):
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

        # See set_operator_speed_limit()'s docstring for the full "why".
        # 1.0 (no extra cap beyond the trained envelope itself) matches
        # cfg.commands.ranges as trained — the same [-1, 1] m/s / [-1, 1]
        # rad/s default this repo shares with upstream unitree_rl_gym.
        self._operator_speed_limit = 1.0
        self.set_operator_speed_limit(operator_speed_limit)

        # The live control web's Stress Stimuli panel defaults both toggles
        # to unchecked — an operator opening the viewer should see the robot
        # hold still and driveable, not immediately get shoved/re-commanded
        # by the same domain-randomization stressors used during training.
        # This only overrides SimAdapter's own initial state (the task's own
        # cfg.domain_rand.push_robots, read by training's web_train.py
        # directly, is untouched) — see set_random_events for the same
        # toggle wired to the checkboxes.
        self.set_random_events(push_robots=False, auto_commands=False)

        # Disabled by default: legged_robot.py's own check_termination() (see
        # set_episode_timeout's docstring) is training-episode machinery that
        # would otherwise silently teleport the robot on a fixed timer, with
        # no signal to ControlService/the web UI — indistinguishable from an
        # intentional restart. Off here means only an explicit restart() or a
        # real fall/contact-force trip ever resets the robot; the web UI's
        # Pause & Restart panel can re-enable it at a chosen interval.
        self._episode_timeout_s: Optional[float] = None
        self.set_episode_timeout(None)

    def set_episode_timeout(self, seconds: Optional[float]) -> None:
        """Configures/disables legged_robot.py's own timer-based episode
        reset (env.step() -> check_termination() -> reset_idx(), driven by
        cfg.env.episode_length_s at env-construction time). That machinery
        makes sense for RL training rollouts but, left on for a live control
        session, silently resets the robot on a schedule that has nothing to
        do with anything the operator did.

        None disables it (the timeout never fires — env.max_episode_length
        is set to infinity). A positive number of seconds re-enables it at
        that interval. Either way this rewrites env.max_episode_length(_s)
        directly and zeroes the current episode's tick counter, so the new
        setting is exactly what's in effect starting next tick — not
        whatever was left on the previous window.

        Fall/contact-force termination (fail_buf, a separate condition ORed
        into the same reset_buf in check_termination) is untouched here —
        that's a real safety trip, not a timer, and this method never
        relaxes it."""
        if seconds is not None and seconds <= 0:
            raise ValueError("episode timeout must be a positive number of seconds, or None to disable")
        self._episode_timeout_s = seconds
        if seconds is None:
            self.env.max_episode_length_s = float("inf")
            self.env.max_episode_length = float("inf")
        else:
            self.env.max_episode_length_s = seconds
            self.env.max_episode_length = math.ceil(seconds / self.env.dt)
        self.env.episode_length_buf[:] = 0

    @property
    def episode_timeout_s(self) -> Optional[float]:
        return self._episode_timeout_s

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

    def get_camera_frame(self):
        """Not part of RobotAdapter (see get_observations' docstring for
        why — this is an optional, backend-specific extra, not something
        PolicySupervisor/SafetyGovernor/Selector ever touch). Returns a
        (H, W, 3) uint8 numpy array from the simulator's RGB camera (see
        GenesisSimulator.get_camera_frame(), gated by
        cfg.sensor.add_rgb_camera), or None if the current backend has no
        camera support or it wasn't enabled — callers (rugiar_driver.py's
        control loop) must treat None as "nothing to stream this tick", not
        an error."""
        get_frame = getattr(self.env.simulator, "get_camera_frame", None)
        return get_frame() if get_frame is not None else None

    def get_target_relative_pos(self) -> Optional[torch.Tensor]:
        """Not part of RobotAdapter (same reasoning as get_camera_frame() above)
        -- an optional, backend-specific extra. Returns the `--ball` prop's live
        world position transformed into the robot's own base frame (x forward,
        y left, z up), shape (num_envs, 3), or None if no ball prop exists (
        `--ball` wasn't passed) or this backend has no prop tracking at all.
        This is ground truth, not detection -- the real-camera equivalent
        (RealAdapter) has no prop to read and has no get_target_relative_pos
        at all; callers must treat a missing/absent method or a None return
        the same way get_camera_frame()'s callers already do: 'nothing to
        feed this tick', not an error."""
        props = getattr(self.env.simulator, "_props", None)
        if not props or "ball" not in props:
            return None
        ball_pos_world = props["ball"]["pos"]
        offset_world = ball_pos_world - self.env.simulator.base_pos
        return quat_rotate_inverse(self.env.simulator.base_quat, offset_world)

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
        used during training (cfg.commands.ranges) scaled by
        operator_speed_limit (see set_operator_speed_limit — 1.0 by
        default, i.e. no extra cap beyond the trained envelope itself) — a
        manual command is never allowed to ask the policy for something
        outside the envelope it was actually trained across, or past
        whatever narrower limit this session has chosen to operate under.

        Also switches off cfg.commands.heading_command for as long as a
        manual command is active: G1's default config computes yaw-RATE
        from a heading TARGET every tick (see _post_physics_step_callback in
        legged_robot.py) — with that on, a direct yaw-rate command would be
        silently overwritten within the very same tick it was set."""
        self._manual_command = self._clamp_command(vx, vy, yaw)
        self._auto_commands = False
        self.env.cfg.commands.heading_command = False
        self._apply_manual_command()

    def _clamp_command(self, vx: float, vy: float, yaw: float) -> tuple:
        """Shared by set_command() and set_operator_speed_limit() so the
        trained-range-times-limit math exists in exactly one place."""
        ranges = self.env.cfg.commands.ranges
        limit = self._operator_speed_limit
        vx = max(min(vx, ranges.lin_vel_x[1] * limit), ranges.lin_vel_x[0] * limit)
        vy = max(min(vy, ranges.lin_vel_y[1] * limit), ranges.lin_vel_y[0] * limit)
        yaw = max(min(yaw, ranges.ang_vel_yaw[1] * limit), ranges.ang_vel_yaw[0] * limit)
        return (vx, vy, yaw)

    def set_operator_speed_limit(self, fraction: float) -> None:
        """Scales set_command()'s own clamp to `fraction` of the trained
        cfg.commands.ranges envelope (e.g. 0.5 caps a [-1, 1] m/s trained
        range to [-0.5, 0.5] m/s) — 1.0 means no extra cap at all, just the
        trained envelope itself, which is this repo's (and upstream
        unitree_rl_gym's) own community-standard default.

        Values above 1.0 (up to OPERATOR_SPEED_LIMIT_MAX) are allowed ON
        PURPOSE — commanding a policy PAST what it was ever trained
        across, deliberately out-of-distribution. In sim this is a safe,
        genuinely useful thing to be able to try (worst case: it falls
        over in Genesis, nothing is at risk) — the web UI surfaces this as
        an explicit "beyond the trained range" warning rather than hiding
        the option. This is NOT extended to RealAdapter, which has no
        set_operator_speed_limit of its own (see
        ControlService.set_operator_speed_limit's NotImplementedError path)
        — asking a real, physical robot for a velocity it was never
        trained to produce is a different risk category entirely and
        deliberately requires a separate, more deliberate decision later,
        not a side effect of this sim-side experiment knob.

        This lives HERE, server-side, on purpose — not as a per-client
        scaling constant — so every client that ever calls set_command
        (the web UI, examples/joystick_controller.py, a future CLI driving
        a different robot over its own token-gated connection) shares the
        exact same enforced limit for free, set in exactly one place,
        instead of each one duplicating (and inevitably drifting out of
        sync with) its own local cap. A client is always free to just ask
        for the full trained range (vx=ranges.lin_vel_x[1], etc.) and let
        this be the one thing standing between that ask and what actually
        reaches the robot.

        Re-clamps whatever manual command is already in effect immediately
        — so lowering this while already cruising at speed takes effect on
        the very next tick, not just on the next set_command call — but
        deliberately does NOT flip into manual mode or touch
        heading_command by itself: adjusting the limit is not the same
        gesture as issuing a command, and must not silently hijack control
        away from auto mode."""
        if not (0.0 < fraction <= OPERATOR_SPEED_LIMIT_MAX):
            raise ValueError(
                f"operator_speed_limit must be in (0.0, {OPERATOR_SPEED_LIMIT_MAX}], got {fraction}")
        self._operator_speed_limit = fraction
        self._manual_command = self._clamp_command(*self._manual_command)
        if not self._auto_commands:
            self._apply_manual_command()

    @property
    def operator_speed_limit(self) -> float:
        return self._operator_speed_limit

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
