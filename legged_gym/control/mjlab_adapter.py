"""
MjlabAdapter — RobotAdapter over an mjlab (MuJoCo Warp) ManagerBasedRlEnv,
the third backend behind this repo's control engine after SimAdapter
(Genesis/Isaac, adapter.py) and RealAdapter (DDS, deploy_real/).

Phase 5 of docs/mjlab_migration.md. Nothing above this file changes for
mjlab: PolicySupervisor/SafetyGovernor/ControlService/ControlServer are
backend-agnostic by construction (see adapter.py's module docstring), so
"support mjlab" is exactly this file plus a driver script that wires it up
(legged_gym/scripts/rugiar_driver_mjlab.py).

Deliberately NOT implemented here: set_command / set_operator_speed_limit /
set_random_events / set_episode_timeout. Rugiar-G1-Mimic is a *motion
tracking* task — there is no velocity command to issue: the command IS the
reference motion clip loaded at env construction (one clip per policy, see
docs/mjlab_migration.md R8), and mjlab's tracking task has no
push_robots/heading_command/commands.ranges knobs to toggle. Following the
precedent RealAdapter already set for set_operator_speed_limit, these are
simply absent rather than stubbed: ControlService looks them up with
getattr() and raises NotImplementedError when they're missing (see
ControlService.set_command/set_operator_speed_limit), and status() omits
the `command`/`random_events`/`operator_speed_limit`/`episode_timeout_s`
keys entirely, which is what web/app.js reads to gray out the controls
that don't apply to this backend. A no-op stub would silently swallow an
operator's input instead.

This module imports mjlab lazily-by-duck-typing: it never imports mjlab at
all. It only reads `env.scene["robot"].data.*` and calls env.step/reset, so
it stays importable (and unit-testable) from the main .venv, which has no
mjlab installed.
"""
from __future__ import annotations

from typing import Optional

import torch

from .adapter import Lifecycle, RobotState


class MjlabAdapter:
    """RobotAdapter over an mjlab ManagerBasedRlEnv.

    Read by ControlService.status() and surfaced to the web UI — same
    contract as SimAdapter/RealAdapter's own class attributes. `restart`
    is genuinely supported here (env.reset() teleports back to the start
    of the reference motion, exactly like Genesis's), so the Pause &
    Restart panel works unchanged.
    """

    backend_name = "mjlab"
    # "motion": True gates the web UI's Motion panel — see
    # ControlService.status()'s docstring on capabilities being
    # adapter-declared, not UI-hardcoded, and web/app.js's applyStatus(),
    # which follows the same "absent key means unsupported" convention the
    # Command/Stimuli panels already use for 'command'/'random_events' in
    # status() itself. SimAdapter's capabilities has no 'motion' key at all
    # — Genesis has no reference-motion command term to switch.
    capabilities = {"restart": True, "motion": True}

    #: Which observation group feeds the policy. mjlab's tracking env
    #: returns a dict ({'actor': [N,154], 'critic': [N,286]}); everything
    #: above this layer expects a single tensor, so this is where the
    #: 154-dim actor group (docs/mjlab_migration.md §2 — the exact vector
    #: Javier's ONNX checkpoints were trained against) gets picked out.
    OBS_GROUP = "actor"

    def __init__(self, env, robot_name: str = "robot"):
        self.env = env
        self.num_envs = env.num_envs
        self._robot_name = robot_name
        self._lifecycle = Lifecycle.READY
        self._obs: Optional[torch.Tensor] = None
        obs, _ = env.reset()
        self._obs = obs[self.OBS_GROUP]

    # ---- RobotAdapter protocol ----

    def reset(self) -> RobotState:
        obs, _ = self.env.reset()
        self._obs = obs[self.OBS_GROUP]
        self._lifecycle = Lifecycle.READY
        return self.get_state()

    def get_state(self) -> RobotState:
        data = self.env.scene[self._robot_name].data
        return RobotState(
            dof_pos=data.joint_pos,
            dof_vel=data.joint_vel,
            default_dof_pos=data.default_joint_pos,
            # mjlab/MuJoCo quaternions are wxyz (same convention as this
            # repo's mjlab motion .npz files, docs/mjlab_migration.md §3).
            # RobotState documents base_quat as "per backend's own
            # convention" — SafetyGovernor only NaN-checks it, and
            # _telemetry() doesn't surface it, so no conversion is needed
            # or wanted here.
            base_quat=data.root_link_quat_w,
            base_ang_vel=data.root_link_ang_vel_b,
            base_lin_vel=data.root_link_lin_vel_b,
            # Already gravity-in-body-frame, normalized the same way
            # legged_robot.py's own projected_gravity is: upright is
            # ~(0,0,-1), so SafetyGovernor's max_projected_gravity_z fall
            # threshold applies unchanged.
            projected_gravity=data.projected_gravity_b,
            base_height=data.root_link_pos_w[:, 2],
            base_pos_xy=data.root_link_pos_w[:, :2],
            # No velocity command exists on a tracking task (see this
            # module's docstring) — a zero triple keeps RobotState's shape
            # contract for any consumer that reads it positionally, and
            # ControlService only surfaces status()['command'] when the
            # ADAPTER has a `command` property, which this one does not.
            commands=torch.zeros(self.num_envs, 3, device=data.joint_pos.device),
            action_scale=self._action_scale(),
            lifecycle=self._lifecycle,
        )

    def send_action(self, action: torch.Tensor) -> RobotState:
        self._lifecycle = Lifecycle.ACTIVE
        obs, _, _, _, _ = self.env.step(action.detach())
        self._obs = obs[self.OBS_GROUP]
        return self.get_state()

    def record(self, obs: torch.Tensor, action: torch.Tensor, state: RobotState) -> None:
        pass

    # ---- backend-specific extras (not part of RobotAdapter) ----

    def get_observations(self) -> torch.Tensor:
        """Same "not part of the protocol, ask the adapter directly" shape
        as SimAdapter.get_observations() — see its docstring. Returns the
        actor group only (see OBS_GROUP)."""
        if self._obs is None:
            self._obs = self.env.get_observations()[self.OBS_GROUP]
        return self._obs

    def get_camera_frame(self):
        """No robot-POV camera is built for the mjlab tracking task — same
        contract as SimAdapter.get_camera_frame() on a task without one:
        None means "nothing to stream this tick", not an error."""
        return None

    def get_depth_frame(self):
        """No robot-POV depth camera is built for the mjlab tracking task — same
        contract as get_camera_frame() above."""
        return None

    def _action_scale(self) -> float:
        """mjlab's G1 action scale is PER-JOINT-GROUP (a dict in
        JointPositionActionCfg.scale, ~0.075 to 0.548 — see
        docs/mjlab_migration.md §6), not the single flat float this repo's
        Genesis configs use, so `ActionTerm.scale` is a
        (num_envs, num_actions) tensor here. RobotState.action_scale is a
        single float by contract, and its only consumers are display/
        telemetry-adjacent, so this reports the mean. It is NOT used to
        compute any target: mjlab's own ActionTerm applies the real
        per-joint tensor inside env.step()."""
        term = self.env.action_manager.get_term(self.env.action_manager.active_terms[0])
        scale = term.scale
        if isinstance(scale, torch.Tensor):
            return float(scale.mean())
        return float(scale)

    def before_step(self) -> None:
        """Hook for a caller that steps the env ITSELF instead of going
        through send_action() (e.g. rugiar_driver_mjlab.py's
        ViserPlayViewer, which owns env.step()). It must be called once per
        tick, before that step. A no-op here; MjlabVelocityAdapter pins its
        manual command in it."""

    def fault(self) -> None:
        self._lifecycle = Lifecycle.FAULT

    def estop(self) -> None:
        """Same as SimAdapter.estop(): there are no motors to cut power to
        in simulation, so the meaningful action is flagging FAULT and
        letting SafetyGovernor force the damping skill."""
        self.fault()


class MjlabVelocityAdapter(MjlabAdapter):
    """MjlabAdapter for an mjlab VELOCITY task (e.g. Rugiar-G1Edu-Velocity):
    the same env plumbing, plus the manual velocity command that a tracking
    task doesn't have.

    This is a separate class, not a flag on MjlabAdapter, because
    ControlService detects command support with hasattr(adapter, "command")
    (see ControlService.status()). A property that exists but raises would
    still pass that check, and the web UI would show a Command panel that
    can't work.

    How the manual command works: mjlab's UniformVelocityCommand resamples
    `vel_command_b` whenever its `time_left` runs out. It also rewrites the
    command for heading, standing and world-frame envs in
    _update_command(). Before every step this adapter pins the command:
    time_left is pushed far into the future, those three per-env flags are
    cleared, and vel_command_b is written. Only then does env.step() run.
    The observation that step returns therefore already carries the
    operator's command, with no one-tick lag and no random resample in
    between.

    Clamping follows SimAdapter.set_command: the trained `ranges` times
    operator_speed_limit (OPERATOR_SPEED_LIMIT_MAX bounds it).
    """

    capabilities = {"restart": True}
    COMMAND_TERM = "twist"

    def __init__(self, env, robot_name: str = "robot", operator_speed_limit: float = 1.0):
        super().__init__(env, robot_name)
        self._manual_command = (0.0, 0.0, 0.0)
        self._operator_speed_limit = 1.0
        self.set_operator_speed_limit(operator_speed_limit)
        self._pin_command()

    def _term(self):
        return self.env.command_manager.get_term(self.COMMAND_TERM)

    def _pin_command(self) -> None:
        term = self._term()
        term.time_left[:] = 1.0e9
        for flag in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env"):
            buf = getattr(term, flag, None)
            if buf is not None:
                buf[:] = False
        vx, vy, yaw = self._manual_command
        term.vel_command_b[:, 0] = vx
        term.vel_command_b[:, 1] = vy
        term.vel_command_b[:, 2] = yaw

    def _clamp_command(self, vx: float, vy: float, yaw: float) -> tuple:
        ranges = self._term().cfg.ranges
        limit = self._operator_speed_limit
        vx = max(min(vx, ranges.lin_vel_x[1] * limit), ranges.lin_vel_x[0] * limit)
        vy = max(min(vy, ranges.lin_vel_y[1] * limit), ranges.lin_vel_y[0] * limit)
        yaw = max(min(yaw, ranges.ang_vel_z[1] * limit), ranges.ang_vel_z[0] * limit)
        return (float(vx), float(vy), float(yaw))

    # ---- command surface (same names/semantics as SimAdapter) ----

    def set_command(self, vx: float, vy: float, yaw: float) -> None:
        self._manual_command = self._clamp_command(vx, vy, yaw)
        self._pin_command()

    def set_operator_speed_limit(self, fraction: float) -> None:
        from .adapter import OPERATOR_SPEED_LIMIT_MAX
        if not (0.0 < fraction <= OPERATOR_SPEED_LIMIT_MAX):
            raise ValueError(f"operator_speed_limit must be in (0, {OPERATOR_SPEED_LIMIT_MAX}], got {fraction}")
        self._operator_speed_limit = float(fraction)
        self._manual_command = self._clamp_command(*self._manual_command)

    @property
    def operator_speed_limit(self) -> float:
        return self._operator_speed_limit

    @property
    def command(self) -> tuple:
        return self._manual_command

    # ---- RobotAdapter overrides: keep the command pinned ----

    def reset(self):
        state = super().reset()
        self._pin_command()
        return state

    def before_step(self) -> None:
        self._pin_command()

    def send_action(self, action: torch.Tensor):
        self._pin_command()
        return super().send_action(action)

    def get_state(self):
        state = super().get_state()
        vx, vy, yaw = self._manual_command
        state.commands[:, 0] = vx
        state.commands[:, 1] = vy
        state.commands[:, 2] = yaw
        return state


def adapter_for_env(env, robot_name: str = "robot") -> MjlabAdapter:
    """Picks the adapter class from the env's own command terms: a `twist`
    command means a velocity task (MjlabVelocityAdapter), anything else
    (e.g. the tracking task's `motion`) gets the plain MjlabAdapter."""
    terms = getattr(env.command_manager, "active_terms", None) or []
    if MjlabVelocityAdapter.COMMAND_TERM in terms:
        return MjlabVelocityAdapter(env, robot_name)
    return MjlabAdapter(env, robot_name)
