"""
RealAdapter — RobotAdapter over a real Unitree robot, via unitree_sdk2py's DDS
channels (rt/lowcmd out, rt/lowstate in), following the exact structure of
unitree_rl_gym's own deploy/deploy_real/deploy_real.py.

⚠️  UNTESTED — written and reviewed against unitree_rl_gym's real deploy code
and Unitree's SDK2 docs, but this repo was built entirely on a Mac with no
unitree_sdk2py installed and no physical robot attached. Treat this as a
careful port to review/validate against real hardware, not as proven code.
Before trusting it on an actual robot: re-verify motor index mapping, the
IMU frame transform for torso-mounted-IMU robots (H1/H1-2), and every
threshold in legged_gym/control/safety.py against your specific robot.
See docs/index.html §13 for the first-boot checklist (zero torque ->
move to default -> hold, each stage gated by the remote control) before
handing control to a policy or a networked client.

Design notes (why this looks the way it does):
  - unitree_sdk2py is imported lazily (inside __init__), not at module level,
    so `import deploy_real.real_adapter` doesn't fail on a machine without
    the SDK — you can still read/typecheck this file anywhere.
  - reset() is NOT instantaneous like SimAdapter's. A real robot cannot be
    teleported to a safe pose; it must be walked through the same
    zero_torque -> move_to_default -> hold physical gating sequence
    deploy_real.py already uses, each stage gated by an operator's remote
    control button. reset() here blocks until that sequence completes.
  - The policy's action only ever touches the LEG joints
    (config.leg_joint2motor_idx). Arm/waist joints are held at a fixed
    target with their own (usually stiffer) gains — the policies trained in
    this repo were never taught to control them. This is unrelated to
    policy switching; it's true for whichever leg policy is active.
  - get_observations()'s layout mirrors G1Robot.compute_observations()
    (legged_gym/envs/g1/g1.py) EXACTLY — same field order, same scales
    (config.commands_scale, not unitree_rl_gym's own cmd_scale/max_cmd,
    see deploy_real/configs/g1.yaml's header comment). A mismatch here
    silently feeds the policy an out-of-distribution observation instead
    of erroring, so this is the single most safety-relevant piece of this
    file to re-verify against whatever config you actually load.
  - set_command()/command mirror SimAdapter's manual-command interface
    (legged_gym/control/adapter.py) so the same ControlService.set_command
    RPC — and therefore the same web UI / any home-made joystick client
    speaking the control protocol — works unmodified against a real robot.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from legged_gym.control.adapter import Lifecycle, RobotState


def get_gravity_orientation(quaternion):
    """Ported verbatim from unitree_rl_gym's deploy/deploy_real/common/rotation_helper.py.
    quaternion = (w, x, y, z)."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    """Ported verbatim from unitree_rl_gym — needed for H1/H1-2, whose IMU is
    mounted on the torso rather than the pelvis the policy was trained
    against. G1's IMU is already in the pelvis frame; skip this for G1."""
    from scipy.spatial.transform import Rotation as R
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


class RealAdapter:
    """See module docstring. Construct with a deploy_real/configs/*.yaml-loaded
    RobotConfig (see deploy_real/config.py) and a network interface name
    (e.g. 'enp3s0')."""

    num_envs = 1  # a real robot is always exactly one instance

    # See SimAdapter's matching attributes in legged_gym/control/adapter.py —
    # read by ControlService.status() so a UI can gray out "restart" (a real
    # robot has no instant reset; it needs the physical button-gated startup
    # sequence — see reset()/the _zero_torque_state..._hold_default_pos chain
    # below).
    backend_name = "real"
    capabilities = {"restart": False}

    def __init__(self, config, net_interface: str):
        # Imported here, not at module level — see module docstring.
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
        from unitree_sdk2py.utils.crc import CRC
        from deploy_real.common.remote_controller import RemoteController, KeyMap

        if config.msg_type == "hg":
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd, LowState_ as LowState
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            low_state_default = unitree_hg_msg_dds__LowState_()
        elif config.msg_type == "go":
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd, LowState_ as LowState
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            low_state_default = unitree_go_msg_dds__LowState_()
        else:
            raise ValueError(f"Unsupported msg_type '{config.msg_type}' — expected 'hg' or 'go'")

        self.config = config
        self.low_state = low_state_default
        self._lifecycle = Lifecycle.INACTIVE
        self._crc = CRC()
        self._remote = RemoteController()
        self._KeyMap = KeyMap
        self._mode_machine = 0  # hg only — captured from the first LowState, see _on_low_state

        ChannelFactoryInitialize(0, net_interface)
        self._publisher = ChannelPublisher(config.lowcmd_topic, LowCmd)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber(config.lowstate_topic, LowState)
        self._subscriber.Init(self._on_low_state, 10)

        self.num_actions = config.num_actions
        self._counter = 0
        self._last_action = np.zeros(config.num_actions, dtype=np.float32)
        self._manual_command = (0.0, 0.0, 0.0)

        # Order matters: mode_machine (hg) is only known once the robot's
        # own first LowState arrives, so the low_cmd init happens AFTER the
        # wait — mirrors deploy_real.py's Controller.__init__ exactly.
        self._wait_for_low_state()
        self._init_low_cmd()

    def _on_low_state(self, msg) -> None:
        self.low_state = msg
        self._remote.set(msg.wireless_remote)
        if self.config.msg_type == "hg":
            self._mode_machine = msg.mode_machine

    def _init_low_cmd(self) -> None:
        """Ported from common/command_helper.py's init_cmd_hg/init_cmd_go."""
        if self.config.msg_type == "hg":
            self.low_cmd.mode_machine = self._mode_machine
            self.low_cmd.mode_pr = 0  # MotorMode.PR
            for mc in self.low_cmd.motor_cmd:
                mc.mode = 1
        else:  # "go"
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
            self.low_cmd.level_flag = 0xFF
            self.low_cmd.gpio = 0
            for mc in self.low_cmd.motor_cmd:
                mc.mode = 0x0A

    def _wait_for_low_state(self) -> None:
        """Blocks until the first LowState arrives over DDS — mirrors
        deploy_real.py's Controller.wait_for_low_state(). Without this,
        get_state()/reset() would read the zero-initialized low_state
        default instead of the robot's actual pose."""
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("[RealAdapter] Successfully connected to the robot.")

    # ---- RobotAdapter protocol ----

    def reset(self) -> RobotState:
        """Blocks through the physical gating sequence — this is where a
        human with the remote control (or an automated bench-test rig with
        its own button simulator) has to actually be present. Returns once
        the robot is standing at its default pose, lifecycle READY."""
        self._zero_torque_state()
        self._move_to_default_pos()
        self._hold_default_pos()
        self._last_action[:] = 0.0
        self._counter = 0
        self._lifecycle = Lifecycle.READY
        return self.get_state()

    def get_state(self) -> RobotState:
        cfg = self.config

        qj = np.array([self.low_state.motor_state[i].q for i in cfg.leg_joint2motor_idx], dtype=np.float32)
        dqj = np.array([self.low_state.motor_state[i].dq for i in cfg.leg_joint2motor_idx], dtype=np.float32)

        quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        if cfg.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[cfg.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[cfg.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw, waist_yaw_omega, quat, ang_vel[None, :])
            ang_vel = ang_vel.squeeze()

        gravity = get_gravity_orientation(quat)

        return RobotState(
            dof_pos=torch.from_numpy(qj).unsqueeze(0),
            dof_vel=torch.from_numpy(dqj).unsqueeze(0),
            default_dof_pos=torch.from_numpy(cfg.default_angles.astype(np.float32)).unsqueeze(0),
            base_quat=torch.from_numpy(quat).unsqueeze(0),
            base_ang_vel=torch.from_numpy(ang_vel.astype(np.float32)).unsqueeze(0),
            base_lin_vel=None,  # not directly sensed on real hardware
            projected_gravity=torch.from_numpy(gravity.astype(np.float32)).unsqueeze(0),
            base_height=None,  # no sensor measures this directly — see RobotState's own docstring
            commands=torch.tensor([self._manual_command], dtype=torch.float32),
            action_scale=cfg.action_scale,
            lifecycle=self._lifecycle,
        )

    def send_action(self, action: torch.Tensor) -> RobotState:
        cfg = self.config
        action_np = action.detach().cpu().numpy().squeeze()
        target_dof_pos = cfg.default_angles + action_np * cfg.action_scale

        for i, motor_idx in enumerate(cfg.leg_joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = float(cfg.kps[i])
            self.low_cmd.motor_cmd[motor_idx].kd = float(cfg.kds[i])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        for i, motor_idx in enumerate(cfg.arm_waist_joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = float(cfg.arm_waist_target[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = float(cfg.arm_waist_kps[i])
            self.low_cmd.motor_cmd[motor_idx].kd = float(cfg.arm_waist_kds[i])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        self._lifecycle = Lifecycle.ACTIVE
        self._send_cmd()
        self._last_action = action_np.astype(np.float32).copy()
        self._counter += 1
        time.sleep(self.config.control_dt)
        return self.get_state()

    def get_observations(self) -> torch.Tensor:
        """Builds the same 47-dim obs vector G1Robot.compute_observations()
        (legged_gym/envs/g1/g1.py) produces in sim: ang_vel, gravity,
        commands*commands_scale, (dof_pos-default)*dof_pos_scale,
        dof_vel*dof_vel_scale, previous action, sin_phase, cos_phase.
        Field order and scales MUST match compute_observations() exactly —
        see this module's docstring."""
        cfg = self.config
        state = self.get_state()

        ang_vel = state.base_ang_vel.squeeze(0).numpy() * cfg.ang_vel_scale
        gravity = state.projected_gravity.squeeze(0).numpy()
        commands = np.array(self._manual_command, dtype=np.float32) * cfg.commands_scale
        dof_pos = (state.dof_pos.squeeze(0).numpy() - cfg.default_angles) * cfg.dof_pos_scale
        dof_vel = state.dof_vel.squeeze(0).numpy() * cfg.dof_vel_scale

        period = 0.8
        count = self._counter * cfg.control_dt
        phase = (count % period) / period
        sin_phase = np.array([np.sin(2 * np.pi * phase)], dtype=np.float32)
        cos_phase = np.array([np.cos(2 * np.pi * phase)], dtype=np.float32)

        obs = np.concatenate([
            ang_vel, gravity, commands, dof_pos, dof_vel, self._last_action, sin_phase, cos_phase,
        ]).astype(np.float32)
        return torch.from_numpy(obs).unsqueeze(0)

    def record(self, obs: torch.Tensor, action: torch.Tensor, state: RobotState) -> None:
        pass

    def get_camera_frame(self):
        """Mirrors SimAdapter.get_camera_frame()'s optional extra — no real
        camera is wired up yet (this robot's onboard RGB-D sensor, e.g. the
        G1's D435 head camera, isn't read anywhere in this file). Returns
        None until that's added (a lazy-imported reader module under
        deploy_real/common/, the same pattern unitree_sdk2py itself is
        imported here — see this module's docstring)."""
        return None

    def estop(self) -> None:
        """True emergency stop — cuts to zero torque immediately. Deliberately
        NOT routed through PolicySupervisor/SafetyGovernor's damping-policy
        fallback (which still runs the PD controller at some Kp/Kd): this is
        the one action that must never depend on the policy-switching layer
        working correctly."""
        for motor_idx in list(self.config.leg_joint2motor_idx) + list(self.config.arm_waist_joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = 0.0
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = 0.0
            self.low_cmd.motor_cmd[motor_idx].kd = 0.0
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0
        self._send_cmd()
        self._lifecycle = Lifecycle.FAULT

    # ---- manual velocity command (mirrors SimAdapter.set_command) ----

    def set_command(self, vx: float, vy: float, yaw: float) -> None:
        """Directly commands a target walking velocity, clamped to the exact
        ranges used during training (config.command_ranges) — see
        deploy_real/configs/g1.yaml. Used by ControlService.set_command, so
        the web UI and any home-made controller speaking the control
        protocol (docs/index.html §13) can drive this the same way
        they drive SimAdapter."""
        ranges = self.config.command_ranges
        vx = max(min(vx, ranges["lin_vel_x"][1]), ranges["lin_vel_x"][0])
        vy = max(min(vy, ranges["lin_vel_y"][1]), ranges["lin_vel_y"][0])
        yaw = max(min(yaw, ranges["ang_vel_yaw"][1]), ranges["ang_vel_yaw"][0])
        self._manual_command = (vx, vy, yaw)

    @property
    def command(self) -> tuple:
        return self._manual_command

    # ---- physical gating sequence (ported from deploy_real.py's Controller) ----

    def _zero_torque_state(self) -> None:
        print("[RealAdapter] Zero torque state — waiting for the remote control START button...")
        while self._remote.button[self._KeyMap.start] != 1:
            self._create_zero_cmd()
            self._send_cmd()
            time.sleep(self.config.control_dt)

    def _move_to_default_pos(self) -> None:
        print("[RealAdapter] Moving to default pos.")
        cfg = self.config
        total_time = 2.0
        num_steps = int(total_time / cfg.control_dt)

        dof_idx = list(cfg.leg_joint2motor_idx) + list(cfg.arm_waist_joint2motor_idx)
        kps = list(cfg.kps) + list(cfg.arm_waist_kps)
        kds = list(cfg.kds) + list(cfg.arm_waist_kds)
        default_pos = np.concatenate([cfg.default_angles, cfg.arm_waist_target])

        init_dof_pos = np.array(
            [self.low_state.motor_state[idx].q for idx in dof_idx], dtype=np.float32,
        )

        for step in range(num_steps):
            alpha = step / num_steps
            for j, motor_idx in enumerate(dof_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(init_dof_pos[j] * (1 - alpha) + default_pos[j] * alpha)
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(kps[j])
                self.low_cmd.motor_cmd[motor_idx].kd = float(kds[j])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self._send_cmd()
            time.sleep(cfg.control_dt)

    def _hold_default_pos(self) -> None:
        print("[RealAdapter] Holding default pos — waiting for the remote control A button...")
        cfg = self.config
        while self._remote.button[self._KeyMap.A] != 1:
            for i, motor_idx in enumerate(cfg.leg_joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(cfg.default_angles[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(cfg.kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(cfg.kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            for i, motor_idx in enumerate(cfg.arm_waist_joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(cfg.arm_waist_target[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(cfg.arm_waist_kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(cfg.arm_waist_kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self._send_cmd()
            time.sleep(cfg.control_dt)

    def _create_zero_cmd(self) -> None:
        for mc in self.low_cmd.motor_cmd:
            mc.q = 0.0
            mc.qd = 0.0
            mc.kp = 0.0
            mc.kd = 0.0
            mc.tau = 0.0

    def _send_cmd(self) -> None:
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        self._publisher.Write(self.low_cmd)
