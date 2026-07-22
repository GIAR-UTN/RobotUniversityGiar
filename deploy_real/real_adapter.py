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
    """See module docstring. Construct with a deploy_real-style yaml config
    (same schema as unitree_rl_gym's deploy/deploy_real/configs/*.yaml) and
    a network interface name (e.g. 'enp3s0')."""

    num_envs = 1  # a real robot is always exactly one instance

    def __init__(self, config, net_interface: str):
        # Imported here, not at module level — see module docstring.
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
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

        ChannelFactoryInitialize(0, net_interface)
        self._publisher = ChannelPublisher(config.lowcmd_topic, LowCmd)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber(config.lowstate_topic, LowState)
        self._subscriber.Init(self._on_low_state, 10)

        self.num_actions = config.num_actions
        self._counter = 0

    def _on_low_state(self, msg) -> None:
        self.low_state = msg

    # ---- RobotAdapter protocol ----

    def reset(self) -> RobotState:
        """Blocks through the physical gating sequence — this is where a
        human with the remote control (or an automated bench-test rig with
        its own button simulator) has to actually be present. Returns once
        the robot is standing at its default pose, lifecycle READY."""
        self._zero_torque_state()
        self._move_to_default_pos()
        self._hold_default_pos()
        self._lifecycle = Lifecycle.READY
        return self.get_state()

    def get_state(self) -> RobotState:
        cfg = self.config
        num_actions = cfg.num_actions

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
            commands=torch.zeros(1, 3),  # caller (e.g. a joystick reader) is expected to fill this in
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
        self._counter += 1
        time.sleep(self.config.control_dt)
        return self.get_state()

    def get_observations(self) -> torch.Tensor:
        """Not part of RobotAdapter (see SimAdapter's identical note) — the
        obs vector layout is env/task-specific. Kept here as a convenience
        matching deploy_real.py's own obs-building so a script wiring this
        up doesn't have to duplicate it, but callers should treat this as
        RealAdapter-specific, not part of the generic interface."""
        raise NotImplementedError(
            "Port deploy_real.py's Controller.run() observation-building block here "
            "once wiring this up against real hardware — it depends on the specific "
            "task's obs layout (gait-phase clock, command scaling, etc.) the same way "
            "SimAdapter's env.get_observations() does."
        )

    def record(self, obs: torch.Tensor, action: torch.Tensor, state: RobotState) -> None:
        pass

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

    # ---- physical gating sequence (ported from deploy_real.py's Controller) ----

    def _zero_torque_state(self) -> None:
        raise NotImplementedError(
            "Port Controller.zero_torque_state() from unitree_rl_gym's deploy_real.py: "
            "holds zero torque, polls the remote control's 'start' button, requires a "
            "human present. Not implementable/testable without hardware."
        )

    def _move_to_default_pos(self) -> None:
        raise NotImplementedError(
            "Port Controller.move_to_default_pos() — ramps to default_angles over ~2s."
        )

    def _hold_default_pos(self) -> None:
        raise NotImplementedError(
            "Port Controller.default_pos_state() — holds pose, polls the remote "
            "control's 'A' button before handing control to the policy loop."
        )

    def _send_cmd(self) -> None:
        # Port of Controller.send_cmd(): CRC the message and publish it.
        raise NotImplementedError(
            "Port Controller.send_cmd() — computes CRC via unitree_sdk2py's crc_module "
            "and calls self._publisher.Write(self.low_cmd)."
        )
