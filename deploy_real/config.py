"""RobotConfig — loads a deploy_real/configs/*.yaml (see g1.yaml's own header
comment for why this is NOT the same schema as unitree_rl_gym's deploy_real
config.py: the normalization/command_ranges block is this repo's training
config, not unitree_rl_gym's stock cmd_scale/max_cmd)."""
from __future__ import annotations

import numpy as np
import yaml


class RobotConfig:
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.control_dt = config["control_dt"]

        self.msg_type = config["msg_type"]
        self.imu_type = config["imu_type"]

        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]

        self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
        self.kps = config["kps"]
        self.kds = config["kds"]
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)

        self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
        self.arm_waist_kps = config["arm_waist_kps"]
        self.arm_waist_kds = config["arm_waist_kds"]
        self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

        self.ang_vel_scale = config["ang_vel_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = config["action_scale"]
        self.commands_scale = np.array(config["commands_scale"], dtype=np.float32)

        ranges = config["command_ranges"]
        self.command_ranges = {
            "lin_vel_x": tuple(ranges["lin_vel_x"]),
            "lin_vel_y": tuple(ranges["lin_vel_y"]),
            "ang_vel_yaw": tuple(ranges["ang_vel_yaw"]),
        }

        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
