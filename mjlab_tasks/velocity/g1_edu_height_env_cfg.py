"""Env config for Rugiar-G1Edu-Velocity-Height — velocity-command walking
with a controllable base height, for GIAR's 29-DOF G1-EDU-U8.

Why a separate task (AGENTS.md rule 4): the command is 4-wide
``[vx, vy, wz, height]`` instead of 3-wide, the action space is the 12 leg
joints only (the upper body holds its default pose, exactly as agile's
sim2mujoco deployment does), and the actor observation is the raw-state
contract the LEAPP ``Velocity-Height-G1-History-v0`` ONNX expects:

    actor obs (69) = command(4) + base_ang_vel(3) + root_quat_w(4)
                     + joint_pos(29) + joint_vel(29)

The policy's own ONNX graph owns the 5-frame history and the projected
gravity computation (see legged_gym/control/policy.py's
LeappHistoryObsPolicy), so the env only has to produce the current raw
state. That same 69-dim actor is what a policy trained here sees, so a
mjlab-trained checkpoint and the external LEAPP export are interchangeable
on this task.

PD gains and the default pose come from the LEAPP bundle's
``agile.articulations.robot`` block (the values the policy was trained
with), not mjlab's armature-derived G1 defaults. The ankle gains in
particular differ enough to change the gait.

Command envelope matches agile's training config: vx [-0.5, 1.5],
vy [-0.5, 0.5], yaw [-1.0, 1.0], height [0.4, 0.72] m.
"""
from __future__ import annotations

import copy

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    ACTUATOR_4010,
    ACTUATOR_5020,
    ACTUATOR_7520_14,
    ACTUATOR_7520_22,
)
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp
from .height_command import UniformVelocityHeightCommandCfg

# The LEAPP bundle's default pose, expressed as name patterns in SDK order.
# Legs match mjlab's own home keyframe; the upper body is held at zero
# because the policy only controls the 12 leg joints.
LEAPP_HOME_POSE = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_shoulder_roll_joint": 0.0,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.0,
    ".*_wrist_roll_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
}

LEG_ACTUATOR_PATTERNS = (
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
)


def _leapp_actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    """The LEAPP bundle's per-joint stiffness/damping (SDK order), grouped
    by joint type. Effort limits/armatures stay at mjlab's own G1 values —
    the bundle only overrides the PD gains."""

    def group(names, stiffness, damping, actuator, multiplier=1.0):
        return BuiltinPositionActuatorCfg(
            target_names_expr=names,
            stiffness=stiffness,
            damping=damping,
            effort_limit=actuator.effort_limit * multiplier,
            armature=actuator.reflected_inertia * multiplier,
        )

    return (
        group((".*_hip_pitch_joint", ".*_hip_yaw_joint"), 100.0, 2.5, ACTUATOR_7520_14),
        group((".*_hip_roll_joint",), 100.0, 2.5, ACTUATOR_7520_22),
        group((".*_knee_joint",), 200.0, 5.0, ACTUATOR_7520_22),
        group(("waist_yaw_joint",), 300.0, 5.0, ACTUATOR_7520_14),
        group(("waist_roll_joint", "waist_pitch_joint"), 300.0, 5.0, ACTUATOR_5020, 2.0),
        group((".*_ankle_pitch_joint",), 20.0, 0.2, ACTUATOR_5020, 2.0),
        group((".*_ankle_roll_joint",), 20.0, 0.1, ACTUATOR_5020, 2.0),
        group((".*_shoulder_pitch_joint",), 90.0, 2.0, ACTUATOR_5020),
        group((".*_shoulder_roll_joint",), 60.0, 1.0, ACTUATOR_5020),
        group((".*_shoulder_yaw_joint",), 20.0, 0.4, ACTUATOR_5020),
        group((".*_elbow_joint",), 60.0, 1.0, ACTUATOR_5020),
        group((".*_wrist_roll_joint",), 4.0, 0.2, ACTUATOR_5020),
        group((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 4.0, 0.2, ACTUATOR_4010),
    )


def rugiar_g1_edu_height_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)

    robot = get_g1_robot_cfg()
    robot.articulation = EntityArticulationInfoCfg(
        actuators=_leapp_actuators(), soft_joint_pos_limit_factor=0.9
    )
    # get_g1_robot_cfg() shares its init_state object across calls; copy it
    # before overriding the pose or every other G1 task in this process
    # would silently inherit this one's home pose.
    robot.init_state = copy.deepcopy(robot.init_state)
    robot.init_state.joint_pos = dict(LEAPP_HOME_POSE)
    cfg.scene.entities["robot"] = robot

    # Legs only: the policy's action output is 12 absolute joint targets,
    # so the action term must be an identity transform over the same 12
    # joints. The 17 upper-body actuators are left out of the action term;
    # their position targets stay at zero, which is exactly the LEAPP
    # bundle's upper-body target.
    cfg.actions["joint_pos"].actuator_names = LEG_ACTUATOR_PATTERNS
    cfg.actions["joint_pos"].scale = 1.0
    cfg.actions["joint_pos"].use_default_offset = False

    cfg.commands["twist"] = UniformVelocityHeightCommandCfg(
        entity_name="robot",
        resampling_time_range=(7.5, 10.0),
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        ),
        height_range=(0.4, 0.72),
        default_height=0.72,
    )

    # Raw-state actor contract (69 values). Insertion order is the order
    # LeappHistoryObsPolicy parses, so keep it command -> ang_vel -> quat ->
    # joint_pos -> joint_vel.
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={
            "command": ObservationTermCfg(
                func=mjlab_mdp.generated_commands,
                params={"command_name": "twist"},
            ),
            "base_ang_vel": ObservationTermCfg(
                func=mjlab_mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_ang_vel"},
                noise=Unoise(n_min=-0.2, n_max=0.2),
            ),
            "root_quat_w": ObservationTermCfg(
                func=mdp.root_quat_w,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.01, n_max=0.01),
            ),
            "joint_vel": ObservationTermCfg(
                func=mdp.joint_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-1.5, n_max=1.5),
            ),
        },
        concatenate_terms=True,
        enable_corruption=not play,
    )

    # Height tracking alongside the stock velocity rewards. std=0.2 matches
    # agile's track_base_height_exp_smooth.
    cfg.rewards["track_base_height"] = RewardTermCfg(
        func=mdp.track_base_height,
        weight=1.0,
        params={"command_name": "twist", "std": 0.2},
    )

    return cfg
