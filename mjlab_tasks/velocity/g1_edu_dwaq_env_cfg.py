"""Env config for Rugiar-G1Edu-Velocity-DWAQ — the SAME 29-DOF G1-EDU body
as the other velocity tasks, with the observation/action contract of the
DWAQ blind-stairs policy (`policies/unitree_g1_29dof_velocity_stairs`,
imported from GIAR's deploy_mujoco2026 `velocity/dwaq`).

Why a separate task (AGENTS.md rule 4): every part of the contract below is
STRUCTURAL, and none of it matches Rugiar-G1Edu-Velocity-UnitreeObs:

  actor obs (100) = base_ang_vel(3) projected_gravity(3) command(3)
                    joint_pos_rel(29) joint_vel_rel(29) last_action(29)
                    gait_phase(4)                       -> 100
  history         = 5 frames of the same 100, term-major, fed to a SECOND
                    ONNX input ("history", 500). The env itself only produces
                    the current 100; OnnxHistoryObsPolicy owns the 5-frame
                    buffer (see legged_gym/control/policy.py).
  gait phase      = [sin_L, sin_R, cos_L, cos_R], period 0.8 s, offset 0.5,
                    NOT zeroed while standing.
  default pose    = DWAQ's training pose (arms 0.3/±0.25/0/0.97/±0.15),
                    which feeds joint_pos_rel AND the action offset.
  action scale    = 0.25 for every joint (not mjlab's per-actuator
                    effort/stiffness scale).
  joint order     = the policy was trained in Isaac Lab order, not the
                    SDK/mjlab order. deploy.yaml's `joint_ids_map` (policy
                    index -> SDK index) is applied by
                    OnnxHistoryObsPolicy on obs in and actions out.
  PD gains        = deploy.yaml's stiffness/damping (SDK order): the policy
                    was trained with these, not mjlab's armature-derived
                    gains.
  command ranges  = vx [-0.6, 1.0], vy [-0.4, 0.4], yaw [-1.57, 1.57].

Terrain: flat plane. This task's DEFAULT PLAY SCENARIO is
`obstacle_course` (see mjlab_scenarios.py's TASK_DEFAULT_SCENARIOS and
rugiar_driver_mjlab.py) — the 10-obstacle event's props are built into the
MjSpec on top of this plane, so the blind policy gets the obstacle course
the rest of the project already uses, not mjlab's random terrain
generator. The policy is blind (no height scan), so `terrain_scan` /
`height_scan` are not part of the actor.
"""
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
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from . import mdp

# deploy.yaml's default_joint_pos is in the policy's Isaac Lab joint order.
# Reordered here into mjlab/SDK order with that file's joint_ids_map:
#   [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
#    24, 18, 25, 19, 26, 20, 27, 21, 28]
# This is the same pose deploy_mujoco2026 starts the DWAQ policy from.
DWAQ_HOME_POSE = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.25,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.97,
    "right_wrist_roll_joint": -0.15,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


def _dwaq_actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    """deploy.yaml's stiffness/damping, in SDK order, grouped by joint type.

    Armature/effort limits are left at mjlab's own G1 values — deploy.yaml
    only overrides the PD gains, and the DWAQ deployment keeps the model's
    actuator limits.
    """

    def group(names, stiffness, damping, actuator, multiplier=1.0):
        return BuiltinPositionActuatorCfg(
            target_names_expr=names,
            stiffness=stiffness,
            damping=damping,
            effort_limit=actuator.effort_limit * multiplier,
            armature=actuator.reflected_inertia * multiplier,
        )

    return (
        group((".*_hip_pitch_joint", ".*_hip_yaw_joint"), 100.0, 2.0, ACTUATOR_7520_14),
        group((".*_hip_roll_joint",), 100.0, 2.0, ACTUATOR_7520_22),
        group((".*_knee_joint",), 150.0, 4.0, ACTUATOR_7520_22),
        group(("waist_yaw_joint",), 200.0, 5.0, ACTUATOR_7520_14),
        group(("waist_roll_joint", "waist_pitch_joint"), 40.0, 5.0, ACTUATOR_5020, 2.0),
        group((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), 40.0, 2.0, ACTUATOR_5020, 2.0),
        group(
            (
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ),
            40.0,
            1.0,
            ACTUATOR_5020,
        ),
        group((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 40.0, 1.0, ACTUATOR_4010),
    )


def rugiar_g1_edu_dwaq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)

    robot = get_g1_robot_cfg()
    robot.articulation = EntityArticulationInfoCfg(
        actuators=_dwaq_actuators(), soft_joint_pos_limit_factor=0.9
    )
    # get_g1_robot_cfg() shares its init_state object across calls; copy it
    # before overriding the pose or the last-registered G1 task would win.
    robot.init_state = copy.deepcopy(robot.init_state)
    robot.init_state.joint_pos = dict(DWAQ_HOME_POSE)
    cfg.scene.entities["robot"] = robot

    cfg.actions["joint_pos"].scale = 0.25

    old = cfg.observations["actor"].terms
    cfg.observations["actor"].terms = {
        "base_ang_vel": old["base_ang_vel"],
        "projected_gravity": old["projected_gravity"],
        "command": old["command"],
        "joint_pos": old["joint_pos"],
        "joint_vel": old["joint_vel"],
        "actions": old["actions"],
        "phase": ObservationTermCfg(
            func=mdp.gait_phase, params={"period": 0.8, "offset": 0.5}
        ),
    }

    # unitree_g1_flat_env_cfg already drops the height scan and terrain
    # sensor; keep this defensive so a future base change can't silently
    # reintroduce a scan the blind policy does not expect.
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )

    r = cfg.commands["twist"].ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.6, 1.0), (-0.4, 0.4), (-1.57, 1.57)
    return cfg
