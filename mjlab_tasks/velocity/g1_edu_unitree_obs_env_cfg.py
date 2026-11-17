"""Env config for Rugiar-G1Edu-Velocity-UnitreeObs — the SAME 29-DOF G1
walking world as Rugiar-G1Edu-Velocity, but with the observation contract
and home pose of unitree_rl_mjlab's published G1 velocity policy
(deploy/robots/g1/config/policy/velocity/v0, Apache-2.0). Its exported
policy.onnx runs here unmodified.

Why a separate task (AGENTS.md rule 4): the observation layout (98 values,
with a gait-phase clock) and the default joint pose are both STRUCTURAL.
The default pose feeds joint_pos_rel and the action offset, so a policy
trained against one pose is wrong against another.

Contract, from that policy's params/deploy.yaml:
  actor obs = base_ang_vel(3) projected_gravity(3) command(3) phase(2)
              joint_pos_rel(29) joint_vel_rel(29) last_action(29) = 98
  PD gains and action scale: identical to mjlab's own G1 (verified
  2026-11-17), so they are left untouched.
  default pose: legs -0.1/0/0/0.3/-0.2/0, waist 0,
                arms 0.35/±0.18/0/0.87/0/0/0
  command ranges: vx [-0.5, 1.0], vy [-0.5, 0.5], yaw [-1.0, 1.0]
"""
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from . import mdp

UNITREE_HOME_POSE = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    ".*_elbow_joint": 0.87,
}


def rugiar_g1_edu_unitree_obs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)
    cfg.scene.entities["robot"].init_state.joint_pos = dict(UNITREE_HOME_POSE)

    old = cfg.observations["actor"].terms
    cfg.observations["actor"].terms = {
        "base_ang_vel": old["base_ang_vel"],
        "projected_gravity": old["projected_gravity"],
        "command": old["command"],
        "phase": ObservationTermCfg(func=mdp.phase, params={"period": 0.6, "command_name": "twist"}),
        "joint_pos": old["joint_pos"],
        "joint_vel": old["joint_vel"],
        "actions": old["actions"],
    }

    r = cfg.commands["twist"].ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0)
    return cfg
