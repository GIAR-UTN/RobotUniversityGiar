"""Env config for Rugiar-G1Edu-Velocity — velocity-command walking for
GIAR's real robot, a Unitree G1-EDU-U8.

The EDU-U8's body is the full 29-DOF G1: legs 6+6, waist 3 (unlocked) and
arms 7+7. That is exactly mjlab's own `unitree_g1` asset, so this task
builds on mjlab's `unitree_g1_flat_env_cfg()` unmodified except for one
STRUCTURAL change (AGENTS.md rule 4): `base_lin_vel` is removed from the
ACTOR observation group. The real robot has no sensor for base linear
velocity, and a policy that reads it cannot be deployed (README
"Real Robot Deployment" step 1). The critic keeps it, because privileged
critic observations are fine at training time.

The EDU-U8's two BrainCo Revo 2 hands (12 more DOF) are NOT modeled here.
mjlab's G1 has no hands, and locomotion doesn't need them. Hand modeling is
a separate task, once a Revo 2 MJCF exists.

Actor obs = base_ang_vel(3) + projected_gravity(3) + joint_pos(29) +
joint_vel(29) + actions(29) + command(3) = 96.
"""
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def rugiar_g1_edu_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)
    del cfg.observations["actor"].terms["base_lin_vel"]
    return cfg
