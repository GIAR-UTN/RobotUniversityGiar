from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from .g1_edu_dwaq_env_cfg import rugiar_g1_edu_dwaq_env_cfg
from .g1_edu_env_cfg import rugiar_g1_edu_velocity_env_cfg
from .g1_edu_height_env_cfg import rugiar_g1_edu_height_env_cfg
from .g1_edu_unitree_obs_env_cfg import rugiar_g1_edu_unitree_obs_env_cfg

_rl_cfg = unitree_g1_ppo_runner_cfg()
_rl_cfg.experiment_name = "rugiar_g1_edu_velocity"

register_mjlab_task(
    task_id="Rugiar-G1Edu-Velocity",
    env_cfg=rugiar_g1_edu_velocity_env_cfg(),
    play_env_cfg=rugiar_g1_edu_velocity_env_cfg(play=True),
    rl_cfg=_rl_cfg,
    runner_cls=MjlabOnPolicyRunner,
)

_rl_cfg_unitree = unitree_g1_ppo_runner_cfg()
_rl_cfg_unitree.experiment_name = "rugiar_g1_edu_velocity_unitree_obs"

register_mjlab_task(
    task_id="Rugiar-G1Edu-Velocity-UnitreeObs",
    env_cfg=rugiar_g1_edu_unitree_obs_env_cfg(),
    play_env_cfg=rugiar_g1_edu_unitree_obs_env_cfg(play=True),
    rl_cfg=_rl_cfg_unitree,
    runner_cls=MjlabOnPolicyRunner,
)

_rl_cfg_dwaq = unitree_g1_ppo_runner_cfg()
_rl_cfg_dwaq.experiment_name = "rugiar_g1_edu_velocity_dwaq"

register_mjlab_task(
    task_id="Rugiar-G1Edu-Velocity-DWAQ",
    env_cfg=rugiar_g1_edu_dwaq_env_cfg(),
    play_env_cfg=rugiar_g1_edu_dwaq_env_cfg(play=True),
    rl_cfg=_rl_cfg_dwaq,
    runner_cls=MjlabOnPolicyRunner,
)

_rl_cfg_height = unitree_g1_ppo_runner_cfg()
_rl_cfg_height.experiment_name = "rugiar_g1_edu_velocity_height"

register_mjlab_task(
    task_id="Rugiar-G1Edu-Velocity-Height",
    env_cfg=rugiar_g1_edu_height_env_cfg(),
    play_env_cfg=rugiar_g1_edu_height_env_cfg(play=True),
    rl_cfg=_rl_cfg_height,
    runner_cls=MjlabOnPolicyRunner,
)
