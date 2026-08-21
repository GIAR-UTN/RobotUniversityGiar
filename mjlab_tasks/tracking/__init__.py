from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .g1_env_cfg import rugiar_g1_mimic_env_cfg
from .rl_cfg import rugiar_g1_mimic_ppo_runner_cfg

register_mjlab_task(
    task_id="Rugiar-G1-Mimic",
    env_cfg=rugiar_g1_mimic_env_cfg(),
    play_env_cfg=rugiar_g1_mimic_env_cfg(play=True),
    rl_cfg=rugiar_g1_mimic_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
