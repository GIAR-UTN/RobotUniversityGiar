"""Env config for Rugiar-G1-Mimic.

Calls mjlab's OWN `unitree_g1_flat_tracking_env_cfg()` unmodified -- not
unitree_rl_mjlab's near-identical fork of it (see docs/mjlab_migration.md
§0 for why that fork isn't the dependency target) -- and not a repo-local
copy either. This keeps our task's env identical to mjlab's own
`Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation`, which is what
Javier Villalba's checkpoints were trained against (§0/§1), so registering
this task doesn't change their observation contract at all: it only gives
this repo's own training runs (Phase 6) a task_id and experiment_name
that are ours, not mjlab's stock ones.

If a repo-specific delta is ever needed (reward tweak, DR range change),
add it here by mutating the returned cfg -- same pattern
unitree_rl_mjlab's own fork uses relative to mjlab core -- so upstream
mjlab fixes keep flowing in instead of living in a frozen copy.
"""
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


def rugiar_g1_mimic_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)
