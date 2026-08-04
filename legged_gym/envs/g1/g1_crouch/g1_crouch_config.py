# The real definition of G1CrouchCfg / G1CrouchCfgPPO lives in
# legged_gym/envs/g1/g1_config.py, alongside G1RoughCfg (they share the base config's
# control/asset/PD-gain blocks and only diverge on rewards/commands/domain_rand). Re-exported
# here so train.py's log-dir backup copy step has a task_config.py file to copy.
from legged_gym.envs.g1.g1_config import G1CrouchCfg, G1CrouchCfgPPO  # noqa: F401
