# g1_crouch reuses the plain G1Robot env class (legged_gym/envs/g1/g1.py) —
# only the reward weights/commands differ (see g1_crouch_config.py). This file exists
# so train.py's log-dir backup copy step (which expects env.asset.name/task/task.py
# for tasks whose name differs from the robot's asset.name) has something to copy.
from legged_gym.envs.g1.g1 import G1Robot  # noqa: F401
