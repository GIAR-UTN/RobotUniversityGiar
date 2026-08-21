import os
import sys

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(f"LEGGED_GYM_ROOT_DIR: {LEGGED_GYM_ROOT_DIR}")
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')

if sys.version_info[1] >= 10: # >=3.10 for genesis and isaacsim
    simulator_type = os.getenv("SIMULATOR")
    if simulator_type == "genesis":
        SIMULATOR = "genesis"
    elif simulator_type == "isaaclab":
        SIMULATOR = "isaaclab"
    elif simulator_type == "mjlab":
        # mjlab (MuJoCo Warp) tasks don't live in legged_gym/envs at all --
        # they're registered through mjlab's own registry by the repo-root
        # `mjlab_tasks/` package (see docs/mjlab_migration.md phase 3), and
        # run from a separate venv (.venv-mjlab) that deliberately has no
        # Genesis/Isaac installed (R1: incompatible mujoco pins + an rsl_rl
        # name collision). legged_gym/scripts/rugiar_driver_mjlab.py still
        # needs legged_gym.control (ControlService/ControlServer/
        # PolicySupervisor/SafetyGovernor are backend-agnostic by design --
        # see legged_gym/control/adapter.py's docstring), so this value
        # exists purely to let that package import with NO simulator import
        # at all. Nothing under legged_gym/envs or legged_gym/simulator
        # supports it, and nothing there should ever be imported under it.
        SIMULATOR = "mjlab"
    else:
        raise ValueError("Unsupported SIMULATOR type. Please set the SIMULATOR environment variable to 'genesis', 'isaaclab' or 'mjlab'.")
elif sys.version_info[1] <= 8 and sys.version_info[1] >= 6: # >=3.6 and <3.9 for isaacgym
    SIMULATOR = "isaacgym"

if SIMULATOR == "genesis":
    try: 
        import genesis as gs
    except ImportError as e:
        print("Failed to import Genesis. Please ensure that the Genesis is properly installed and configured.")
        raise e
elif SIMULATOR == "isaacgym":
    try:
        import isaacgym
    except ImportError as e:
        print("Failed to import Isaac Gym. Please ensure that the Isaac Gym is properly installed and configured.")
        raise e
elif SIMULATOR == "isaaclab":
    try:
        import isaaclab
    except ImportError as e:
        print("Failed to import Isaac Lab. Please ensure that the Isaac Lab is properly installed and configured.")
        raise e