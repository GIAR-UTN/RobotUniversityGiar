"""
`local-mjlab` — an mjlab motion-tracking task trained on THIS machine, as a
subprocess running legged_gym/scripts/mjlab_train.py under .venv-mjlab.

Jobs launched here are CPU (CUDA_VISIBLE_DEVICES is emptied); a CUDA-capable
variant is what backends/local_nvidia.py is a placeholder for.
"""
from __future__ import annotations

from typing import Dict

import os

from legged_gym.control.backends import base
from legged_gym.control.backends.base import TrainingBackend


def mjlab_interpreter(manager, task: str) -> str:
    """Always .venv-mjlab: neither venv can import the other's simulator
    (docs/mjlab_migration.md R1), so this is an interpreter choice, not just
    a script choice."""
    if not base.MJLAB_PYTHON.exists():
        raise ValueError(f"no mjlab venv at {base.MJLAB_PYTHON} — mjlab training isn't "
                         f"set up on this machine")
    return str(base.MJLAB_PYTHON)


def mjlab_prepare_env(env: Dict[str, str]) -> None:
    """The exact OPPOSITE of the Genesis case: REPO_ROOT must NOT be
    prepended here. The repo vendors a top-level rsl_rl/ that would shadow
    .venv-mjlab's PyPI rsl-rl-lib (docs/mjlab_migration.md R1). mjlab_train.py
    puts REPO_ROOT back on sys.path itself — LAST — so `mjlab_tasks`/
    `legged_gym` still resolve while rsl-rl-lib wins."""
    existing = env.get("PYTHONPATH", "")
    stripped = [p for p in existing.split(os.pathsep) if p and p != str(base.REPO_ROOT)]
    if stripped:
        env["PYTHONPATH"] = os.pathsep.join(stripped)
    else:
        env.pop("PYTHONPATH", None)
    env["SIMULATOR"] = "mjlab"
    # mjlab's own run_train() reads this to decide cpu vs cuda; the
    # jobs launched here are CPU unless a CUDA device was requested.
    env["CUDA_VISIBLE_DEVICES"] = ""


# Every start() knob that only means something for a Genesis locomotion task —
# a motion-tracking task has no velocity command, no stability targets and no
# pushes, so passing one is a mistake worth reporting in the panel rather than
# 10 seconds later as "the subprocess exited with code 2".
GENESIS_ONLY_PARAMS = (
    "cmd_vx", "cmd_vy", "cmd_yaw",
    "base_height_target", "lin_vel_z_target", "ang_vel_xy_target",
    "orientation_tilt_target",
    "push_robots", "max_push_vel_xy", "push_interval_s", "push_dir",
)


def mjlab_validate_params(params: Dict[str, object]) -> None:
    inapplicable = [name for name in GENESIS_ONLY_PARAMS if params.get(name) is not None]
    if inapplicable:
        raise ValueError(
            f"{', '.join(inapplicable)} don't apply to mjlab task '{params['task']}' "
            f"(motion-tracking task: no velocity command, no stability targets, "
            f"no pushes)")
    if not params.get("motion_file"):
        raise ValueError(f"task '{params['task']}' needs a --motion_file (reference-motion clip)")


BACKEND = TrainingBackend(
    id="local-mjlab",
    requested_as="local",
    task_stack="mjlab",
    job_backend="local",
    simulator="mjlab",
    command_prefix="rugiar train ",
    script=base.MJLAB_TRAIN_SCRIPT,
    # No --cpu/--headless prefix: mjlab_train.py never opens a viewer
    # and takes --device instead (it defaults to cpu).
    fixed_flags=(),
    interpreter=mjlab_interpreter,
    prepare_env=mjlab_prepare_env,
    validate_params=mjlab_validate_params,
    label="This machine",
)
