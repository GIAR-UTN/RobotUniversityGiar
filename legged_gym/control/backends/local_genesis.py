"""
`local-genesis` — a Genesis locomotion task trained on THIS machine, as a
subprocess running legged_gym/scripts/web_train.py under this repo's .venv.

Compute is whatever `gs.init()` resolves for the platform (Metal on an Apple
Silicon Mac, CUDA on a Linux box with an NVIDIA GPU) — the descriptor itself
makes no hardware claim beyond "here".
"""
from __future__ import annotations

from typing import Dict

import os

from legged_gym.control.backends import base
from legged_gym.control.backends.base import TrainingBackend


def genesis_interpreter(manager, task: str) -> str:
    """This process's own interpreter — unless it's the mjlab venv, which
    has no Genesis at all. Same 'switch venv, or refuse' shape as
    rugiar_driver_mjlab.py's family switch."""
    interpreter = manager.python_exe
    # Compared UNRESOLVED on purpose: a venv's bin/python is a symlink
    # to the base interpreter, so .resolve() throws away the very
    # ".venv-mjlab" marker this needs to see.
    if ".venv-mjlab" in str(interpreter):
        if not base.GENESIS_PYTHON.exists():
            raise ValueError(f"no Genesis venv at {base.GENESIS_PYTHON} — can't train "
                             f"task '{task}' from an mjlab session")
        interpreter = str(base.GENESIS_PYTHON)
    return interpreter


def genesis_prepare_env(env: Dict[str, str]) -> None:
    """Pin PYTHONPATH to THIS repo checkout explicitly rather than trusting
    whatever the parent process happened to be launched with — an editable
    `pip install -e` of legged_gym elsewhere (e.g. a sibling checkout of this
    same repo) would otherwise silently win, running web_train.py's file from
    here against a DIFFERENT legged_gym package. Bit us once already getting
    the control server itself to run against the right checkout — not
    leaving it to chance twice."""
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(base.REPO_ROOT) + (os.pathsep + existing if existing else "")
    # Explicit: an mjlab session's inherited SIMULATOR=mjlab would
    # otherwise make legged_gym/__init__.py skip the Genesis import.
    env["SIMULATOR"] = "genesis"


BACKEND = TrainingBackend(
    id="local-genesis",
    requested_as="local",
    task_stack="genesis",
    job_backend="local",
    simulator="genesis",
    command_prefix="rugiar train ",
    script=base.TRAIN_SCRIPT,
    fixed_flags=("--headless", "--cpu"),
    interpreter=genesis_interpreter,
    prepare_env=genesis_prepare_env,
    label="This machine",
)
