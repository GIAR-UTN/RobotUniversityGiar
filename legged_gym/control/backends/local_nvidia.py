"""
PLACEHOLDER — `local-nvidia`: training on a dedicated NVIDIA GPU attached to
the machine running this control server (CUDA, not Apple Metal and not a
remote kernel). Nothing here is implemented; this file exists so a
collaborator with such a box knows exactly where the work goes.

WHAT IT HAS TO PROVIDE, and nothing more (see local_genesis.py /
local_mjlab.py for the shape, backends/__init__.py for the 3-step
walkthrough):
  - `interpreter(manager, task) -> str` — the CUDA-capable venv's python,
    raising ValueError if it isn't installed on this machine.
  - `prepare_env(env) -> None` — PYTHONPATH/SIMULATOR like the local
    backends do, plus a real CUDA_VISIBLE_DEVICES instead of the ""
    local_mjlab.py forces.
  - a `TrainingBackend(...)` with `requested_as="local-nvidia"` (a NEW
    requestable name, so the CLI/web pick it up automatically) and the
    `task_stack` it serves ("genesis" and/or "mjlab" — one descriptor each,
    `(requested_as, task_stack)` must stay unique).

It is deliberately NOT in BACKENDS: an unimplemented entry would be offered
as a real choice by `rugiar --backend` and the control web. Register it by
adding `local_nvidia.BACKEND` to backends/__init__.py's BACKENDS once the
hooks above are real.
"""
from __future__ import annotations


def not_implemented(*_args, **_kwargs):
    raise NotImplementedError(
        "the local-nvidia training backend isn't implemented yet — see "
        "legged_gym/control/backends/local_nvidia.py for what it has to provide")
