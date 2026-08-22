"""
PLACEHOLDER — `nvidia-cloud`: training on NVIDIA's cloud stack (Isaac Lab /
Isaac Gym on rented GPUs), which a separate team is advancing in parallel.
Nothing here is implemented; this file exists so that work has an obvious
landing spot.

This is a REMOTE backend, so it looks like kaggle.py, not like the local
ones (see backends/__init__.py's walkthrough):
  - no interpreter/prepare_env at all — the job doesn't run on this machine;
  - `preflight() -> None` — credentials / quota check, raising ValueError
    with a human explanation when the account isn't set up here;
  - `launch_remote(manager, job, ctx) -> None` — hands the job to a
    background runner (kaggle.py's KaggleRunner is the reference: ALL
    network I/O on its own thread, polled cheaply from the sim loop);
  - a `TrainingBackend(..., remote=True, accepts_local_checkpoint=False)`
    with `requested_as="nvidia-cloud"` and the `simulator` string that
    should land on the resulting policy (e.g. "isaaclab").

Deliberately NOT in BACKENDS until it works — register it by adding
`nvidia_cloud.BACKEND` to backends/__init__.py's BACKENDS.
"""
from __future__ import annotations


def not_implemented(*_args, **_kwargs):
    raise NotImplementedError(
        "the nvidia-cloud training backend isn't implemented yet — see "
        "legged_gym/control/backends/nvidia_cloud.py for what it has to provide")
