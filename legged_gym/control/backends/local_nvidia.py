"""
`local-nvidia` — training on a dedicated NVIDIA GPU attached to the machine
running this control server (CUDA), as a subprocess. The GPU counterpart to
the two CPU-only local backends (local_genesis.py / local_mjlab.py), which
are what `--backend local` resolves to on every platform today.

Two descriptors, one per task stack, both requestable as `local-nvidia`:

  - genesis tasks -> web_train.py under the Genesis venv, with the fixed
    flag pair (--headless --gpu), i.e. gs.init(backend=gs.gpu) + rsl_rl on
    cuda:0 (task_registry.py's sim_device derives from args.cpu). Genesis's
    GPU backend JIT requires Volta+ (sm_70+) hardware — a pre-sm_70 GPU
    should be treated as "no local-nvidia here", which is exactly what the
    preflight's real-context probe surfaces at job start rather than a
    mid-`gs.init` JIT crash;
  - mjlab tasks -> mjlab_train.py under .venv-mjlab, with --device cuda:0
    (mujoco-warp's own device for the env and the runner).

Both share ONE preflight: a real CUDA-usability probe
(cuda_utils.cuda_is_usable) that actually creates and uses a context — the
same guard the rugiar drivers apply before handing a device to a simulator
(see cuda_utils.py's module docstring) — so an enumerated-but-unusable GPU
(wedged driver, broken GSP firmware) fails fast with an actionable error
before a job is even launched, instead of dying inside the training
subprocess minutes in.

The two descriptors deliberately persist job_backend="local-nvidia" (not
"local"): (job_backend, simulator) is the unique key backend_for_job()
recovers a descriptor from (backends/__init__.py), and a separate history
bucket keeps estimate() from pooling GPU throughput with CPU-local runs —
the exact cross-regime mixing its docstring warns corrupts every estimate.

Not registered (see backends/__init__.py for the 3-step walkthrough) — this
file USED to be that placeholder; its hooks are real now, so the registration
is the single `BACKENDS` list there.
"""
from __future__ import annotations

from typing import Dict

import os

from legged_gym.control import cuda_utils
from legged_gym.control.backends import base
from legged_gym.control.backends.base import TrainingBackend
from legged_gym.control.backends.local_genesis import genesis_interpreter, genesis_prepare_env
from legged_gym.control.backends.local_mjlab import (
    mjlab_interpreter,
    mjlab_prepare_env,
    mjlab_validate_params,
)


def nvidia_preflight() -> None:
    """'Can this backend even run here' — a usable CUDA context on THIS
    machine, probed the way the rugiar drivers probe it (not just
    torch.cuda.is_available(), which can't tell an enumerated-but-unusable
    GPU apart from a working one). Raises ValueError with the probe's own
    actionable reason; TrainingManager.start() surfaces that in the panel."""
    usable, reason = cuda_utils.cuda_is_usable()
    if not usable:
        raise ValueError(
            f"the local-nvidia backend needs a usable CUDA device, but none was found: {reason}. "
            f"Use --backend local (CPU) instead, or fix the GPU and retry.")


def _ensure_cuda_visible(env: Dict[str, str]) -> None:
    """CUDA_VISIBLE_DEVICES="" (inherited from an mjlab session's own
    prepare_env, or a shell that emptied it) silently disables CUDA for the
    child process — the exact opposite of what this backend exists for.
    Leave any non-empty value alone (an operator pinning a specific GPU that
    way keeps it); only the explicit 'no GPU' marker is undone."""
    if env.get("CUDA_VISIBLE_DEVICES") == "":
        env.pop("CUDA_VISIBLE_DEVICES", None)


def nvidia_genesis_prepare_env(env: Dict[str, str]) -> None:
    genesis_prepare_env(env)  # PYTHONPATH pin + SIMULATOR=genesis (same as local-genesis)
    _ensure_cuda_visible(env)


def nvidia_mjlab_prepare_env(env: Dict[str, str]) -> None:
    mjlab_prepare_env(env)    # REPO_ROOT stripped + SIMULATOR=mjlab (same as local-mjlab)
    # mjlab_prepare_env's job is a CPU-only child, so it empties
    # CUDA_VISIBLE_DEVICES — undo exactly that: this backend exists to hand
    # the GPU to mjlab.
    _ensure_cuda_visible(env)


BACKEND_GENESIS = TrainingBackend(
    id="local-nvidia-genesis",
    requested_as="local-nvidia",
    task_stack="genesis",
    job_backend="local-nvidia",
    simulator="genesis",
    command_prefix="rugiar train --backend local-nvidia ",
    script=base.TRAIN_SCRIPT,
    # --headless is web_train.py's default anyway; passed explicitly to match
    # what a human would type. --gpu is the whole point: gs.init(backend=gs.gpu)
    # and rsl_rl's tensors on cuda:0 (see web_train.py's --gpu docstring).
    fixed_flags=("--headless", "--gpu"),
    interpreter=genesis_interpreter,
    prepare_env=nvidia_genesis_prepare_env,
    preflight=nvidia_preflight,
    label="This machine (NVIDIA GPU)",
)

BACKEND_MJLAB = TrainingBackend(
    id="local-nvidia-mjlab",
    requested_as="local-nvidia",
    task_stack="mjlab",
    job_backend="local-nvidia",
    simulator="mjlab",
    command_prefix="rugiar train --backend local-nvidia ",
    script=base.MJLAB_TRAIN_SCRIPT,
    # mjlab_train.py takes --device, not --cpu/--headless/--gpu (it never
    # opens a viewer and its cpu default is what local-mjlab relies on) — the
    # CUDA request is exactly the one flag pair that differs from CPU.
    fixed_flags=("--device", "cuda:0"),
    interpreter=mjlab_interpreter,
    prepare_env=nvidia_mjlab_prepare_env,
    validate_params=mjlab_validate_params,
    preflight=nvidia_preflight,
    label="This machine (NVIDIA GPU)",
)
