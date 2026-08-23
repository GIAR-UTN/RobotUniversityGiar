"""
The training-backend registry.

WHAT THIS IS. A training backend is one answer to "where and how does a
training job actually run": which interpreter, which entrypoint script,
which subprocess env, which simulator ends up recorded on the resulting
policy, and which tasks it can serve at all. There are three today —
`local-genesis`, `local-mjlab`, `kaggle`, one module each — and
TrainingManager.start() contains ZERO knowledge of any of them: it validates
the request, resolves ONE descriptor out of BACKENDS, and drives that
descriptor's hooks. This replaced a set of scattered `if backend == "kaggle"
/ if train_backend == "mjlab"` branches that had to be edited in five places
at once.

  base.py           the TrainingBackend dataclass + the repo paths/interpreters
  local_genesis.py  Genesis locomotion tasks, on this machine
  local_mjlab.py    mjlab motion-tracking tasks, on this machine (CPU)
  kaggle.py         KaggleRunner + the remote Kaggle GPU descriptor
  local_nvidia.py   PLACEHOLDER — a dedicated local NVIDIA GPU (not registered)
  nvidia_cloud.py   PLACEHOLDER — NVIDIA's cloud stack (not registered)

HOW TO ADD A NEW BACKEND. Nothing in start()/poll()/the validation block
changes — adding one is these three steps and no others:

  1. Write a new module here with the hooks it needs, next to the existing
     ones:
       - an interpreter resolver `(manager, task) -> str` (raise ValueError
         with a human explanation if that venv isn't installed here);
       - a `prepare_env(env)` that mutates the subprocess env in place
         (PYTHONPATH/SIMULATOR/CUDA_VISIBLE_DEVICES/...);
       - optionally a `validate_params(params)` that rejects knobs this
         backend has no analogue for, and a `preflight()` for credentials
         or other "can this even run here" checks.
     A REMOTE backend (runs off this machine) skips interpreter/script/env
     entirely and supplies `launch_remote(manager, job, ctx)` instead —
     see kaggle.py's, which hands the job to a KaggleRunner thread.
  2. Declare one `TrainingBackend(...)` in that module and append it to
     BACKENDS below, declaring which `requested_as` value selects it and
     which `task_stack` it serves. `(requested_as, task_stack)` is the
     lookup key and must be unique; so is `(job_backend, simulator)`.
  3. If it's requestable by a NEW name (not "local"/"kaggle"), that name
     becomes valid automatically — REQUESTABLE_BACKENDS, the `rugiar
     --backend` choices and start()'s "unknown backend" message are all
     derived from BACKENDS.

The two NVIDIA placeholders are deliberately NOT in BACKENDS: they carry no
implementation yet, and a registered-but-broken entry would show up as a
real choice in the CLI and the web. Registering one is literally adding
`local_nvidia.BACKEND` to the list below once its hooks are real.

Deliberately NOT a plugin system: no dynamic imports, no config files, no
entry points. A new backend is a Python literal in this list, which is
exactly as much extensibility as three-to-five backends warrant.
"""
from __future__ import annotations

from typing import List, Optional

from legged_gym.control.backends import kaggle, local_genesis, local_mjlab
from legged_gym.control.backends.base import (
    GENESIS_PYTHON, MJLAB_PYTHON, MJLAB_TRAIN_SCRIPT, REPO_ROOT, TRAIN_SCRIPT,
    TrainingBackend,
)

BACKENDS: List[TrainingBackend] = [
    local_genesis.BACKEND,
    local_mjlab.BACKEND,
    kaggle.BACKEND,
]

# What start(backend=...) accepts, derived — never hand-maintained.
REQUESTABLE_BACKENDS = tuple(dict.fromkeys(b.requested_as for b in BACKENDS))


def requestable_backend_options() -> List[dict]:
    """[{id, label}] for every name start(backend=...) accepts — what a UI
    (the control web's toggle) or the CLI's `--backend` choices are built
    from, so neither hardcodes 'local'/'kaggle'. One entry per
    `requested_as`, labelled by the first descriptor that offers it (the two
    local ones are the same choice to the user; the task decides which stack
    actually runs)."""
    options: List[dict] = []
    for requested in REQUESTABLE_BACKENDS:
        descriptor = next(b for b in BACKENDS if b.requested_as == requested)
        options.append({"id": requested, "label": descriptor.label or descriptor.id,
                        "remote": descriptor.remote})
    return options


def resolve_training_backend(task: str, requested: str) -> TrainingBackend:
    """The one lookup: (what the caller asked for, what stack this task needs)
    -> the descriptor that serves it. Raises ValueError with the same
    messages start() used to raise inline — an unknown `requested` name, or a
    known one that can't serve this task's stack."""
    # Imported lazily: training.py imports THIS package at module level, and
    # which stack a task needs is a property of the task registries, which
    # live there. Looked up per call, so a test patching
    # training.training_backend_for_task still takes effect.
    from legged_gym.control.training import training_backend_for_task

    if requested not in REQUESTABLE_BACKENDS:
        raise ValueError(f"unknown backend '{requested}' — must be "
                         f"{' or '.join(repr(b) for b in REQUESTABLE_BACKENDS)}")
    stack = training_backend_for_task(task)
    for backend in BACKENDS:
        if backend.requested_as == requested and backend.task_stack == stack:
            return backend
    # The requested backend exists, it just can't serve this task's stack.
    candidate = next(b for b in BACKENDS if b.requested_as == requested)
    raise ValueError(candidate.unsupported_task_stack_error
                     or f"the {candidate.id} backend doesn't support {stack} tasks")


def backend_descriptor(job_backend: str, simulator: str) -> Optional[TrainingBackend]:
    """(job_backend, simulator) -> the one BACKENDS entry that combination
    identifies, or None if nothing matches (an unknown simulator recorded by
    an older/newer build). This pair is the real "throughput regime" key —
    two backends can share job_backend ("local-genesis" and "local-mjlab"
    both persist backend="local") but never share simulator, so the pair is
    unique. Shared by backend_for_job() (a live TrainingJob) and
    estimate()/history recording (a persisted history dict) so both use
    exactly the same resolution rule."""
    for backend in BACKENDS:
        if backend.job_backend == job_backend and backend.simulator == simulator:
            return backend
    return None


def backend_for_job(job) -> Optional[TrainingBackend]:
    """The descriptor a job already in flight was launched under, recovered
    from its own persisted (job_backend, simulator) pair rather than from a
    field stored on the job — so a TrainingJob written before this registry
    existed still resolves. None if nothing matches (an unknown simulator
    recorded by an older/newer build); callers fall back rather than raise."""
    return backend_descriptor(job.backend, job.simulator)


__all__ = [
    "BACKENDS", "REQUESTABLE_BACKENDS", "TrainingBackend",
    "requestable_backend_options", "resolve_training_backend",
    "backend_descriptor", "backend_for_job",
    "REPO_ROOT", "TRAIN_SCRIPT", "MJLAB_TRAIN_SCRIPT", "MJLAB_PYTHON", "GENESIS_PYTHON",
    "kaggle", "local_genesis", "local_mjlab",
]
