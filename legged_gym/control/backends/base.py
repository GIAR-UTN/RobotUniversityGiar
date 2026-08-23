"""
The TrainingBackend descriptor — the shape every training backend in this
package fills in — plus the repo paths those descriptors point at.

A descriptor is data, not a class hierarchy: one frozen dataclass per place
a training job can run, with optional callables for the few things that
genuinely differ (which interpreter, how the subprocess env is prepared,
what this backend refuses). TrainingManager.start() contains ZERO knowledge
of any individual backend — it resolves ONE descriptor and drives its hooks.
See backends/__init__.py for the registry and the "how do I add one" walkthrough.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "web_train.py"
# mjlab tasks train through their own entrypoint under their own interpreter —
# neither venv can import the other's simulator (docs/mjlab_migration.md R1),
# so this is an interpreter choice, not just a script choice. Mirrors
# rugiar_driver_mjlab.py's _script_for_task()/_argv_for_family_switch() pair.
MJLAB_TRAIN_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "mjlab_train.py"
MJLAB_PYTHON = REPO_ROOT / ".venv-mjlab" / "bin" / "python"
GENESIS_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


@dataclasses.dataclass(frozen=True)
class TrainingBackend:
    """One place a training job can run. See backends/__init__.py's own
    comment for the "how do I add one" walkthrough."""

    id: str                     # registry key / display name, e.g. "local-mjlab"
    requested_as: str           # the start(backend=...) value that selects this one
    task_stack: str             # which training_backend_for_task() answer it serves
    job_backend: str            # what lands in TrainingJob.backend — PERSISTED SHAPE,
                                # keep it stable ("local"/"kaggle") for meta.json/UI compat
    simulator: str              # what lands in TrainingJob.simulator / meta.json
    command_prefix: str         # the `rugiar` CLI equivalent shown to the user
    script: Path                # entrypoint the job actually runs (remotely, for a remote backend)
    remote: bool = False        # runs off this machine: no local subprocess to launch or poll
    fixed_flags: tuple = ()     # argv flags this entrypoint always takes (e.g. --headless --cpu)
    interpreter: Optional[object] = None     # (manager, task) -> str; None for a remote backend
    prepare_env: Optional[object] = None     # (env: dict) -> None, mutated in place
    validate_params: Optional[object] = None  # (params: dict) -> None, raises ValueError
    preflight: Optional[object] = None       # () -> None, "can this even run here" check
    launch_remote: Optional[object] = None   # (manager, job, ctx) -> None; remote backends only
    # Whether a local --from_checkpoint absolute path is meaningful to this
    # backend. False for anything running off this machine — it has no access
    # to this filesystem (KaggleRunner uploads the file as a private Dataset
    # and adds its own --from_checkpoint pointing at the /kaggle/input/ mount).
    accepts_local_checkpoint: bool = True
    # Message for "this backend exists but can't serve this task's stack".
    # None falls back to a generic sentence.
    unsupported_task_stack_error: Optional[str] = None

    # Display label for a UI that offers this backend as a choice (the
    # control web builds its toggle from system_info()'s backends list).
    label: Optional[str] = None
