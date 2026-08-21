"""
TrainingManager — lets the control web start a new policy training run and
find out, asynchronously, when it's ready to load. It owns exactly one
thing: launching legged_gym/scripts/web_train.py as a subprocess per job and
polling it. It never touches PolicySupervisor/ControlService/RobotAdapter —
same boundary the rest of legged_gym/control/ keeps (see
HANDOFF_control_web.md §5); the caller (rugiar_driver.py's sim loop, via
ControlService — see service.py's start_training()/poll_finished_training())
is what actually loads the resulting checkpoint and registers it as a new
policy, exactly like restart_requested is drained there today.

Why a subprocess instead of an in-process training loop: `train.py`'s whole
stack (Genesis/gs.init, task_registry.make_env, the PPO runner) is built to
own a single process's global simulator state — running it would collide
with the rugiar_driver.py sim already using the same globals. A subprocess
is the natural isolation boundary, and it's also what makes this safe to
poll cheaply (Popen.poll(), no subprocess.wait()) from a real-time control
loop.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from legged_gym.control import kaggle_backend

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "web_train.py"
DISTILL_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "web_distill.py"
# mjlab tasks train through their own entrypoint under their own interpreter —
# neither venv can import the other's simulator (docs/mjlab_migration.md R1),
# so this is an interpreter choice, not just a script choice. Mirrors
# rugiar_driver_mjlab.py's _script_for_task()/_argv_for_family_switch() pair.
MJLAB_TRAIN_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "mjlab_train.py"
MJLAB_PYTHON = REPO_ROOT / ".venv-mjlab" / "bin" / "python"
GENESIS_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
JOBS_DIR = REPO_ROOT / "logs" / "_web_training"
HISTORY_PATH = JOBS_DIR / "history.json"
# One self-contained folder per policy trained through this UI — see
# finalize_policy()'s docstring for why this replaced leaving the exported
# checkpoint sitting wherever rsl_rl's log_dir happened to be.
POLICIES_DIR = REPO_ROOT / "policies"
# Display order for the local policy catalog — deliberately NOT inside any
# policies/<name>/ folder (those are self-contained and get renamed/deleted
# as a unit) or inside meta.json (order is a property of the CATALOG, not of
# any one policy). A separate control-layer app is expected to read this via
# TrainingManager.get_policy_order() to decide what it offers first, without
# having any opinion of its own about training/naming.
POLICY_ORDER_PATH = POLICIES_DIR / ".policy_order.json"


def _cpu_brand() -> str:
    """platform.processor() returns '' on macOS — sysctl has the real
    string ('Apple M1 Pro', etc.); Linux falls back to /proc/cpuinfo's
    'model name'. Either miss just falls back to platform.machine()."""
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001 - this is cosmetic, never worth failing over
        pass
    return platform.processor() or platform.machine() or "unknown"


def _total_ram_bytes() -> Optional[int]:
    try:
        if platform.system() == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip())
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 - cosmetic
        pass
    return None


# ---- which training stack a task belongs to ----
#
# Neither venv can import the other's registry: under `.venv-mjlab`,
# `from legged_gym.utils import task_registry` raises (no Genesis), and under
# the main `.venv`, `import mjlab` raises (no mjlab). So whichever registry IS
# importable here is authoritative, and the other side is decided by exclusion
# — the same asymmetry rugiar_driver_mjlab.py's _script_for_task() relies on.


def _mjlab_registered_tasks() -> Optional[set]:
    """Task ids in mjlab's registry, or None if mjlab isn't importable here."""
    try:
        import mjlab_tasks  # noqa: F401 - import side effect: registers this repo's own tasks
        from mjlab.tasks import registry
    except ImportError:
        return None
    tasks = registry.list_tasks()
    # Same "empty means unknown, not none-exist" reasoning as
    # _legged_gym_registered_tasks() below.
    return set(tasks) if tasks else None


def _legged_gym_registered_tasks() -> Optional[set]:
    """Genesis/Isaac task names, or None if legged_gym's registry isn't
    importable here (the normal case under `.venv-mjlab` — same
    try/except ImportError shape as ControlService._registered_task_names()).

    `legged_gym.envs` is imported for its SIDE EFFECT: task_registry starts
    empty and every task_registry.register() call lives in that module. A
    bare `from legged_gym.utils import task_registry` yields an empty dict
    in a process that hasn't imported envs yet — which would make
    training_backend_for_task() answer "mjlab" (by exclusion) for every
    Genesis task, silently routing e.g. `g1` to the mjlab entrypoint. The
    long-running drivers happen to import envs before ever calling this,
    which is exactly what would have hidden it."""
    try:
        import legged_gym.envs  # noqa: F401 - import side effect: populates task_classes
        from legged_gym.utils import task_registry
    except ImportError:
        return None
    # `from legged_gym.utils import task_registry` normally yields the
    # TaskRegistry INSTANCE that legged_gym/utils/__init__.py re-exports, but
    # resolves to the SUBMODULE of the same name whenever that __init__ hasn't
    # run its re-export (e.g. a partially-stubbed legged_gym package). Accept
    # either shape rather than depending on which one this process ended up with.
    tasks = getattr(task_registry, "task_classes", None)
    if tasks is None:
        tasks = getattr(getattr(task_registry, "task_registry", None), "task_classes", None)
    # Empty is NOT "there are no Genesis tasks" — it's "this process couldn't
    # read the registry", and answering "mjlab" by exclusion off an empty set
    # would misroute every Genesis job. Say "unknown" instead.
    return set(tasks) if tasks else None


def _mjlab_registry_probe_source() -> str:
    """The one-shot script run via `_mjlab_registry_snapshot()`'s subprocess
    fallback. Run as a SCRIPT FILE (not `python -c`) — `-c` makes Python
    insert cwd as sys.path[0], which if cwd is REPO_ROOT reintroduces the
    exact R1 collision `_mjlab_prepare_env()`'s PYTHONPATH-stripping exists
    to avoid (the repo's own vendored rsl_rl/ shadowing .venv-mjlab's PyPI
    rsl-rl-lib — see docs/mjlab_migration.md R1). As a script file, sys.path[0]
    is the script's own directory instead, so REPO_ROOT only needs to go back
    on sys.path explicitly, LAST, exactly like mjlab_train.py's own header —
    see that file's module docstring for the same reasoning."""
    return (
        "import json\n"
        "import sys\n"
        f"sys.path.append({str(REPO_ROOT)!r})  # LAST -- see this function's docstring\n"
        "import mjlab_tasks  # noqa: F401,E402 - import side effect: registers this repo's own tasks\n"
        "from mjlab.tasks import registry\n"
        "out = {}\n"
        "for t in registry.list_tasks():\n"
        "    cfg = registry.load_env_cfg(t)\n"
        "    out[t] = {\n"
        "        'reward_scales': {n: term.weight for n, term in cfg.rewards.items()},\n"
        "        'needs_motion_file': 'motion' in getattr(cfg, 'commands', {}),\n"
        "    }\n"
        "print(json.dumps(out))\n"
    )


@functools.lru_cache(maxsize=1)
def _mjlab_registry_snapshot() -> Optional[dict]:
    """{task_id: {'reward_scales': {...}, 'needs_motion_file': bool}} for
    every mjlab-registered task — computed in-process when mjlab is
    importable here, otherwise via a one-shot subprocess into
    `.venv-mjlab` (the same interpreter `_mjlab_interpreter()` dispatches
    training to), so a process that can't import mjlab itself (e.g.
    rugiar's own `.venv` install, where `rugiar` the console-script
    actually lives) still gets real reward-term data instead of an empty
    result — this is what task_defaults() and rugiar's `--list_tasks`/
    `--list_reward_scales` rely on. Returns None only when mjlab isn't set
    up on this machine at all (no MJLAB_PYTHON) or the probe fails
    outright (treated as "can't validate", not "no mjlab tasks exist" —
    same reasoning as _mjlab_registered_tasks()/_legged_gym_registered_tasks()).

    Cached per-process (@lru_cache) — task/reward-scale defaults don't
    change during a single CLI invocation or driver session; tests that
    monkeypatch the probe must call `_mjlab_registry_snapshot.cache_clear()`
    first."""
    try:
        import mjlab_tasks  # noqa: F401 - import side effect: registers this repo's own tasks
        from mjlab.tasks import registry
    except ImportError:
        if not MJLAB_PYTHON.exists():
            return None
        env = dict(os.environ)
        _mjlab_prepare_env(env)
        # A real script FILE, not `python -c` -- see _mjlab_registry_probe_source()'s
        # docstring for why (-c would put REPO_ROOT on sys.path[0] via cwd,
        # reintroducing the R1 vendored-rsl_rl shadowing this whole dance avoids).
        try:
            with tempfile.NamedTemporaryFile(
                    "w", suffix="_mjlab_registry_probe.py", delete=False) as f:
                f.write(_mjlab_registry_probe_source())
                probe_path = f.name
            proc = subprocess.run(
                [str(MJLAB_PYTHON), probe_path],
                capture_output=True, text=True, timeout=90, env=env, cwd=str(REPO_ROOT),
            )
        except (subprocess.SubprocessError, OSError):
            return None
        finally:
            try:
                os.unlink(probe_path)
            except OSError:
                pass
        if proc.returncode != 0:
            return None
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None
    tasks = registry.list_tasks()
    if not tasks:
        return None
    out = {}
    for t in tasks:
        cfg = registry.load_env_cfg(t)
        out[t] = {
            "reward_scales": {name: term.weight for name, term in cfg.rewards.items()},
            "needs_motion_file": "motion" in getattr(cfg, "commands", {}),
        }
    return out


def training_backend_for_task(task: str) -> str:
    """'mjlab' | 'genesis' — which training entrypoint/interpreter `task` needs."""
    mjlab_tasks_ = _mjlab_registered_tasks()
    if mjlab_tasks_ is not None:                 # we ARE in .venv-mjlab: authoritative
        return "mjlab" if task in mjlab_tasks_ else "genesis"
    genesis_tasks = _legged_gym_registered_tasks()
    if genesis_tasks is not None:                # we ARE in .venv: authoritative by exclusion
        return "genesis" if task in genesis_tasks else "mjlab"
    raise ValueError(f"cannot determine a training backend for task '{task}': neither "
                     f"mjlab's nor legged_gym's task registry is importable here")


# ---- the training-backend registry ----
#
# WHAT THIS IS. A training backend is one answer to "where and how does a
# training job actually run": which interpreter, which entrypoint script,
# which subprocess env, which simulator ends up recorded on the resulting
# policy, and which tasks it can serve at all. There are three today —
# `local-genesis`, `local-mjlab`, `kaggle` — and TrainingManager.start()
# contains ZERO knowledge of any of them: it validates the request, resolves
# ONE descriptor out of BACKENDS, and drives that descriptor's hooks. This
# replaced a set of scattered `if backend == "kaggle" / if train_backend ==
# "mjlab"` branches that had to be edited in five places at once.
#
# HOW TO ADD A NEW BACKEND (e.g. a CUDA-enabled "local-nvidia", or a second
# cloud provider). Nothing in start()/poll()/the validation block changes —
# adding one is these three steps and no others:
#
#   1. Write the hooks it needs, next to the existing ones below:
#        - an interpreter resolver `(manager, task) -> str` (raise ValueError
#          with a human explanation if that venv isn't installed here);
#        - a `prepare_env(env)` that mutates the subprocess env in place
#          (PYTHONPATH/SIMULATOR/CUDA_VISIBLE_DEVICES/...);
#        - optionally a `validate_params(params)` that rejects knobs this
#          backend has no analogue for, and a `preflight()` for credentials
#          or other "can this even run here" checks.
#      A REMOTE backend (runs off this machine) skips interpreter/script/env
#      entirely and supplies `launch_remote(manager, job, ctx)` instead —
#      see kaggle's, which hands the job to a KaggleRunner thread.
#   2. Append one TrainingBackend(...) entry to BACKENDS, declaring which
#      `requested_as` value selects it and which `task_stack` it serves.
#      `(requested_as, task_stack)` is the lookup key and must be unique.
#   3. If it's requestable by a NEW name (not "local"/"kaggle"),
#      that name becomes valid automatically — REQUESTABLE_BACKENDS and
#      start()'s "unknown backend" message are both derived from BACKENDS.
#
# Deliberately NOT a plugin system: no dynamic imports, no config files, no
# entry points. A new backend is a Python literal in this list, which is
# exactly as much extensibility as three-to-five backends warrant.


def _genesis_interpreter(manager: "TrainingManager", task: str) -> str:
    """This process's own interpreter — unless it's the mjlab venv, which
    has no Genesis at all. Same 'switch venv, or refuse' shape as
    rugiar_driver_mjlab.py's family switch."""
    interpreter = manager.python_exe
    # Compared UNRESOLVED on purpose: a venv's bin/python is a symlink
    # to the base interpreter, so .resolve() throws away the very
    # ".venv-mjlab" marker this needs to see.
    if ".venv-mjlab" in str(interpreter):
        if not GENESIS_PYTHON.exists():
            raise ValueError(f"no Genesis venv at {GENESIS_PYTHON} — can't train "
                             f"task '{task}' from an mjlab session")
        interpreter = str(GENESIS_PYTHON)
    return interpreter


def _mjlab_interpreter(manager: "TrainingManager", task: str) -> str:
    """Always .venv-mjlab: neither venv can import the other's simulator
    (docs/mjlab_migration.md R1), so this is an interpreter choice, not just
    a script choice."""
    if not MJLAB_PYTHON.exists():
        raise ValueError(f"no mjlab venv at {MJLAB_PYTHON} — mjlab training isn't "
                         f"set up on this machine")
    return str(MJLAB_PYTHON)


def _genesis_prepare_env(env: Dict[str, str]) -> None:
    """Pin PYTHONPATH to THIS repo checkout explicitly rather than trusting
    whatever the parent process happened to be launched with — an editable
    `pip install -e` of legged_gym elsewhere (e.g. a sibling checkout of this
    same repo) would otherwise silently win, running web_train.py's file from
    here against a DIFFERENT legged_gym package. Bit us once already getting
    the control server itself to run against the right checkout — not
    leaving it to chance twice."""
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    # Explicit: an mjlab session's inherited SIMULATOR=mjlab would
    # otherwise make legged_gym/__init__.py skip the Genesis import.
    env["SIMULATOR"] = "genesis"


def _mjlab_prepare_env(env: Dict[str, str]) -> None:
    """The exact OPPOSITE of the Genesis case: REPO_ROOT must NOT be
    prepended here. The repo vendors a top-level rsl_rl/ that would shadow
    .venv-mjlab's PyPI rsl-rl-lib (docs/mjlab_migration.md R1). mjlab_train.py
    puts REPO_ROOT back on sys.path itself — LAST — so `mjlab_tasks`/
    `legged_gym` still resolve while rsl-rl-lib wins."""
    existing = env.get("PYTHONPATH", "")
    stripped = [p for p in existing.split(os.pathsep) if p and p != str(REPO_ROOT)]
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
_GENESIS_ONLY_PARAMS = (
    "cmd_vx", "cmd_vy", "cmd_yaw",
    "base_height_target", "lin_vel_z_target", "ang_vel_xy_target",
    "orientation_tilt_target",
    "push_robots", "max_push_vel_xy", "push_interval_s", "push_dir",
)


def _mjlab_validate_params(params: Dict[str, object]) -> None:
    inapplicable = [name for name in _GENESIS_ONLY_PARAMS if params.get(name) is not None]
    if inapplicable:
        raise ValueError(
            f"{', '.join(inapplicable)} don't apply to mjlab task '{params['task']}' "
            f"(motion-tracking task: no velocity command, no stability targets, "
            f"no pushes)")
    if not params.get("motion_file"):
        raise ValueError(f"task '{params['task']}' needs a --motion_file (reference-motion clip)")


def _kaggle_preflight() -> None:
    if not kaggle_backend.kaggle_credentials_available():
        raise ValueError(
            "no Kaggle credentials found at ~/.kaggle/kaggle.json — the Kaggle backend "
            "isn't set up on this machine")


def _kaggle_launch_remote(manager: "TrainingManager", job: "TrainingJob", ctx: dict) -> None:
    runner = kaggle_backend.KaggleRunner(
        job_id=job.id, train_flags=ctx["train_flags"],
        result_path=ctx["result_path"], log_path=ctx["log_path"],
        base_checkpoint_path=Path(ctx["from_checkpoint"]) if ctx["from_checkpoint"] else None,
    )
    # kaggle_kernel_slug stays None until poll() sees runner.kernel_ref
    # populated (set inside the background thread once its push succeeds).
    manager._kaggle_runners[job.id] = runner
    runner.start()


@dataclasses.dataclass(frozen=True)
class TrainingBackend:
    """One place a training job can run. See the registry's own comment
    above for the "how do I add one" walkthrough."""

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


BACKENDS: List[TrainingBackend] = [
    TrainingBackend(
        id="local-genesis",
        requested_as="local",
        task_stack="genesis",
        job_backend="local",
        simulator="genesis",
        command_prefix="rugiar train ",
        script=TRAIN_SCRIPT,
        fixed_flags=("--headless", "--cpu"),
        interpreter=_genesis_interpreter,
        prepare_env=_genesis_prepare_env,
    ),
    TrainingBackend(
        id="local-mjlab",
        requested_as="local",
        task_stack="mjlab",
        job_backend="local",
        simulator="mjlab",
        command_prefix="rugiar train ",
        script=MJLAB_TRAIN_SCRIPT,
        # No --cpu/--headless prefix: mjlab_train.py never opens a viewer
        # and takes --device instead (it defaults to cpu).
        fixed_flags=(),
        interpreter=_mjlab_interpreter,
        prepare_env=_mjlab_prepare_env,
        validate_params=_mjlab_validate_params,
    ),
    TrainingBackend(
        id="kaggle",
        requested_as="kaggle",
        # Genesis-only, and not by omission: the kernel bootstrap installs
        # Isaac Gym specifically (see kaggle_backend._build_kernel_script).
        task_stack="genesis",
        job_backend="kaggle",
        # Genesis's GPU JIT needs Volta+ (sm_70+) hardware Kaggle's free-tier
        # P100 (Pascal, sm_60) doesn't have — see TrainingJob.simulator.
        simulator="isaacgym",
        command_prefix="rugiar train --backend kaggle ",
        script=TRAIN_SCRIPT,
        remote=True,
        preflight=_kaggle_preflight,
        launch_remote=_kaggle_launch_remote,
        accepts_local_checkpoint=False,
        unsupported_task_stack_error=(
            "the Kaggle backend doesn't support mjlab tasks (its bootstrap is "
            "IsaacGym-specific)"),
    ),
]

# What start(backend=...) accepts, derived — never hand-maintained.
REQUESTABLE_BACKENDS = tuple(dict.fromkeys(b.requested_as for b in BACKENDS))


def resolve_training_backend(task: str, requested: str) -> TrainingBackend:
    """The one lookup: (what the caller asked for, what stack this task needs)
    -> the descriptor that serves it. Raises ValueError with the same
    messages start() used to raise inline — an unknown `requested` name, or a
    known one that can't serve this task's stack."""
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


def _backend_descriptor(job_backend: str, simulator: str) -> Optional[TrainingBackend]:
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


def backend_for_job(job: "TrainingJob") -> Optional[TrainingBackend]:
    """The descriptor a job already in flight was launched under, recovered
    from its own persisted (job_backend, simulator) pair rather than from a
    field stored on the job — so a TrainingJob written before this registry
    existed still resolves. None if nothing matches (an unknown simulator
    recorded by an older/newer build); callers fall back rather than raise."""
    return _backend_descriptor(job.backend, job.simulator)


def _history_entry_backend_id(entry: dict) -> Optional[str]:
    """Same (job_backend, simulator) -> descriptor resolution as
    backend_for_job(), applied to a raw history.json record instead of a live
    TrainingJob. `simulator` defaults to "genesis" for entries written before
    this field existed — every local job predating mjlab training WAS a
    Genesis run, so this is a correct backfill, not a guess. Returns the
    descriptor's `id` (e.g. "local-genesis", "local-mjlab", "kaggle") — the
    actual throughput-regime bucket estimate() groups by, NOT the raw
    "backend" field alone (which conflates local-genesis and local-mjlab,
    both persisted as backend="local")."""
    descriptor = _backend_descriptor(entry.get("backend", "local"), entry.get("simulator", "genesis"))
    return descriptor.id if descriptor is not None else None


# ---- turning a raw job .log into something a human (or the info popup) can read ----

_ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
_STAT_RES = {
    # "noise " is optional: this repo's vendored rsl_rl prints "Mean action
    # noise std:", while rsl-rl-lib 5.x (.venv-mjlab, used by mjlab_train.py)
    # prints "Mean action std:". Same number either way.
    "noise_std": re.compile(r"Mean action (?:noise )?std:\s*([-\d.]+)"),
    "reward": re.compile(r"Mean reward:\s*([-\d.]+)"),
    "episode_length": re.compile(r"Mean episode length:\s*([-\d.]+)"),
}
# Genesis: "Mean episode rew_<term>: <v>"   mjlab/rsl-rl 5.x: "Episode_Reward/<term>: <v>"
# (rsl-rl 5.x's Episode_Termination/* and Metrics/* lines correctly don't match.)
_TERM_RE = re.compile(r"(?:Mean episode rew_(\w+)|Episode_Reward/(\w+)):\s*([-\d.]+)")
SERIES_MAX_POINTS = 60  # downsample target — a training run can print thousands of
                         # iteration blocks; the popup's chart only needs enough
                         # points to see the trend, not every single one


def parse_training_log(log_path: str) -> dict:
    """rsl_rl's OnPolicyRunner prints one human-readable stats block per
    logged iteration (see the sample block this regex set is built from, in
    HANDOFF_stability_curriculum.md §1) — this is the ONLY place those
    numbers exist; nothing in this codebase stores them structured today.
    Rather than teach web_train.py to duplicate rsl_rl's own logging as
    structured JSON mid-run (real surgery, for a value only the info popup
    needs), this just re-reads the plain-text log after the fact. Best-
    effort: a missing/unreadable/log-format-mismatched file returns empty
    results rather than raising — a policy trained before this parser
    existed, or from a log that's since been cleaned up, should still show
    its meta.json fields, just without the chart.

    Returns {"series": [{"iteration", "noise_std", "reward",
    "episode_length", "rew_<term>": value, ...}, ...] (downsampled to
    SERIES_MAX_POINTS; the rew_* keys are whichever reward terms this run
    printed on EVERY sampled block — a term only added mid-run via a config
    change is dropped rather than drawing a line with gaps),
    "final": {"noise_std", "reward", "episode_length"} | None,
    "final_reward_terms": {<term>: value} | None} — the LAST fully-parsed
    block's per-reward-term breakdown (`Mean episode rew_*` lines), for
    "what is this policy actually optimized for" at a glance."""
    empty = {"series": [], "final": None, "final_reward_terms": None}
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except OSError:
        return empty

    records = []
    current = None
    for line in lines:
        m = _ITER_RE.search(line)
        if m:
            if current and current.get("_complete"):
                records.append(current)
            current = {"iteration": int(m.group(1)), "_complete": False, "terms": {}}
            continue
        if current is None:
            continue
        for key, pattern in _STAT_RES.items():
            m = pattern.search(line)
            if m:
                current[key] = float(m.group(1))
                # A block is "complete" once its three headline stats are in —
                # the per-term rew_* lines that follow are extra detail, not
                # required for the point to count.
                if all(k in current for k in _STAT_RES):
                    current["_complete"] = True
                break
        else:
            m = _TERM_RE.search(line)
            if m:
                current["terms"][m.group(1) or m.group(2)] = float(m.group(3))
    if current and current.get("_complete"):
        records.append(current)

    if not records:
        return empty

    step = max(1, len(records) // SERIES_MAX_POINTS)
    sampled = records[::step]
    if sampled[-1] is not records[-1]:
        sampled.append(records[-1])  # always keep the true final point

    # A term only counts as a chartable series if every sampled block has it —
    # one printed by rsl_rl only after a mid-run config change would otherwise
    # draw a polyline with holes in it.
    common_terms = set.intersection(*(set(r["terms"]) for r in sampled)) if sampled else set()
    series = [
        {"iteration": r["iteration"], "noise_std": r["noise_std"],
         "reward": r["reward"], "episode_length": r["episode_length"],
         **{f"rew_{term}": r["terms"][term] for term in common_terms}}
        for r in sampled
    ]
    last = records[-1]
    final = {"noise_std": last["noise_std"], "reward": last["reward"],
              "episode_length": last["episode_length"]}
    return {"series": series, "final": final, "final_reward_terms": last["terms"] or None}


@dataclasses.dataclass
class TrainingJob:
    id: str
    policy_name: str
    task: str
    command: str  # display string — exactly what the UI previewed, minus the interpreter path
    log_path: str
    result_path: str
    progress_path: str  # see web_train.py's --progress_path — overwritten mid-run, read by poll()
    started_at: float
    max_iterations: Optional[int]  # requested cap — None if only --max_minutes was given
    max_minutes: Optional[float]
    num_envs: int
    iterations_done: Optional[int] = None  # live while running (from progress_path), final on success
                                            # (from result.json, which then wins — see poll())
    status: str = "running"  # running | done | failed
    finished_at: Optional[float] = None
    error: Optional[str] = None
    base_policy: Optional[str] = None  # clone-from source name, if this job fine-tuned one
    entropy_coef: Optional[float] = None  # None = task default was used, not overridden
    reward_scale_overrides: Optional[Dict[str, float]] = None  # name -> overridden value,
                                                                 # only the terms actually changed
    policy_path: Optional[str] = None
    train_checkpoint_path: Optional[str] = None  # rsl_rl's raw model_N.pt for this run,
                                                  # if web_train.py found one — see poll()
    backend: str = "local"  # "local" | "kaggle" — see kaggle_backend.py's module docstring
                             # for why a Kaggle job's poll() branch never touches the network
    kaggle_kernel_slug: Optional[str] = None  # "<username>/<slug>" once push succeeds —
                                               # the UI's "view on Kaggle" link, kaggle jobs only
    motion_file: Optional[str] = None  # the exact --motion_file this job was launched with, for
                                        # mjlab/tracking tasks (see _mjlab_validate_params) —
                                        # persisted into the resulting policy's meta.json by
                                        # finalize_policy() so list_motions() can cross-reference
                                        # clip<->policy explicitly instead of the old name-substring
                                        # heuristic (see HANDOFF_mimic_motion_library_ux.md's Item 2
                                        # follow-up). None for any task with no motion command term.
    simulator: str = "genesis"  # "genesis" | "isaacgym" | "mjlab" — which Simulator backend actually
                                 # trained this policy (see legged_gym/simulator/). Kaggle jobs
                                 # use isaacgym: Genesis's GPU JIT needs Volta+ (sm_70+) hardware
                                 # Kaggle's free-tier P100 (Pascal, sm_60) doesn't have, while
                                 # Isaac Gym's PhysX GPU pipeline runs on Pascal fine (see
                                 # HANDOFF_kaggle_cloud_gpu.md). Recorded here (not derived from
                                 # `backend`) because it's what actually determines sim2sim risk
                                 # for a policy trained under it — surfaced in meta.json so the
                                 # UI can flag it (see finalize_policy()).
    job_type: str = "train"  # "train" | "distill" — lets poll()/finalize_policy()/the web UI's
                              # renderTrainingJobs() branch without a parallel job-tracking
                              # structure; a distill job reuses every other TrainingJob field
                              # (result_path/progress_path/policy_path/train_checkpoint_path all
                              # mean the same thing) except base_policy, which distillation has
                              # no use for — see teacher_policy below instead.
    teacher_policy: Optional[str] = None  # "distill" jobs only — the source policy name being
                                           # behavior-cloned (see TrainingManager.start_distillation())
    distill_method: Optional[str] = None  # "distill" jobs only — "behavior_cloning" | "dagger"
                                           # (distillation.DISTILL_METHODS key actually used)
    final_bc_loss: Optional[float] = None  # "distill" jobs only — web_distill.py's result.json
                                            # final_bc_loss, read back by poll() on success
    rollout_diagnostics: Optional[Dict[str, Dict[str, float]]] = None  # "distill" jobs only —
        # web_distill.py's result.json rollout_diagnostics (distillation.summarize_rollout()):
        # actual/commanded lin_vel/ang_vel coverage of the teacher rollout the student was
        # trained on — a narrow-range commanded_ang_vel_yaw here explains a clone that doesn't
        # turn like its teacher without needing to re-run/re-diagnose anything
    distill_loss_curve: Optional[List[dict]] = None  # "distill" jobs with method="dagger" only —
        # distillation.dagger_train()'s own loss_curve: one {"round","epoch","loss"} point per
        # (round, epoch) — None for "behavior_cloning" (nothing round-based to chart there)
    distill_round_diagnostics: Optional[List[dict]] = None  # "dagger" jobs only — one
        # {"round","beta","final_loss",**summarize_rollout(...)} entry per round, computed from
        # THAT round's own data — see dagger_train()'s docstring for why a single final_bc_loss
        # is misleading for dagger (loss trends UP round-over-round by design, not a regression)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "policy_name": self.policy_name,
            "task": self.task,
            "max_iterations": self.max_iterations,
            "max_minutes": self.max_minutes,
            "num_envs": self.num_envs,
            "iterations_done": self.iterations_done,
            "command": self.command,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
            "error": self.error,
            "log_path": self.log_path,
            "base_policy": self.base_policy,
            "entropy_coef": self.entropy_coef,
            "reward_scale_overrides": self.reward_scale_overrides,
            "backend": self.backend,
            "kaggle_kernel_slug": self.kaggle_kernel_slug,
            "simulator": self.simulator,
            "motion_file": self.motion_file,
            "job_type": self.job_type,
            "teacher_policy": self.teacher_policy,
            "distill_method": self.distill_method,
            "final_bc_loss": self.final_bc_loss,
            "rollout_diagnostics": self.rollout_diagnostics,
        }


class TrainingManager:
    def __init__(self, python_exe: str = sys.executable):
        self.python_exe = python_exe
        self.jobs: Dict[str, TrainingJob] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._log_files: Dict[str, "object"] = {}
        # Kaggle jobs have no local subprocess to poll — this is their
        # analog of _procs, one background thread per job (see
        # kaggle_backend.KaggleRunner's module docstring for why it's a
        # thread and not a call made straight from start()/poll()).
        self._kaggle_runners: Dict[str, kaggle_backend.KaggleRunner] = {}
        # name -> {"task": str, "checkpoint": Optional[str]} — every policy
        # currently known to be clonable from, seeded at boot from the
        # --policy specs and extended as new jobs complete. checkpoint is
        # None for policies with no known .pt on this machine (still shown,
        # just not usable as a --from_checkpoint base).
        self.policy_sources: Dict[str, dict] = {}
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._history: List[dict] = self._load_history()

    # ---- what this machine actually is (for the "System" panel + sizing) ----

    def system_info(self) -> dict:
        """No claims beyond what's directly measurable on THIS machine —
        the panel this feeds exists specifically so the user isn't guessing
        at what their hardware can do (see the conversation that asked for
        this). cuda/mps availability is informational only: every training
        job launched from this UI runs Genesis on CPU (see web_train.py's
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu) — cli.cpu defaults
        True), so a GPU being present doesn't currently change anything."""
        cpu_count = os.cpu_count() or 1
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        except Exception:  # noqa: BLE001 - torch import shouldn't be fatal to a status panel
            cuda_available = False
            mps_available = False
        return {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "cpu_brand": _cpu_brand(),
            "cpu_count": cpu_count,
            "ram_gb": round(_total_ram_bytes() / (1024 ** 3), 1) if _total_ram_bytes() else None,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            # Whether the Create Policy panel should even offer "Run on Kaggle" —
            # true once ~/.kaggle/kaggle.json exists (see kaggle_backend.py). Not a
            # guarantee start(backend="kaggle") will succeed (e.g. GPU quota could
            # still be exhausted), just enough to know the option is worth showing.
            "kaggle_available": kaggle_backend.kaggle_credentials_available(),
            "simulator": os.environ.get("SIMULATOR", "unknown"),
            "genesis_backend": os.environ.get("GENESIS_BACKEND", "cpu"),
            # Not a measurement — a starting-point heuristic (envs run
            # vectorized but not free; more than a few per core stops
            # scaling on CPU). Gets less relevant once real history exists;
            # estimate()/the UI prefer measured numbers when they're available.
            "suggested_num_envs": {"comfortable": max(4, cpu_count * 4), "upper": max(8, cpu_count * 16)},
            # NOT measured on THIS machine — Kaggle jobs run on Kaggle's own
            # infrastructure, not here, so this can never be a live-per-
            # request probe the way everything above is (there's no session
            # to query without spending a kernel). These numbers ARE real
            # measurements, though, not guesses: a dedicated diagnostic
            # kernel (multiprocessing.cpu_count(), /proc/meminfo, nvidia-smi,
            # torch.cuda.get_device_properties) was run specifically to
            # capture them — see HANDOFF_kaggle_cloud_gpu.md. Kaggle can
            # still hand out something different next session (this is a
            # free tier, not a reserved instance), which is why the Hardware
            # panel labels this "measured once", not "live".
            "kaggle_profile": {
                "gpu": "Tesla P100-PCIE-16GB",
                "compute_capability": "6.0 (Pascal)",
                "vram_gb": 15.89,
                "cpu_cores": 4,
                "ram_gb": 31.3,
                "simulator": "isaacgym",
                "bootstrap_overhead_s": 180,
                "session_cap_hours": 12,
                # Real, not a guess: a dedicated kernel created the g1 task's
                # IsaacGymSimulator env at num_envs=4096 on this exact P100
                # and it succeeded (no OOM, no crash) — confirmed alongside
                # general community guidance for a 16GB GPU on non-vision
                # humanoid tasks (~2048-4096 is the typical range; see
                # HANDOFF_kaggle_cloud_gpu.md). "comfortable" is set below
                # the confirmed-working ceiling, same conservative spirit as
                # the local suggestion, not AT the edge of what's proven to
                # still work.
                "suggested_num_envs": {"comfortable": 1024, "upper": 4096},
            },
        }

    # ---- timing history (persisted so estimates survive a server restart) ----

    def _load_history(self) -> List[dict]:
        try:
            with open(HISTORY_PATH) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_history(self) -> None:
        try:
            with open(HISTORY_PATH, 'w') as f:
                json.dump(self._history[-200:], f)  # unbounded growth guard — 200 runs is plenty of signal
        except OSError:
            pass  # best-effort — a failed write must not crash the sim loop

    def estimate(self, num_envs: int, max_iterations: Optional[int] = None,
                 max_minutes: Optional[float] = None, backend: str = "local",
                 task: Optional[str] = None) -> dict:
        """Estimated (iterations, seconds) for a job on the given backend,
        from THAT backend's own completed-job history — pooled across tasks
        that share the same underlying simulator/hardware regime (dominated
        by robot/obs/action space size and sim engine, not which reward
        function is being trained, so cost-per-iteration is comparable
        across tasks on the SAME simulator), but never pooled across
        different regimes: a local-CPU Genesis run, a local-CPU mjlab run,
        and a Kaggle Isaac-Gym-GPU run are three different throughput
        regimes entirely (real numbers: local Genesis CPU ran single-digit
        iterations/sec on g1; local mjlab CPU measured ~1.27s/iteration at
        num_envs=8; the Kaggle GPU smoke test did 5 iterations with 16 envs
        in ~254s wall-clock including Isaac Gym's own ~3-4min per-job
        bootstrap) — mixing them would corrupt every estimate.

        `task` is what disambiguates "local" into the right regime: both
        local-genesis and local-mjlab persist history entries with
        backend="local" (see TrainingBackend.job_backend's docstring), so
        the SIMULATOR each entry actually ran under (recorded alongside
        elapsed_s/num_envs — see poll()) is what separates them, resolved
        the same way start() resolves which backend serves a task
        (resolve_training_backend()). Passing no `task` (or one that can't
        be resolved for this backend) falls back to the legacy raw-backend
        filter — every history entry with this literal `backend` string,
        regardless of simulator — which is what every caller did before
        this distinction existed; new callers should always pass `task`.
        This generalizes forward with zero code changes here: a future
        backend (local-NVIDIA, a second cloud) gets its own BACKENDS entry
        with its own `id`/`simulator`, its jobs' history entries carry that
        `simulator` automatically (see poll()), and `_history_entry_backend_id()`
        buckets them separately the moment the first one finishes — see
        tests/test_training_estimate.py's synthetic-backend test.

        Works with either or both of max_iterations/max_minutes, mirroring
        the actual job's own 'whichever hits first' semantics (see
        web_train.py's chunked learn() loop) — if both are given, whichever
        resolves to fewer seconds wins. This is always an estimate, not a
        promise: per-iteration cost varies with machine load (and, for
        Kaggle, with whatever GPU that session happens to get), so a
        wall-clock budget may stop a run a bit short of or past the
        iteration count shown here — it still stops on time; the iteration
        count just moves. Returns basis='none' (no invented number) when
        there's no history yet for this backend/simulator regime — see
        system_info()'s suggested_num_envs for a sizing starting point in
        that case. Deliberately never falls back to a hardcoded constant or
        another regime's numbers — an mjlab job with zero local history
        must show basis='none', not Genesis's rate."""
        num_envs = max(1, int(num_envs))
        none_result = {"basis": "none", "samples": 0, "seconds": None, "iterations": None}
        target_id = None
        if task is not None:
            try:
                target_id = resolve_training_backend(task, backend).id
            except ValueError:
                target_id = None  # unresolvable (task) -> fall back to the legacy raw filter below
        if target_id is not None:
            backend_history = [h for h in self._history if _history_entry_backend_id(h) == target_id]
        else:
            backend_history = [h for h in self._history if h.get("backend", "local") == backend]
        if not backend_history:
            return none_result
        rates = [h["elapsed_s"] / (h["max_iterations"] * h["num_envs"])
                 for h in backend_history if h["max_iterations"] > 0 and h["num_envs"] > 0]
        if not rates:
            return none_result
        rates.sort()
        median_rate = rates[len(rates) // 2]  # seconds per (iteration * env)

        candidates = []  # (seconds, iterations)
        if max_iterations:
            it = max(1, int(max_iterations))
            candidates.append((median_rate * it * num_envs, it))
        if max_minutes:
            budget_s = max(1.0, float(max_minutes) * 60.0)
            it = max(1, int(budget_s / (median_rate * num_envs)))
            candidates.append((budget_s, it))
        if not candidates:
            return none_result
        seconds, iterations = min(candidates, key=lambda c: c[0])
        return {"basis": "measured", "samples": len(rates), "seconds": round(seconds, 1), "iterations": iterations}

    # ---- catalog the web UI's form renders from ----

    # Every variable the Create Policy panel's 4 always-visible target
    # fields can offer, in one place — add an entry here (plus a matching
    # cfg.rewards scalar target field the existing tracking reward already
    # reads) and it shows up as a 5th field-row IF a matching block is also
    # added to web/index.html (unlike the old single-dropdown design, the
    # UI doesn't grow a field automatically from a registry entry alone —
    # see the plan's "Explicitly NOT in scope" note). 'flag' is the exact
    # web_train.py CLI arg the resolved number is sent as — both the
    # Absolute and '% change' fields funnel through the SAME flag (see
    # app.js's resolveTarget()).
    VARIABLE_REGISTRY = {
        "base_height": {
            "label": "Base height",
            "unit": "m",
            "source": "sim_ground_truth",
            "flag": "base_height_target",
            "config_attr": "base_height_target",
            "note": "Not measured by any real sensor — see RobotState.base_height's docstring. "
                    "Fine as a training target since training only ever runs in sim.",
        },
        "lin_vel_z": {
            "label": "Vertical velocity",
            "unit": "m/s",
            "source": "sim_ground_truth",
            "flag": "lin_vel_z_target",
            "config_attr": "lin_vel_z_target",
            "note": "Not measured by any real sensor (no IMU measures velocity, only acceleration) — "
                    "see RobotState.base_lin_vel's docstring. Fine as a training target since training "
                    "only ever runs in sim. Defaults to 0 — no vertical bobbing.",
        },
        "ang_vel_xy": {
            "label": "Roll/pitch rate",
            "unit": "rad/s",
            "source": "sensor",
            "flag": "ang_vel_xy_target",
            "config_attr": "ang_vel_xy_target",
            "note": "Real IMU gyroscope signal — same one Live Telemetry's 'Angular velocity' shows. "
                    "Defaults to 0 — no roll/pitch wobble.",
        },
        "orientation_tilt": {
            "label": "Body tilt",
            "unit": "g",
            "source": "sensor",
            "flag": "orientation_tilt_target",
            "config_attr": "orientation_tilt_target",
            "note": "Real IMU signal (gravity vector in the body frame) — same one Live Telemetry's "
                    "'Orientation' shows. Defaults to 0 — perfectly upright.",
        },
    }

    # Short explanations for reward-scale terms in the raw "Reward weights
    # (advanced)" grid (renderRewardScaleFields() in app.js) that aren't
    # self-explanatory from their name alone — NOT a promotion to
    # VARIABLE_REGISTRY above. These are reward WEIGHTS (how much a term is
    # pushed), not target VARIABLES (what value a term converges to), so they
    # stay in the raw grid; this dict only adds the missing context (see
    # HANDOFF_task_reward_harmony.md §3/§5 step 2). Add an entry here whenever
    # a reward term's purpose isn't obvious from `TRACKING_LIN_VEL`-style
    # all-caps rendering alone — most don't need one.
    REWARD_SCALE_NOTES = {}

    # One-line reason each registered task exists as a TASK rather than a UI override on
    # its robot's base task — i.e. what's structural about it (new reward term, obs/action
    # space, termination condition, or training architecture), per the rule in
    # HANDOFF_control_web.md §5b. Populated from the audit in
    # HANDOFF_task_reward_harmony.md §4a. A task's own base (g1, go2, k1, tron1pf, tron1sf)
    # doesn't need an entry — it's the default, nothing to explain relative to itself.
    TASK_NOTES = {
        "k1_deepmimic": "Motion-imitation architecture: frame-stacked observations, 22 actions, its own PPO setup.",
        "k1_motion_vis": "Visualization only, not for training — a separate env class with no training loop.",
        "k1_amp": "Adversarial Motion Prior: adds a discriminator/replay-buffer training pipeline, not just reward weights.",
        "k1_cts_amp": "Concurrent teacher-student + AMP (unvalidated): splits envs into teacher/student, different privileged-obs shape.",
        "g1_deepmimic": "Same motion-imitation architecture as k1_deepmimic, 29 DoF.",
        "g1_motion_vis": "Visualization only, not for training — same pattern as k1_motion_vis.",
        "go2_wtw": "Walk-These-Ways: behavior-parameter resampling (foot clearance/pitch/height) built into the env, different obs shape.",
        "go2_ts": "Teacher-student architecture: privileged/history observations and an encoder, not just reward weights.",
        "go2_ee": "Explicit-estimator architecture: different critic-observation shape.",
        "go2_cts": "Concurrent teacher-student: splits envs into teacher/student groups.",
        "go2_dreamwaq": "DreamWaQ architecture: its own decoder output and observation shape.",
        "go2_cat": "Constraint-as-termination: adds a new termination condition on top of go2_ts, not just reward weights.",
        "go2_ts_depth": "Teacher-student + depth camera (unvalidated): adds camera observations, much larger privileged-obs.",
        "go2_nav": "Navigation task: observations include a heightmap, different reward/termination structure.",
        "tron1pf_ee": "Explicit-estimator architecture (same pattern as go2_ee): different critic-observation shape.",
    }

    @staticmethod
    def _train_checkpoint_from_export(export_path: Optional[str]) -> Optional[str]:
        """`checkpoint` (export_policy()'s output, e.g. `<log_dir>/exported/
        policy_lstm_1.pt`) is a deployable TorchScript/ONNX artifact — the
        right thing to hot-load into the live supervisor, and exactly the
        WRONG thing to pass to `ppo_runner.load()`/--from_checkpoint, which
        needs rsl_rl's own raw format (weights + shapes for resuming
        training, saved as `<log_dir>/model_<iter>.pt` by
        OnPolicyRunner.learn() — see on_policy_runner.py). Passing the
        exported file there raises NotImplementedError deep in torch's
        jit loader — confusing, and it happened for real the first time
        this UI's 'Clone from' was used.

        This derives the raw checkpoint from the exported one by the
        directory convention this whole repo already uses everywhere else
        (web_train.py's own `export_dir = os.path.join(log_dir, 'exported')`):
        walk up one level from `exported/`, take the highest-iteration
        `model_*.pt` sibling. Returns None if that convention doesn't hold
        (e.g. `stable`'s checkpoint is a completely separate, external
        unitree_rl_gym clone with no local training history at all — see
        HANDOFF_control_web.md's policy table) — those sources correctly
        stay un-fine-tunable rather than silently guessing.

        Checked BEFORE that guess: the self-contained `policies/<name>/`
        folder convention finalize_policy() itself writes — train_checkpoint.pt
        sits right next to checkpoint.pt, no guessing needed at all. This
        matters for a policy passed via rugiar_driver.py's `--policy
        name:policies/<name>/checkpoint.pt` (rather than picked up through
        discover_local_policies(), which already checks this directly) —
        without this check, launching a self-contained policy this way left
        it permanently un-fine-tunable/un-fusable despite its
        train_checkpoint.pt being right there on disk."""
        if not export_path:
            return None
        sibling = os.path.join(os.path.dirname(export_path), "train_checkpoint.pt")
        if os.path.isfile(sibling):
            return sibling
        log_dir = os.path.dirname(os.path.dirname(export_path))
        if not os.path.isdir(log_dir):
            return None
        try:
            candidates = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        except OSError:
            return None
        if not candidates:
            return None

        def _iter_num(fname: str) -> int:
            try:
                return int(fname[len("model_"):-len(".pt")])
            except ValueError:
                return -1

        candidates.sort(key=_iter_num)
        return os.path.join(log_dir, candidates[-1])

    def register_source(self, name: str, task: str, checkpoint: Optional[str],
                         train_checkpoint: Optional[str] = None,
                         simulator: str = "genesis",
                         category: Optional[str] = None,
                         motion_file: Optional[str] = None) -> None:
        """`train_checkpoint` is the raw rsl_rl checkpoint to resume PPO
        from (see finalize_policy()'s docstring for how a fresh training
        job gets one). Pass None (the --policy CLI path, via
        rugiar_driver.py) to fall back to guessing it from `checkpoint`'s
        directory layout — the only option for a checkpoint that was never
        produced by this UI in the first place (e.g. an externally-sourced
        one with no raw training history at all, which correctly stays
        un-fine-tunable either way).

        `simulator` ("genesis" | "isaacgym") is which Simulator backend
        actually trained this policy — see TrainingJob.simulator's own
        docstring for why this matters: a policy fine-tuned across
        simulators (e.g. cloning an isaacgym-trained base into a local
        Genesis run, or vice versa) is a sim2sim transfer, not a guaranteed-
        compatible continuation. Surfaced in catalog() so the Create Policy
        panel can flag a mismatch instead of silently fine-tuning across
        engines.

        `category` is a free-form, purely-cosmetic label (e.g. "g1-legs",
        "g1-full-body", "go2") — NOT used for any obs/action-space
        compatibility check (that's `task`'s job). It exists so the Family/
        Policy panels can group/label sources that share a `task` but come
        from meaningfully different places — e.g. an externally-imported
        full-body G1 policy vs one this repo trained itself under the same
        `g1_deepmimic` task. None (the default) means "no opinion" — the UI
        falls back to showing `task` alone, same as before this field
        existed."""
        self.policy_sources[name] = {
            "task": task, "checkpoint": checkpoint,
            "train_checkpoint": train_checkpoint or self._train_checkpoint_from_export(checkpoint),
            "simulator": simulator,
            "category": category,
            "motion_file": motion_file,
        }

    def finalize_policy(self, name: str, task: str, checkpoint: str,
                         train_checkpoint: Optional[str],
                         job: Optional["TrainingJob"] = None) -> str:
        """Called once a training job finishes (see rugiar_driver.py's
        drain_finished_training()) to copy its two checkpoints out of
        rsl_rl's log_dir — which is logging/TensorBoard scratch space, not
        somewhere a policy is meant to live long-term — into one
        self-contained `policies/<name>/` folder: `checkpoint.pt` (the
        deployable export, hot-loaded into the supervisor) alongside
        `train_checkpoint.pt` (the raw PPO state, fine-tunable via
        Clone-from), `train.log` (the job's own log, copied so the run's
        full history survives even if `logs/_web_training/` is ever
        cleaned up), and a `meta.json`. This is what makes Clone-from for
        anything trained through this UI *always* work — no more walking
        back from an exported path guessing whether its log_dir still
        exists (see _train_checkpoint_from_export()'s docstring for the bad
        old way, still used as a fallback for --policy-supplied checkpoints
        this UI never produced). Copies rather than moves, so the original
        log_dir (TensorBoard events, every intermediate model_N.pt) is
        untouched. Returns the new checkpoint path — load THAT into the
        supervisor, not the original export, so what's running matches
        what's registered.

        `job` is the finished TrainingJob, when the caller has one (every
        real caller does — see drain_finished_training()) — it's what lets
        meta.json carry the exact command that produced this policy, what
        it was cloned from, the entropy_coef used, and (via
        parse_training_log()) how its `Mean action noise std` / `Mean
        reward` / `Mean episode length` actually moved over the run. This
        is the ONE place that data gets captured — the info popup reads it
        straight back out of meta.json rather than re-parsing anything.
        Passing None (or a checkpoint this UI didn't itself launch) just
        means a leaner meta.json — see policy_info()'s fallback for
        policies with no job at all, e.g. `stable`."""
        dest_dir = POLICIES_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Keep the source's suffix: load_policy_backend() dispatches on it, so
        # copying an mjlab job's .onnx export to "checkpoint.pt" would make it
        # unloadable. discover_local_policies() already accepts either name.
        suffix = ".onnx" if str(checkpoint).endswith(".onnx") else ".pt"
        dest_checkpoint = dest_dir / f"checkpoint{suffix}"
        shutil.copyfile(checkpoint, dest_checkpoint)
        dest_train_checkpoint = None
        if train_checkpoint and os.path.isfile(train_checkpoint):
            dest_train_checkpoint = dest_dir / "train_checkpoint.pt"
            shutil.copyfile(train_checkpoint, dest_train_checkpoint)

        meta = {"task": task, "created_at": time.time(),
                "trained_via": "distillation" if job is not None and job.job_type == "distill" else "control web",
                "simulator": job.simulator if job is not None else "genesis"}
        if job is not None and job.job_type == "distill":
            # Distillation has no PPO learning-iteration log to chart
            # (parse_training_log()'s regexes are all rsl_rl-specific) — its
            # own result.json fields (read back via job.command/started_at/
            # finished_at, same as a normal run) are what's meaningful here.
            dest_log = None
            if os.path.isfile(job.log_path):
                dest_log = dest_dir / "train.log"
                shutil.copyfile(job.log_path, dest_log)
            meta.update({
                "command": job.command,
                "num_envs": job.num_envs,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "elapsed_s": round((job.finished_at or time.time()) - job.started_at, 1),
                "log_path": str(dest_log) if dest_log else None,
                "distillation": {
                    "method": job.distill_method or "behavior_cloning",
                    "teacher": job.teacher_policy,
                    "bc_epochs": job.max_iterations,
                    "final_bc_loss": job.final_bc_loss,
                    "rollout_diagnostics": job.rollout_diagnostics,
                    # dagger only — None for behavior_cloning, see dagger_train()'s docstring
                    # on why the round trend matters more than the bare final_bc_loss above.
                    "loss_curve": job.distill_loss_curve,
                    "round_diagnostics": job.distill_round_diagnostics,
                },
            })
        elif job is not None:
            metrics = parse_training_log(job.log_path)
            dest_log = None
            if os.path.isfile(job.log_path):
                dest_log = dest_dir / "train.log"
                shutil.copyfile(job.log_path, dest_log)
            meta.update({
                "command": job.command,
                "base_policy": job.base_policy,
                "entropy_coef": job.entropy_coef,
                "reward_scale_overrides": job.reward_scale_overrides,
                "num_envs": job.num_envs,
                "max_iterations": job.max_iterations,
                "max_minutes": job.max_minutes,
                "iterations_done": job.iterations_done,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "elapsed_s": round((job.finished_at or time.time()) - job.started_at, 1),
                "source_log_dir": os.path.dirname(os.path.dirname(checkpoint))
                    if os.path.basename(os.path.dirname(checkpoint)) == "exported" else None,
                "log_path": str(dest_log) if dest_log else None,
                "metrics": metrics,
                # Explicit clip<->policy link — None for any task with no
                # motion command term (e.g. Genesis locomotion tasks).
                # list_motions() reads this directly instead of guessing
                # from a name-substring match; see its own docstring for the
                # fallback that still applies to policies finalized before
                # this field existed.
                "motion_file": job.motion_file,
            })
        with open(dest_dir / "meta.json", "w") as f:
            json.dump(meta, f)
        self.register_source(
            name, task=task, checkpoint=str(dest_checkpoint),
            train_checkpoint=str(dest_train_checkpoint) if dest_train_checkpoint else None,
            simulator=meta["simulator"],
        )
        return str(dest_checkpoint)

    def fuse_policies(self, names: Sequence[str], out_name: str,
                       weights: Optional[Sequence[float]] = None,
                       method: str = "weighted_average",
                       export_task: Optional[str] = None) -> dict:
        """Merges 2+ already-trained policies' weights into one new local
        policy — no further training involved. See legged_gym/control/
        fusion.py for the actual tensor math/module (re)construction; this
        method is the disk/registration layer on top of it, mirroring
        finalize_policy()'s own `policies/<out_name>/` conventions
        (checkpoint.pt + train_checkpoint.pt + meta.json,
        register_source()) so a fused policy is indistinguishable, to every
        other caller, from a normally-trained one — fine-tunable via
        --from_policy, and fusable again.

        Each of `names` must already be in self.policy_sources (same
        precondition run_train()/run_order() already establish via
        discover_local_policies()+register_source() before calling in) and
        have a train_checkpoint — the raw rsl_rl weights, not just an
        exported TorchScript module, same restriction --from_policy has.

        Sources must be architecturally compatible (matching obs/action/
        hidden dims and recurrent-or-not — see fusion.architectures_
        compatible()) regardless of what task each was trained on; a
        mismatched TASK across sources is only a warning (returned, not
        raised) since two different tasks can share an identical network
        shape and still be a perfectly reasonable thing to try merging.
        `export_task` picks which task the fused result gets registered
        under (defaults to the first source's) — this determines the
        activation function used to rebuild the network (not recoverable
        from a state_dict) and is cross-checked against the merged weights'
        own obs/action dims, which IS a hard error: a mismatched task here
        would silently break ObsSpec for whoever loads this policy later
        (see service.py's refresh_local_policies()).

        Returns {"name", "checkpoint_path", "warnings": [str, ...]}."""
        from legged_gym.control import fusion

        method_info = fusion.FUSION_METHODS.get(method)
        if method_info is None:
            raise ValueError(f"unknown fusion method '{method}' — see fusion.FUSION_METHODS")
        if not method_info["available"]:
            raise ValueError(f"fusion method '{method}' is planned but not yet implemented")
        if len(names) < 2:
            raise ValueError("need at least 2 policies to fuse")
        if (POLICIES_DIR / out_name).exists():
            raise ValueError(f"'{out_name}' already exists — pick a different output name")

        sources = []
        for name in names:
            info = self.policy_sources.get(name)
            if info is None:
                raise ValueError(f"'{name}' is not a known local policy")
            if not info.get("train_checkpoint"):
                raise ValueError(f"'{name}' has no train_checkpoint.pt — not fine-tunable, so not fusable either")
            sources.append({"name": name, **info})

        if weights is None:
            weights = [1.0] * len(sources)
        elif len(weights) != len(sources):
            raise ValueError(f"got {len(weights)} weight(s) for {len(sources)} polic(y/ies)")

        import torch
        state_dicts = []
        for src in sources:
            ck = torch.load(src["train_checkpoint"], map_location="cpu", weights_only=False)
            state_dicts.append(ck["model_state_dict"])

        archs = [fusion.infer_architecture(sd) for sd in state_dicts]
        for i, arch in enumerate(archs[1:], start=1):
            mismatch = fusion.architectures_compatible(archs[0], arch)
            if mismatch is not None:
                raise ValueError(
                    f"'{sources[0]['name']}' and '{sources[i]['name']}' aren't fusable: {mismatch}")

        warnings: List[str] = []
        tasks = sorted({src["task"] for src in sources})
        if export_task is None:
            export_task = sources[0]["task"]
        if len(tasks) > 1:
            warnings.append(
                f"sources were trained on different tasks ({', '.join(tasks)}) — registering the "
                f"fused policy under '{export_task}'; this only affects which reward/config reference "
                f"values it's shown against, not the merged weights themselves")
        simulators = sorted({src.get("simulator", "genesis") for src in sources})
        if len(simulators) > 1:
            warnings.append(f"sources were trained on different simulators ({', '.join(simulators)})")

        from legged_gym.utils import task_registry
        env_cfg, train_cfg = task_registry.get_cfgs(name=export_task)
        if env_cfg.env.num_observations != archs[0]["num_actor_obs"] or \
           env_cfg.env.num_actions != archs[0]["num_actions"]:
            raise ValueError(
                f"'{export_task}' expects {env_cfg.env.num_observations} obs / "
                f"{env_cfg.env.num_actions} actions, but the sources have "
                f"{archs[0]['num_actor_obs']} obs / {archs[0]['num_actions']} actions — "
                f"pick a different --export_task")
        # Export needs a *string* load_run to build its output filename (see
        # PolicyExporter.export() in helpers.py) — left at its int sentinel default,
        # a fresh task_registry.get_cfgs() config was never resolved against a real
        # run directory the way an actual training/play invocation resolves it.
        train_cfg.runner.load_run = "fusion"

        if method == "git_rebasin":
            # Align every non-reference source's hidden units to the first source
            # before averaging — see fusion.rebasin_align()'s docstring for why this
            # (rather than plain elementwise averaging) is the whole point of the method.
            state_dicts = [state_dicts[0]] + [
                fusion.rebasin_align(state_dicts[0], sd) for sd in state_dicts[1:]]

        merged = fusion.merge_state_dicts(state_dicts, list(weights))
        actor_critic = fusion.build_actor_critic(archs[0], merged, activation=train_cfg.policy.activation)

        task_type = "_".join(export_task.split("_")[1:])
        dest_dir = POLICIES_DIR / out_name
        try:
            with tempfile.TemporaryDirectory() as tmp:
                exported = fusion.export_actor_critic(actor_critic, tmp, env_cfg, train_cfg, task_type)

                dest_dir.mkdir(parents=True)
                dest_checkpoint = dest_dir / "checkpoint.pt"
                shutil.copyfile(exported, dest_checkpoint)

            dest_train_checkpoint = dest_dir / "train_checkpoint.pt"
            torch.save({"model_state_dict": merged, "optimizer_state_dict": {}, "iter": 0, "infos": {}},
                       dest_train_checkpoint)

            meta = {
                "task": export_task, "created_at": time.time(), "trained_via": "fusion",
                "simulator": sources[0].get("simulator", "genesis"),
                "fusion": {
                    "method": method,
                    "sources": [{"name": s["name"], "task": s["task"], "weight": w}
                                for s, w in zip(sources, weights)],
                    "warnings": warnings,
                },
            }
            with open(dest_dir / "meta.json", "w") as f:
                json.dump(meta, f)
        except Exception:
            # Partial writes must not leave a half-built policies/<out_name>/ behind —
            # that would block every retry with this name with a confusing "already
            # exists" instead of surfacing the real error that just occurred.
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise

        self.register_source(
            out_name, task=export_task, checkpoint=str(dest_checkpoint),
            train_checkpoint=str(dest_train_checkpoint), simulator=meta["simulator"],
        )
        return {"name": out_name, "checkpoint_path": str(dest_checkpoint), "warnings": warnings}

    def discover_local_policies(self, exclude: Sequence[str] = ()) -> Dict[str, dict]:
        """Every self-contained `policies/<name>/` folder on disk that
        finalize_policy() ever wrote — i.e. everything trained through this
        UI — regardless of whether it's still loaded in THIS process's
        memory. `policy_sources`/the running supervisor only know about
        whatever was passed via --policy at launch plus whatever's
        completed *in this process's lifetime*; restarting the server (to
        pick up new code, or after a crash) starts both of those empty
        again even though nothing on disk changed — finalize_policy()
        copies checkpoints out of scratch log_dir space specifically so
        they'd survive that. This is what lets rugiar_driver.py's
        startup re-offer every previously-trained policy instead of just
        whatever --policy flags happened to be typed that time.

        Returns name -> {"task", "checkpoint", "train_checkpoint", "simulator",
        "category", "motion_file"} for every folder with a checkpoint.pt (or
        checkpoint.onnx — see below),
        skipping names in `exclude` (already loaded a different way, e.g.
        via --policy) and skipping (with nothing raised — this must never
        crash startup) anything without a readable meta.json giving its
        task, since loading a checkpoint from the wrong task/observation-
        space would crash load_policy() rather than just fail to appear.

        checkpoint.onnx (in addition to checkpoint.pt) is recognized so an
        externally-sourced ONNX export (policy.py's load_policy_backend()
        already dispatches on the .onnx suffix — see OnnxStatelessPolicy/
        OnnxExplicitStatePolicy) can live in the same policies/<name>/
        folder convention as a jit one, instead of only being loadable via
        an ad-hoc --policy path. Prefers .pt if a folder somehow has both."""
        found = {}
        if not POLICIES_DIR.is_dir():
            return found
        for entry in sorted(POLICIES_DIR.iterdir()):
            name = entry.name
            if name in exclude or not entry.is_dir():
                continue
            checkpoint = entry / "checkpoint.pt"
            if not checkpoint.is_file():
                checkpoint = entry / "checkpoint.onnx"
            if not checkpoint.is_file():
                continue
            try:
                with open(entry / "meta.json") as f:
                    meta = json.load(f)
                task = meta["task"]
            except (OSError, KeyError, json.JSONDecodeError):
                continue
            train_checkpoint = entry / "train_checkpoint.pt"
            found[name] = {
                "task": task,
                "checkpoint": str(checkpoint),
                "train_checkpoint": str(train_checkpoint) if train_checkpoint.is_file() else None,
                "simulator": meta.get("simulator", "genesis"),
                "category": meta.get("category"),
                # The exact clip this policy was trained against, if any —
                # None for anything trained before finalize_policy() started
                # recording it (or any non-motion task). list_motions() reads
                # this to cross-reference clip<->policy explicitly.
                "motion_file": meta.get("motion_file"),
            }
        return found

    def get_policy_order(self) -> List[str]:
        """The configured display order for local policies — what a
        downstream control-layer app (a separate process/UI that only
        SELECTS among policies this tool creates, never trains them) should
        read to decide what to offer first, instead of inventing its own
        opinion about naming or recency. Any local policy not yet mentioned
        in .policy_order.json floats to the end, alphabetically, so a freshly
        trained policy always shows up somewhere without needing an explicit
        set_policy_order() call first. Names in the file that no longer
        exist on disk are silently dropped. Never raises — a missing or
        corrupt order file just means "no preference yet" (empty list)."""
        known = sorted(self.discover_local_policies().keys())
        try:
            with open(POLICY_ORDER_PATH) as f:
                configured = json.load(f)
        except (OSError, json.JSONDecodeError):
            configured = []
        ordered = [name for name in configured if name in known]
        ordered += [name for name in known if name not in ordered]
        return ordered

    def set_policy_order(self, names: Sequence[str]) -> None:
        """Persists the display order for local policies to
        POLICY_ORDER_PATH, read back by get_policy_order(). `names` need not
        list every local policy — anything omitted keeps floating to the end
        alphabetically via get_policy_order(), so this only needs to name
        the ones you actually want to pin/reorder. Every given name MUST
        already be a local policy (checked against discover_local_policies())
        — this catches a typo'd name up front instead of it silently having
        no effect."""
        known = set(self.discover_local_policies().keys())
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(
                f"unknown local polic{'y' if len(unknown) == 1 else 'ies'}: "
                f"{', '.join(unknown)} — see discover_local_policies()"
            )
        POLICIES_DIR.mkdir(parents=True, exist_ok=True)
        with open(POLICY_ORDER_PATH, "w") as f:
            json.dump(list(names), f, indent=2)

    def policy_info(self, name: str) -> dict:
        """Everything the info popup shows for one policy — a light read of
        `policies/<name>/meta.json` plus the file-existence facts a popup
        needs to gray out a Clone-from-only action (e.g. no
        train_checkpoint.pt). Deliberately does NOT re-parse the log on
        every call — finalize_policy() already did that once and baked the
        result into meta.json, so this stays cheap enough to call from a
        button click. Raises FileNotFoundError for a name with no
        policies/<name>/ folder at all (nothing to show — e.g. a policy
        registered from a bare --policy CLI path with no self-contained
        folder, like `stable`); the caller/UI is expected to handle that as
        'no extra info available' rather than an error message."""
        meta_path = POLICIES_DIR / name / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        dest_dir = POLICIES_DIR / name
        meta["name"] = name
        meta["has_train_checkpoint"] = (dest_dir / "train_checkpoint.pt").is_file()
        meta["checkpoint_path"] = str(dest_dir / "checkpoint.pt")
        return meta

    def forget_source(self, name: str) -> None:
        """Drops a policy from the clone-from catalog and deletes its files
        on disk — the counterpart to register_source()/finalize_policy(),
        for discarding a training experiment that didn't work out (see
        ControlService.delete_policy()). For a policy finalize_policy()
        created, `checkpoint` lives in its own dedicated `policies/<name>/`
        folder with nothing else in it, so the whole folder (checkpoint +
        train_checkpoint + meta.json) is removed. For anything registered
        the old way — a bare --policy CLI path — only that one file is
        removed, same as before; there's no dedicated folder to reach into,
        and no other file to guess about. Best-effort — a source with no
        checkpoint on this machine, or one already deleted, isn't an
        error; the point is the CATALOG no longer listing it, not
        enforcing the file existed."""
        source = self.policy_sources.pop(name, None)
        if source is None:
            return
        checkpoint = source.get("checkpoint")
        if not checkpoint:
            return
        policy_dir = POLICIES_DIR / name
        if Path(checkpoint).parent == policy_dir:
            shutil.rmtree(policy_dir, ignore_errors=True)
        else:
            try:
                os.remove(checkpoint)
            except OSError:
                pass

    def rename_policy(self, old_name: str, new_name: str) -> None:
        """Renames a UI-trained policy's dedicated `policies/<name>/`
        folder — the counterpart to finalize_policy() creating it and
        forget_source() deleting it. Only works for a policy that HAS such
        a folder (i.e. one this UI trained or cloned into existence); a
        bare --policy CLI source like `stable`, which lives wherever its
        checkpoint originally was and has no folder of its own to rename,
        is rejected rather than silently doing nothing. meta.json's
        contents don't need rewriting — the name isn't stored inside it,
        policy_info() stamps it back in from the folder name on every
        read (see its docstring)."""
        old_dir = POLICIES_DIR / old_name
        if not old_dir.is_dir():
            raise ValueError(f"'{old_name}' has no dedicated policies/ folder — nothing to rename")
        new_dir = POLICIES_DIR / new_name
        if new_dir.exists():
            raise FileExistsError(f"a policy named '{new_name}' already exists")
        old_dir.rename(new_dir)

        source = self.policy_sources.pop(old_name, None)
        if source is not None:
            def _repoint(path: Optional[str]) -> Optional[str]:
                if not path or Path(path).parent != old_dir:
                    return path
                return str(new_dir / Path(path).name)
            source["checkpoint"] = _repoint(source.get("checkpoint"))
            source["train_checkpoint"] = _repoint(source.get("train_checkpoint"))
            self.policy_sources[new_name] = source

    def catalog(self, compatible_tasks: Optional[Sequence[str]] = None) -> dict:
        from legged_gym.control import distillation, fusion
        # Unguarded, this raised under `.venv-mjlab` (no Genesis to import),
        # which broke the Create Policy panel outright on an mjlab session.
        # There, mjlab's own registry is the only task list that exists.
        try:
            from legged_gym.utils import task_registry
            all_tasks = sorted(task_registry.task_classes.keys())
        except ImportError:
            all_tasks = sorted(_mjlab_registered_tasks() or [])
        tasks = sorted(compatible_tasks) if compatible_tasks is not None else all_tasks
        base_policies = [
            # info already carries "simulator" (see register_source()) —
            # surfaced here so Clone-from can flag an isaacgym/genesis
            # sim2sim mismatch instead of silently fine-tuning across engines.
            # Per-variable reference values come from task_defaults(info["task"])
            # instead of being duplicated here — see app.js's refreshTargetReferences().
            {"name": name, **info}
            for name, info in sorted(self.policy_sources.items())
        ]
        return {
            "tasks": tasks,
            "task_notes": {t: self.TASK_NOTES[t] for t in tasks if t in self.TASK_NOTES},
            "base_policies": base_policies,
            # Same filter --from_policy already requires (a train_checkpoint, not just
            # an exported checkpoint.pt) — the Fuse policies panel's source picklist.
            "fusable_policies": [p for p in base_policies if p.get("train_checkpoint")],
            # Every fusion method this build knows about, available or not — see
            # fusion.FUSION_METHODS's own docstring for why unavailable ones (e.g.
            # "git_rebasin") are listed too: the panel/CLI should propose the roadmap,
            # not hide it until it's implemented.
            "fusion_methods": [{"id": key, **info} for key, info in fusion.FUSION_METHODS.items()],
            # Every distillation method this build knows about, available or not —
            # same "propose the roadmap, don't hide it" reasoning as fusion_methods
            # above. Unlike fusable_policies, the Distill panel's own source picker
            # uses base_policies directly (not a filtered list) — a checkpoint-only
            # policy with no train_checkpoint.pt (e.g. 'stable') is exactly the
            # normal case distillation exists to handle, not one to exclude.
            "distill_methods": [{"id": key, **info} for key, info in distillation.DISTILL_METHODS.items()],
            # Task-independent half of VARIABLE_REGISTRY (label/unit/source/
            # flag/note) — populates the 4 always-visible target fields once
            # per connection. The task-dependent half (reference, which
            # differs per task/clone-from base) comes from task_defaults()
            # instead, called again on every task/base change.
            "variables": {
                key: {k: v for k, v in meta.items() if k != "config_attr"}
                for key, meta in self.VARIABLE_REGISTRY.items()
            },
        }

    def task_defaults(self, task: str) -> dict:
        """Reference values the Create Policy panel reads off a task's own
        config — WITHOUT running the sim — so the '% change' fields (e.g.
        raise/lower the base height by some percent) have something concrete
        to apply that percent to. This is the task's config default, not
        necessarily the exact value a specific checkpoint was actually
        trained with (a prior job may have overridden it) — the best
        available reference short of loading and stepping that checkpoint.

        'variables' is the generic form of this — one entry per
        VARIABLE_REGISTRY key, each carrying a reference. Also used, with the
        base policy's OWN task, to resolve a clone-from reference (see
        app.js's refreshTargetReferences()) — every registered variable gets
        the same treatment, base_height included, no special case. The
        task-independent half of the registry (label/unit/source/flag/note)
        comes from catalog() instead — fetched once, not on every task
        change.

        'variables' is EMPTY (not full-with-None) for a task where none of
        VARIABLE_REGISTRY's base-height/tilt-style targets apply at all (a
        motion-tracking task has no such concept) — this is the signal the
        Create Policy panel uses to hide the Command envelope/Target
        variables/Push disturbances field-groups entirely (see app.js's
        refreshTargetReferences()/renderTaskFieldVisibility()), by checking
        whether the relevant keys are PRESENT, not by hardcoding a task-name
        check. A task that legitimately has these targets but momentarily
        can't resolve one (e.g. a broken cfg) still gets the key with
        reference: None — only "this concept doesn't exist for this task"
        omits the key.

        'needs_motion_file' flags whether this task's own command term
        requires a --motion_file (mirrors start()'s own check and
        mjlab_train.py's --motion_file requirement in
        docs/mjlab_training_contract.md §2: "yes for any task whose
        env_cfg.commands contains 'motion'") — read generically off the
        task's own cfg rather than a name check, so a future non-mimic
        tracking task gets the motion-clip picker for free."""
        # An mjlab task's reward weights live in a plain dict of RewardTermCfg
        # (env_cfg.rewards[<term>].weight), and none of VARIABLE_REGISTRY's
        # base-height/tilt-style targets exist for a motion-tracking task —
        # so 'variables' comes back empty (see docstring above), not
        # full-keyed-with-None.
        #
        # training_backend_for_task() (not _mjlab_registered_tasks() alone)
        # decides whether `task` IS an mjlab task, so this branch is also
        # taken from a process that can't import mjlab itself (e.g. rugiar's
        # own `.venv` install — see _mjlab_registry_snapshot()'s docstring).
        # A neither-registry-importable ValueError is treated the same as
        # the Genesis except-ImportError branch below: "can't validate",
        # not "this task has no reward terms".
        try:
            is_mjlab_task = training_backend_for_task(task) == "mjlab"
        except ValueError:
            is_mjlab_task = False
        if is_mjlab_task:
            if _mjlab_registered_tasks() is not None:
                import mjlab_tasks  # noqa: F401 - import side effect: registers this repo's own tasks
                from mjlab.tasks import registry
                cfg = registry.load_env_cfg(task)
                reward_scales = {name: term.weight for name, term in cfg.rewards.items()}
                needs_motion_file = "motion" in getattr(cfg, "commands", {})
            else:
                info = (_mjlab_registry_snapshot() or {}).get(task)
                reward_scales = info["reward_scales"] if info else {}
                needs_motion_file = info["needs_motion_file"] if info else True
            return {
                "variables": {},
                "reward_scales": reward_scales,
                "reward_scale_notes": {
                    term: note for term, note in self.REWARD_SCALE_NOTES.items()
                    if term in reward_scales
                },
                "needs_motion_file": needs_motion_file,
            }

        # Guarded for the same reason as catalog()'s: unimportable under
        # `.venv-mjlab`, where an empty result means "this process can't
        # validate", not "this task has no reward terms" (see start()).
        try:
            from legged_gym.utils import task_registry
            env_cfg, _ = task_registry.get_cfgs(name=task)
        except ImportError:
            return {"variables": {key: {"reference": None} for key in self.VARIABLE_REGISTRY},
                    "reward_scales": {}, "reward_scale_notes": {}, "needs_motion_file": False}
        except Exception:  # noqa: BLE001 - a broken/unregistered cfg shouldn't break the panel
            env_cfg = None

        variables = {}
        for key, meta in self.VARIABLE_REGISTRY.items():
            reference = getattr(env_cfg.rewards, meta["config_attr"], None) if env_cfg is not None else None
            variables[key] = {"reference": reference}

        # Every reward-term weight this task's own config defines, name ->
        # current default — read straight off <Cfg>.rewards.scales rather
        # than hand-maintaining a list here, so a task that adds/removes a
        # term shows up correctly with zero changes on this side. This is
        # what lets the Create Policy panel's "Reward weights" section be
        # fully task-driven: it doesn't know what a task rewards, it just
        # renders whatever this returns and lets --reward_scale override
        # any subset of it (see web_train.py's --reward_scale).
        reward_scales = {}
        if env_cfg is not None:
            from legged_gym.utils.helpers import class_to_dict
            reward_scales = {
                k: v for k, v in class_to_dict(env_cfg.rewards.scales).items()
                if isinstance(v, (int, float))
            }

        return {
            "variables": variables,
            "reward_scales": reward_scales,
            "reward_scale_notes": {
                term: note for term, note in self.REWARD_SCALE_NOTES.items() if term in reward_scales
            },
            "needs_motion_file": False,
        }

    # ---- launching ----

    def start(self, policy_name: str, task: str, num_envs: int = 64,
               max_iterations: Optional[int] = None, max_minutes: Optional[float] = None,
               base_policy: Optional[str] = None,
               cmd_vx: Optional[Sequence[float]] = None,
               cmd_vy: Optional[Sequence[float]] = None,
               cmd_yaw: Optional[Sequence[float]] = None,
               base_height_target: Optional[float] = None,
               lin_vel_z_target: Optional[float] = None,
               ang_vel_xy_target: Optional[float] = None,
               orientation_tilt_target: Optional[float] = None,
               push_robots: Optional[bool] = None,
               max_push_vel_xy: Optional[float] = None,
               push_interval_s: Optional[float] = None,
               push_dir: Optional[str] = None,
               entropy_coef: Optional[float] = None,
               reward_scale_overrides: Optional[Dict[str, float]] = None,
               motion_file: Optional[str] = None,
               backend: str = "local") -> str:
        """`backend` is a REQUEST ("local" | "kaggle"), not the descriptor
        that ends up serving it: which concrete backend runs the job is
        (request, task stack) -> BACKENDS entry, resolved by
        resolve_training_backend() below. Nothing in this method knows what
        Genesis/mjlab/Kaggle are — see the registry's comment for how to add
        a fourth."""
        if backend not in REQUESTABLE_BACKENDS:
            raise ValueError(f"unknown backend '{backend}' — must be "
                             f"{' or '.join(repr(b) for b in REQUESTABLE_BACKENDS)}")
        policy_name = (policy_name or "").strip()
        if not policy_name:
            raise ValueError("policy_name is required")
        if policy_name == "damping":
            raise ValueError("'damping' is reserved for the built-in safety fallback")
        if any(j.status == "running" and j.policy_name == policy_name for j in self.jobs.values()):
            raise ValueError(f"a training job for policy '{policy_name}' is already running")
        if max_iterations is None and max_minutes is None:
            raise ValueError("give at least one of max_iterations / max_minutes")
        max_iterations = int(max_iterations) if max_iterations is not None else None
        max_minutes = float(max_minutes) if max_minutes is not None else None
        num_envs = int(num_envs)
        if max_iterations is not None and max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if max_minutes is not None and max_minutes <= 0:
            raise ValueError("max_minutes must be positive")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if entropy_coef is not None and entropy_coef < 0:
            raise ValueError("entropy_coef can't be negative")

        # Which concrete backend serves this (task, request) pair — resolved
        # before the reward-scale check below, since where the valid
        # reward-term names even come from depends on the task's stack.
        # Raises for kaggle+mjlab and any other unserviceable combination.
        train_backend = resolve_training_backend(task, backend)
        if train_backend.validate_params is not None:
            # Backend-specific knob rejection, done here rather than in the
            # subprocess so the panel shows it immediately — same reasoning as
            # the reward-scale pre-validation below.
            train_backend.validate_params({
                "task": task, "motion_file": motion_file,
                "cmd_vx": cmd_vx, "cmd_vy": cmd_vy, "cmd_yaw": cmd_yaw,
                "base_height_target": base_height_target,
                "lin_vel_z_target": lin_vel_z_target,
                "ang_vel_xy_target": ang_vel_xy_target,
                "orientation_tilt_target": orientation_tilt_target,
                "push_robots": push_robots, "max_push_vel_xy": max_push_vel_xy,
                "push_interval_s": push_interval_s, "push_dir": push_dir,
            })

        if reward_scale_overrides:
            # Validated here, not left to the worker's subprocess exit
            # code — a typo'd term name should be a clear error in the
            # panel immediately, not a job that dies 10s later with
            # "exited with code 2, see the log".
            known = self.task_defaults(task)["reward_scales"]
            # Empty means "this process couldn't import the registry that
            # defines them" (e.g. a Genesis task queried from an mjlab
            # session), NOT "every name is invalid" — leave those to the
            # worker, which can import it.
            unknown = [k for k in reward_scale_overrides if k not in known] if known else []
            if unknown:
                raise ValueError(f"unknown reward scale(s) for task '{task}': {', '.join(unknown)}")

        from_checkpoint = None
        if base_policy:
            source = self.policy_sources.get(base_policy)
            if source is None:
                raise ValueError(f"unknown base policy '{base_policy}'")
            # Deliberately train_checkpoint, NOT checkpoint — see
            # _train_checkpoint_from_export()'s docstring. Passing the
            # exported (checkpoint) path here is exactly the bug that made
            # the first real 'Clone from' run crash instantly.
            from_checkpoint = source.get("train_checkpoint")
            if not from_checkpoint:
                raise ValueError(
                    f"base policy '{base_policy}' has no local training checkpoint to fine-tune from "
                    f"(only an exported/deployable .pt — e.g. an externally-sourced policy with no "
                    f"training history on this machine)")
        if train_backend.preflight is not None:
            # "Can this backend even run here" — credentials, quota, a
            # required daemon. Genesis/mjlab have none; Kaggle checks for
            # ~/.kaggle/kaggle.json.
            train_backend.preflight()
        # Cloning a Genesis-trained base into an isaacgym (Kaggle) run —
        # or vice versa, locally — is mechanically fine (torch doesn't
        # care where the weights came from) but a real sim2sim transfer:
        # the two engines' contact/PD dynamics differ (see
        # TrainingJob.simulator's docstring / HANDOFF_kaggle_cloud_gpu.md
        # open question #4). Deliberately not blocked here — the Create
        # Policy panel flags the mismatch instead (see catalog()'s
        # per-base "simulator" field), so an informed choice still goes
        # through.

        job_id = uuid.uuid4().hex[:8]
        result_path = JOBS_DIR / f"{job_id}.result.json"
        progress_path = JOBS_DIR / f"{job_id}.progress.json"
        log_path = JOBS_DIR / f"{job_id}.log"

        # The flags shared by both backends AND by both the web_train.py argv
        # and the rugiar CLI equivalent shown to the user (job.command) —
        # everything except how the base policy to fine-tune from is
        # expressed, which differs between the two (see below), and
        # --headless/--cpu/--result_path/--progress_path, which differ per
        # backend (a Kaggle kernel always has a GPU and its own filesystem
        # layout — see kaggle_backend._build_kernel_script).
        shared_flags: List[str] = [
            "--task", task,
            "--name", policy_name,
            "--num_envs", str(num_envs),
        ]
        if max_iterations is not None:
            shared_flags += ["--max_iterations", str(max_iterations)]
        if max_minutes is not None:
            shared_flags += ["--max_minutes", str(max_minutes)]
        if cmd_vx:
            shared_flags += ["--cmd_vx_range", str(cmd_vx[0]), str(cmd_vx[1])]
        if cmd_vy:
            shared_flags += ["--cmd_vy_range", str(cmd_vy[0]), str(cmd_vy[1])]
        if cmd_yaw:
            shared_flags += ["--cmd_yaw_range", str(cmd_yaw[0]), str(cmd_yaw[1])]
        if base_height_target is not None:
            shared_flags += ["--base_height_target", str(base_height_target)]
        if lin_vel_z_target is not None:
            shared_flags += ["--lin_vel_z_target", str(lin_vel_z_target)]
        if ang_vel_xy_target is not None:
            shared_flags += ["--ang_vel_xy_target", str(ang_vel_xy_target)]
        if orientation_tilt_target is not None:
            shared_flags += ["--orientation_tilt_target", str(orientation_tilt_target)]
        if push_robots is not None:
            shared_flags += ["--push_robots", "on" if push_robots else "off"]
        if max_push_vel_xy is not None:
            shared_flags += ["--max_push_vel_xy", str(max_push_vel_xy)]
        if push_interval_s is not None:
            shared_flags += ["--push_interval_s", str(push_interval_s)]
        if push_dir is not None:
            shared_flags += ["--push_dir", push_dir]
        if entropy_coef is not None:
            shared_flags += ["--entropy_coef", str(entropy_coef)]
        if reward_scale_overrides:
            for name, value in sorted(reward_scale_overrides.items()):
                shared_flags += ["--reward_scale", name, str(value)]
        if motion_file is not None:
            shared_flags += ["--motion_file", motion_file]

        train_flags = list(shared_flags)
        if from_checkpoint and train_backend.accepts_local_checkpoint:
            # A remote backend can't take this local absolute path directly —
            # it has no access to this machine's filesystem. KaggleRunner
            # uploads the same file as a private Kaggle Dataset and adds its
            # own --from_checkpoint pointing at the /kaggle/input/ mount
            # (see launch_remote below).
            train_flags += ["--from_checkpoint", from_checkpoint]

        # rugiar (legged_gym/cli/rugiar.py) is the same TrainingManager.start()
        # wrapped as a CLI, but it takes a policy NAME (--from_policy) rather
        # than a resolved --from_checkpoint path — so job.command (what the
        # UI shows as "copy the exact command") is built from this list,
        # separate from train_flags/argv which is what actually runs.
        rugiar_flags = list(shared_flags)
        if base_policy:
            rugiar_flags += ["--from_policy", base_policy]

        job = TrainingJob(
            id=job_id, policy_name=policy_name, task=task, command="",
            log_path=str(log_path), result_path=str(result_path), progress_path=str(progress_path),
            started_at=time.time(),
            max_iterations=max_iterations, max_minutes=max_minutes, num_envs=num_envs,
            base_policy=base_policy, entropy_coef=entropy_coef,
            reward_scale_overrides=dict(reward_scale_overrides) if reward_scale_overrides else None,
            backend=train_backend.job_backend,
            simulator=train_backend.simulator,
            motion_file=motion_file,
        )

        # The rugiar CLI equivalent of this run — this string is what the
        # web UI already showed as a preview before Start was clicked
        # (see web/app.js's updateCommandPreview), and is what a user would
        # paste into a terminal to reproduce it, not the raw subprocess
        # invocation actually exec'd below.
        job.command = train_backend.command_prefix + " ".join(rugiar_flags)

        if train_backend.remote:
            # Runs off this machine entirely: no interpreter to pick, no env
            # to build, no Popen to poll. The descriptor owns everything up to
            # "the job is now running" (for Kaggle: hand it to a background
            # KaggleRunner thread — see kaggle_backend's module docstring).
            self.jobs[job_id] = job
            train_backend.launch_remote(self, job, {
                "train_flags": train_flags, "result_path": result_path,
                "log_path": log_path, "from_checkpoint": from_checkpoint,
            })
            return job_id

        argv = [
            train_backend.interpreter(self, task), "-u", str(train_backend.script),
            *train_backend.fixed_flags,
            "--result_path", str(result_path),
            "--progress_path", str(progress_path),
        ] + train_flags

        log_f = open(log_path, "w")
        env = dict(os.environ)
        # Each backend owns its own PYTHONPATH/SIMULATOR story — they are not
        # variations on a theme but exact opposites (see _genesis_prepare_env
        # vs _mjlab_prepare_env), which is precisely why this is a hook and
        # not a shared block with an if in it.
        train_backend.prepare_env(env)
        proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)

        self.jobs[job_id] = job
        self._procs[job_id] = proc
        self._log_files[job_id] = log_f
        return job_id

    def start_distillation(self, teacher: str, task: str, out_name: str,
                            rollout_steps: Optional[int] = None, bc_epochs: Optional[int] = None,
                            lr: Optional[float] = None, num_envs: int = 1,
                            method: str = "behavior_cloning",
                            dagger_rounds: Optional[int] = None,
                            dagger_beta0: Optional[float] = None,
                            dagger_beta_decay: Optional[float] = None) -> str:
        """Behavior-clones `teacher` — ANY known local policy, crucially
        including ones with no train_checkpoint.pt at all (e.g. `stable` —
        see policy.py's module docstring for the checkpoint shapes that
        covers) — into a fresh, fine-tunable policy named `out_name`,
        registered under `task`. Mirrors start()'s shape exactly (out-of-
        process subprocess, job id returned immediately, progress/result
        polled the same way via poll()) because a BC rollout+train run takes
        real wall-clock time like a training job, unlike fuse_policies()'s
        few-seconds blocking tensor op — see legged_gym/control/
        distillation.py's module docstring for the actual algorithm.

        num_envs defaults to 1, NOT start()'s 64 — some externally-sourced
        teachers (e.g. unitree_rl_gym's own TorchScript exports, loaded as
        policy.py's InternalStatePolicy) bake a fixed batch=1 into the
        exported module's own hidden-state buffers and simply crash on a
        larger batch (confirmed the hard way: 'stable' failing with
        `Expected hidden[0] size (1, 64, 64), got [1, 1, 64]`). A locally-
        trained teacher (ExplicitStatePolicy — this repo's own export
        convention) has no such limit and can pass a higher num_envs for a
        faster rollout, but 1 is the only value guaranteed safe for ANY
        teacher, so it's the default."""
        from legged_gym.control import distillation

        method_info = distillation.DISTILL_METHODS.get(method)
        if method_info is None:
            raise ValueError(f"unknown distillation method '{method}' — see distillation.DISTILL_METHODS")
        if not method_info["available"]:
            raise ValueError(f"distillation method '{method}' is planned but not yet implemented")

        out_name = (out_name or "").strip()
        if not out_name:
            raise ValueError("out_name is required")
        if out_name == "damping":
            raise ValueError("'damping' is reserved for the built-in safety fallback")
        if (POLICIES_DIR / out_name).exists():
            raise ValueError(f"'{out_name}' already exists — pick a different output name")
        if any(j.status == "running" and j.policy_name == out_name for j in self.jobs.values()):
            raise ValueError(f"a job for policy '{out_name}' is already running")

        source = self.policy_sources.get(teacher)
        if source is None:
            raise ValueError(f"'{teacher}' is not a known local policy")
        if not source.get("checkpoint"):
            raise ValueError(f"'{teacher}' has no checkpoint.pt to distill from")

        num_envs = int(num_envs)
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        rollout_steps = int(rollout_steps) if rollout_steps is not None else 4000
        bc_epochs = int(bc_epochs) if bc_epochs is not None else 20
        lr = float(lr) if lr is not None else 1e-3
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if bc_epochs <= 0:
            raise ValueError("bc_epochs must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")

        # dagger-only knobs — rollout_steps/bc_epochs above become PER-ROUND
        # values when method == "dagger" (see distillation.dagger_train()'s
        # docstring); ignored entirely for "behavior_cloning".
        dagger_rounds = int(dagger_rounds) if dagger_rounds is not None else 5
        dagger_beta0 = float(dagger_beta0) if dagger_beta0 is not None else 1.0
        dagger_beta_decay = float(dagger_beta_decay) if dagger_beta_decay is not None else 0.5
        if dagger_rounds <= 0:
            raise ValueError("dagger_rounds must be positive")
        if not (0.0 <= dagger_beta0 <= 1.0):
            raise ValueError("dagger_beta0 must be in [0, 1]")
        if not (0.0 <= dagger_beta_decay <= 1.0):
            raise ValueError("dagger_beta_decay must be in [0, 1]")

        job_id = uuid.uuid4().hex[:8]
        result_path = JOBS_DIR / f"{job_id}.result.json"
        progress_path = JOBS_DIR / f"{job_id}.progress.json"
        log_path = JOBS_DIR / f"{job_id}.log"

        argv = [
            self.python_exe, "-u", str(DISTILL_SCRIPT),
            "--task", task, "--name", out_name,
            "--teacher_checkpoint", source["checkpoint"],
            "--method", method,
            "--rollout_steps", str(rollout_steps), "--bc_epochs", str(bc_epochs), "--lr", str(lr),
            "--num_envs", str(num_envs), "--headless", "--cpu",
            "--dagger_rounds", str(dagger_rounds), "--dagger_beta0", str(dagger_beta0),
            "--dagger_beta_decay", str(dagger_beta_decay),
            "--result_path", str(result_path), "--progress_path", str(progress_path),
        ]

        # dagger runs dagger_rounds * bc_epochs total BC epochs (one bc_train()
        # pass per round) — max_iterations must reflect that total, not the
        # per-round bc_epochs, or jobProgress()'s %/ETA math (which divides
        # iterations_done by this) would overshoot 100% partway through.
        total_bc_epochs = dagger_rounds * bc_epochs if method == "dagger" else bc_epochs
        job = TrainingJob(
            id=job_id, policy_name=out_name, task=task,
            command=(f"rugiar distill --teacher {teacher} --task {task} --name {out_name} "
                     f"--rollout_steps {rollout_steps} --bc_epochs {bc_epochs} --lr {lr} --num_envs {num_envs}"
                     + (f" --dagger_rounds {dagger_rounds}" if method == "dagger" else "")),
            log_path=str(log_path), result_path=str(result_path), progress_path=str(progress_path),
            started_at=time.time(),
            max_iterations=total_bc_epochs, max_minutes=None, num_envs=num_envs,
            job_type="distill", teacher_policy=teacher, distill_method=method,
            simulator=source.get("simulator", "genesis"),
        )

        log_f = open(log_path, "w")
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
        proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)

        self.jobs[job_id] = job
        self._procs[job_id] = proc
        self._log_files[job_id] = log_f
        return job_id

    # ---- polling (call once per sim tick — cheap, non-blocking) ----

    def _refresh_progress(self, job: TrainingJob) -> None:
        """Best-effort: read whatever web_train.py's write_progress() last
        wrote (see its own docstring — overwritten every
        TIME_BUDGET_CHUNK_ITERS iterations). The file may not exist yet
        (nothing written before the first chunk completes) or may be
        mid-write (we could race an OS-level partial write, though on Linux/
        macOS a single open+write+close of a small file is effectively
        atomic in practice) — either way, a bad read just means this tick's
        status push doesn't have a fresher number than the last one, never
        a crash. iterations_done is intentionally the SAME field poll()'s
        success path fills in from result.json — that call always comes
        after the process has exited, so it can't race this one, and it's
        authoritative (the exact final count) where this is a snapshot."""
        try:
            with open(job.progress_path) as f:
                progress = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        iterations_done = progress.get("iterations_done")
        if iterations_done is not None:
            job.iterations_done = iterations_done

    def poll(self) -> List[TrainingJob]:
        """Returns jobs that just finished (status 'done') on this call —
        the caller (ControlService.poll_finished_training(), see service.py)
        is responsible for actually loading policy_path and registering it.

        Called once per sim tick — must stay cheap and non-blocking for
        BOTH backends. Local jobs: Popen.poll() (an OS-level check, no
        I/O). Kaggle jobs: Thread.is_alive() — same cheap shape; all the
        actual Kaggle network calls happen inside that thread, never here
        (see kaggle_backend.KaggleRunner's module docstring)."""
        newly_done = []
        for job_id, job in self.jobs.items():
            if job.status != "running":
                continue
            job_backend = backend_for_job(job)
            if job_backend is not None and job_backend.remote:
                # Remote backends have no Popen to poll. Today "remote" means
                # exactly one thing (Kaggle), so the runner map below is
                # kaggle-typed; a second remote backend would add its own map
                # and pick between them off the descriptor, not off a name.
                runner = self._kaggle_runners[job_id]
                if job.kaggle_kernel_slug is None:
                    job.kaggle_kernel_slug = runner.kernel_ref  # set once the push completes
                if runner.is_alive():
                    continue  # no progress signal mid-run — see kaggle_backend's module docstring
                job.finished_at = time.time()
                if runner.error:
                    job.status = "failed"
                    job.error = runner.error
                    continue
                try:
                    with open(job.result_path) as f:
                        result = json.load(f)
                    job.policy_path = result["policy_path"]
                    job.train_checkpoint_path = result.get("train_checkpoint_path")
                    job.iterations_done = result.get("iterations_done")
                    job.status = "done"
                    newly_done.append(job)
                    # Recorded under backend="kaggle" — a separate bucket
                    # from local history (see estimate()'s docstring); mixing
                    # regimes would corrupt both estimates.
                    if job.iterations_done:
                        self._history.append({
                            "task": job.task, "max_iterations": job.iterations_done,
                            "num_envs": job.num_envs, "elapsed_s": job.finished_at - job.started_at,
                            "backend": "kaggle", "simulator": job.simulator,
                        })
                        self._save_history()
                except Exception as e:  # noqa: BLE001 - report to the UI, don't crash the sim loop
                    job.status = "failed"
                    job.error = f"Kaggle job finished but its result file was unreadable: {e}"
                continue

            proc = self._procs[job_id]
            rc = proc.poll()
            if rc is None:
                self._refresh_progress(job)
                continue
            job.finished_at = time.time()
            self._log_files[job_id].close()
            if rc != 0:
                job.status = "failed"
                # Which entrypoint actually ran — off the descriptor, not a
                # simulator-name check. A distill job never goes through a
                # training backend at all (it always runs DISTILL_SCRIPT),
                # so it's answered first.
                script_name = ("web_distill.py" if job.job_type == "distill"
                               else job_backend.script.name if job_backend is not None
                               else TRAIN_SCRIPT.name)
                job.error = f"{script_name} exited with code {rc} — see {job.log_path}"
                continue
            try:
                with open(job.result_path) as f:
                    result = json.load(f)
                job.policy_path = result["policy_path"]
                job.train_checkpoint_path = result.get("train_checkpoint_path")
                job.iterations_done = result.get("iterations_done")
                job.final_bc_loss = result.get("final_bc_loss")
                job.rollout_diagnostics = result.get("rollout_diagnostics")
                job.distill_loss_curve = result.get("loss_curve")
                job.distill_round_diagnostics = result.get("round_diagnostics")
                job.status = "done"
                newly_done.append(job)
                # Record actual iterations completed, not the requested cap —
                # with a --max_minutes budget those can differ a lot, and
                # estimate() needs the real throughput to be useful. Distill
                # jobs are excluded — their "iterations" are BC epochs, not
                # PPO learning iterations, and would skew estimate()'s
                # per-iteration-time model for ordinary training jobs.
                if job.iterations_done and job.job_type != "distill":
                    self._history.append({
                        "task": job.task, "max_iterations": job.iterations_done,
                        "num_envs": job.num_envs, "elapsed_s": job.finished_at - job.started_at,
                        "backend": "local", "simulator": job.simulator,
                    })
                    self._save_history()
            except Exception as e:  # noqa: BLE001 - report to the UI, don't crash the sim loop
                job.status = "failed"
                job.error = f"training process exited cleanly but its result file was unreadable: {e}"
        return newly_done

    def status(self) -> List[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.started_at)]
