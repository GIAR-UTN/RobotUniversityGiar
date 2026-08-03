"""
TrainingManager — lets the control web start a new policy training run and
find out, asynchronously, when it's ready to load. It owns exactly one
thing: launching legged_gym/scripts/web_train.py as a subprocess per job and
polling it. It never touches PolicySupervisor/ControlService/RobotAdapter —
same boundary the rest of legged_gym/control/ keeps (see
HANDOFF_control_web.md §5); the caller (swap_experiment.py's sim loop, via
ControlService — see service.py's start_training()/poll_finished_training())
is what actually loads the resulting checkpoint and registers it as a new
policy, exactly like restart_requested is drained there today.

Why a subprocess instead of an in-process training loop: `train.py`'s whole
stack (Genesis/gs.init, task_registry.make_env, the PPO runner) is built to
own a single process's global simulator state — running it would collide
with the swap_experiment.py sim already using the same globals. A subprocess
is the natural isolation boundary, and it's also what makes this safe to
poll cheaply (Popen.poll(), no subprocess.wait()) from a real-time control
loop.
"""
from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "web_train.py"
JOBS_DIR = REPO_ROOT / "logs" / "_web_training"
HISTORY_PATH = JOBS_DIR / "history.json"


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
    policy_path: Optional[str] = None

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
        }


class TrainingManager:
    def __init__(self, python_exe: str = sys.executable):
        self.python_exe = python_exe
        self.jobs: Dict[str, TrainingJob] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._log_files: Dict[str, "object"] = {}
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
            "simulator": os.environ.get("SIMULATOR", "unknown"),
            "genesis_backend": os.environ.get("GENESIS_BACKEND", "cpu"),
            # Not a measurement — a starting-point heuristic (envs run
            # vectorized but not free; more than a few per core stops
            # scaling on CPU). Gets less relevant once real history exists;
            # estimate()/the UI prefer measured numbers when they're available.
            "suggested_num_envs": {"comfortable": max(4, cpu_count * 4), "upper": max(8, cpu_count * 16)},
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
                 max_minutes: Optional[float] = None) -> dict:
        """Estimated (iterations, seconds) for a job on THIS machine, from
        this machine's own completed-job history — pooled across tasks
        (dominated by robot/obs/action space size, not which reward
        function is being trained, so cost-per-iteration is comparable
        across tasks). Works with either or both of max_iterations/
        max_minutes, mirroring the actual job's own 'whichever hits first'
        semantics (see web_train.py's chunked learn() loop) — if both are
        given, whichever resolves to fewer seconds wins. This is always an
        estimate, not a promise: per-iteration cost varies with machine
        load, so a wall-clock budget may stop a run a bit short of or past
        the iteration count shown here — it still stops on time; the
        iteration count just moves. Returns basis='none' (no invented
        number) when there's no history yet — see system_info()'s
        suggested_num_envs for a sizing starting point in that case."""
        num_envs = max(1, int(num_envs))
        none_result = {"basis": "none", "samples": 0, "seconds": None, "iterations": None}
        if not self._history:
            return none_result
        rates = [h["elapsed_s"] / (h["max_iterations"] * h["num_envs"])
                 for h in self._history if h["max_iterations"] > 0 and h["num_envs"] > 0]
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

    # Every variable the Create Policy panel's target selector can offer, in
    # one place — add an entry here (plus a matching cfg.rewards field pair:
    # a scalar target the existing tracking reward already reads, and a
    # `<field>_range` physical clamp) and it shows up in the UI's variable
    # dropdown with no other backend change. 'flag' is the exact web_train.py
    # CLI arg the resolved number is sent as — Absolute/Relative/Extreme
    # modes all funnel through the SAME flag (see app.js's resolveTarget()),
    # so "extreme" never needs a new reward term: it just resolves to the
    # config's own physical bound instead of a user-typed number.
    VARIABLE_REGISTRY = {
        "base_height": {
            "label": "Pelvis height",
            "unit": "m",
            "source": "sim_ground_truth",
            "flag": "base_height_target",
            "config_attr": "base_height_target",
            "range_attr": "base_height_target_range",
            "note": "Not measured by any real sensor — see RobotState.base_height's docstring. "
                    "Fine as a training target since training only ever runs in sim.",
        },
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
        stay un-fine-tunable rather than silently guessing."""
        if not export_path:
            return None
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

    def register_source(self, name: str, task: str, checkpoint: Optional[str]) -> None:
        self.policy_sources[name] = {
            "task": task, "checkpoint": checkpoint,
            "train_checkpoint": self._train_checkpoint_from_export(checkpoint),
        }

    def forget_source(self, name: str) -> None:
        """Drops a policy from the clone-from catalog and deletes its
        exported checkpoint file — the counterpart to register_source(),
        for discarding a training experiment that didn't work out (see
        ControlService.delete_policy()). Deliberately only deletes
        `checkpoint` (the exported .pt actually referenced by the catalog/
        loadable into the supervisor) — NOT `train_checkpoint` (one of
        several model_N.pt snapshots sharing that run's log_dir alongside
        its TensorBoard event file); leaving those is harmless disk usage,
        not anything exposed in the UI, and avoids this method reaching
        into a whole run directory to guess what else is safe to remove.
        Best-effort on the file removal — a source with no checkpoint on
        this machine, or one already deleted, isn't an error; the point is
        the CATALOG no longer listing it, not enforcing the file existed."""
        source = self.policy_sources.pop(name, None)
        if source is None:
            return
        checkpoint = source.get("checkpoint")
        if checkpoint:
            try:
                os.remove(checkpoint)
            except OSError:
                pass

    def catalog(self, compatible_tasks: Optional[Sequence[str]] = None) -> dict:
        from legged_gym.utils import task_registry
        all_tasks = sorted(task_registry.task_classes.keys())
        tasks = sorted(compatible_tasks) if compatible_tasks is not None else all_tasks
        return {
            "tasks": tasks,
            "base_policies": [
                {"name": name, "base_height_target": self._task_base_height(info["task"]), **info}
                for name, info in sorted(self.policy_sources.items())
            ],
            # Task-independent half of VARIABLE_REGISTRY (label/unit/source/
            # flag/note) — populates the target variable dropdown once per
            # connection. The task-dependent half (reference/range, which
            # differ per task/clone-from base) comes from task_defaults()
            # instead, called again on every task/base change.
            "variables": {
                key: {k: v for k, v in meta.items() if k not in ("config_attr", "range_attr")}
                for key, meta in self.VARIABLE_REGISTRY.items()
            },
        }

    def _task_base_height(self, task: str) -> Optional[float]:
        from legged_gym.utils import task_registry
        try:
            env_cfg, _ = task_registry.get_cfgs(name=task)
        except Exception:  # noqa: BLE001 - a broken/unregistered cfg shouldn't break the catalog
            return None
        return getattr(env_cfg.rewards, "base_height_target", None)

    def task_defaults(self, task: str) -> dict:
        """Reference values the Create Policy panel reads off a task's own
        config — WITHOUT running the sim — so 'relative' target fields (e.g.
        raise/lower the pelvis by N cm) have something concrete to add a
        delta to. This is the task's config default, not necessarily the
        exact value a specific checkpoint was actually trained with (a prior
        job may have overridden it) — the best available reference short of
        loading and stepping that checkpoint.

        'variables' is the generic form of the same idea — one entry per
        VARIABLE_REGISTRY key, each carrying a reference (for Relative mode)
        and a physical range (for Extreme mode's lowest/highest bounds).
        The task-independent half of the registry (label/unit/source/flag/
        note) comes from catalog() instead — fetched once, not on every
        task change. 'base_height_target' at the top level is kept for
        backward compat with the panel's existing pelvis-specific code
        path; it's exactly variables['base_height']['reference']."""
        from legged_gym.utils import task_registry
        try:
            env_cfg, _ = task_registry.get_cfgs(name=task)
        except Exception:  # noqa: BLE001 - a broken/unregistered cfg shouldn't break the panel
            env_cfg = None

        variables = {}
        for key, meta in self.VARIABLE_REGISTRY.items():
            reference = getattr(env_cfg.rewards, meta["config_attr"], None) if env_cfg is not None else None
            value_range = getattr(env_cfg.rewards, meta["range_attr"], None) if env_cfg is not None else None
            variables[key] = {
                "reference": reference,
                "range": list(value_range) if value_range is not None else None,
            }

        return {"base_height_target": self._task_base_height(task), "variables": variables}

    # ---- launching ----

    def start(self, policy_name: str, task: str, num_envs: int = 64,
               max_iterations: Optional[int] = None, max_minutes: Optional[float] = None,
               base_policy: Optional[str] = None,
               cmd_vx: Optional[Sequence[float]] = None,
               cmd_vy: Optional[Sequence[float]] = None,
               cmd_yaw: Optional[Sequence[float]] = None,
               base_height_target: Optional[float] = None,
               push_robots: Optional[bool] = None,
               max_push_vel_xy: Optional[float] = None,
               push_interval_s: Optional[float] = None,
               push_dir: Optional[str] = None) -> str:
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

        job_id = uuid.uuid4().hex[:8]
        result_path = JOBS_DIR / f"{job_id}.result.json"
        progress_path = JOBS_DIR / f"{job_id}.progress.json"
        log_path = JOBS_DIR / f"{job_id}.log"

        argv = [
            self.python_exe, "-u", str(TRAIN_SCRIPT),
            "--task", task,
            "--name", policy_name,
            "--num_envs", str(num_envs),
            "--headless", "--cpu",
            "--result_path", str(result_path),
            "--progress_path", str(progress_path),
        ]
        if max_iterations is not None:
            argv += ["--max_iterations", str(max_iterations)]
        if max_minutes is not None:
            argv += ["--max_minutes", str(max_minutes)]
        if from_checkpoint:
            argv += ["--from_checkpoint", from_checkpoint]
        if cmd_vx:
            argv += ["--cmd_vx_range", str(cmd_vx[0]), str(cmd_vx[1])]
        if cmd_vy:
            argv += ["--cmd_vy_range", str(cmd_vy[0]), str(cmd_vy[1])]
        if cmd_yaw:
            argv += ["--cmd_yaw_range", str(cmd_yaw[0]), str(cmd_yaw[1])]
        if base_height_target is not None:
            argv += ["--base_height_target", str(base_height_target)]
        if push_robots is not None:
            argv += ["--push_robots", "on" if push_robots else "off"]
        if max_push_vel_xy is not None:
            argv += ["--max_push_vel_xy", str(max_push_vel_xy)]
        if push_interval_s is not None:
            argv += ["--push_interval_s", str(push_interval_s)]
        if push_dir is not None:
            argv += ["--push_dir", push_dir]

        # Exactly what a human would type, modulo the interpreter path and
        # `-u` — this string is what the web UI already showed as a preview
        # before Start was clicked; nothing here should surprise it.
        display_command = "python " + " ".join(argv[2:])

        log_f = open(log_path, "w")
        # Pin PYTHONPATH to THIS repo checkout explicitly rather than trusting
        # whatever the parent process happened to be launched with — an
        # editable `pip install -e` of legged_gym elsewhere (e.g. a sibling
        # checkout of this same repo) would otherwise silently win, running
        # web_train.py's file from here against a DIFFERENT legged_gym
        # package. Bit us once already getting the control server itself to
        # run against the right checkout — not leaving it to chance twice.
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
        proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)

        job = TrainingJob(
            id=job_id, policy_name=policy_name, task=task, command=display_command,
            log_path=str(log_path), result_path=str(result_path), progress_path=str(progress_path),
            started_at=time.time(),
            max_iterations=max_iterations, max_minutes=max_minutes, num_envs=num_envs,
        )
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
        is responsible for actually loading policy_path and registering it."""
        newly_done = []
        for job_id, job in self.jobs.items():
            if job.status != "running":
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
                job.error = f"web_train.py exited with code {rc} — see {job.log_path}"
                continue
            try:
                with open(job.result_path) as f:
                    result = json.load(f)
                job.policy_path = result["policy_path"]
                job.iterations_done = result.get("iterations_done")
                job.status = "done"
                newly_done.append(job)
                # Record actual iterations completed, not the requested cap —
                # with a --max_minutes budget those can differ a lot, and
                # estimate() needs the real throughput to be useful.
                if job.iterations_done:
                    self._history.append({
                        "task": job.task, "max_iterations": job.iterations_done,
                        "num_envs": job.num_envs, "elapsed_s": job.finished_at - job.started_at,
                    })
                    self._save_history()
            except Exception as e:  # noqa: BLE001 - report to the UI, don't crash the sim loop
                job.status = "failed"
                job.error = f"training process exited cleanly but its result file was unreadable: {e}"
        return newly_done

    def status(self) -> List[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.started_at)]
