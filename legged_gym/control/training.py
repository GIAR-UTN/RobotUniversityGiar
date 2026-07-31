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
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "web_train.py"
JOBS_DIR = REPO_ROOT / "logs" / "_web_training"


@dataclasses.dataclass
class TrainingJob:
    id: str
    policy_name: str
    task: str
    command: str  # display string — exactly what the UI previewed, minus the interpreter path
    log_path: str
    result_path: str
    started_at: float
    status: str = "running"  # running | done | failed
    finished_at: Optional[float] = None
    error: Optional[str] = None
    policy_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "policy_name": self.policy_name,
            "task": self.task,
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

    # ---- catalog the web UI's form renders from ----

    def register_source(self, name: str, task: str, checkpoint: Optional[str]) -> None:
        self.policy_sources[name] = {"task": task, "checkpoint": checkpoint}

    def catalog(self, compatible_tasks: Optional[Sequence[str]] = None) -> dict:
        from legged_gym.utils import task_registry
        all_tasks = sorted(task_registry.task_classes.keys())
        tasks = sorted(compatible_tasks) if compatible_tasks is not None else all_tasks
        return {
            "tasks": tasks,
            "base_policies": [
                {"name": name, **info} for name, info in sorted(self.policy_sources.items())
            ],
        }

    # ---- launching ----

    def start(self, policy_name: str, task: str, max_iterations: int, num_envs: int = 64,
               base_policy: Optional[str] = None,
               cmd_vx: Optional[Sequence[float]] = None,
               cmd_vy: Optional[Sequence[float]] = None,
               cmd_yaw: Optional[Sequence[float]] = None) -> str:
        policy_name = (policy_name or "").strip()
        if not policy_name:
            raise ValueError("policy_name is required")
        if policy_name == "damping":
            raise ValueError("'damping' is reserved for the built-in safety fallback")
        if any(j.status == "running" and j.policy_name == policy_name for j in self.jobs.values()):
            raise ValueError(f"a training job for policy '{policy_name}' is already running")
        max_iterations = int(max_iterations)
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        from_checkpoint = None
        if base_policy:
            source = self.policy_sources.get(base_policy)
            if source is None:
                raise ValueError(f"unknown base policy '{base_policy}'")
            from_checkpoint = source.get("checkpoint")
            if not from_checkpoint:
                raise ValueError(f"base policy '{base_policy}' has no known checkpoint file to fine-tune from")

        job_id = uuid.uuid4().hex[:8]
        result_path = JOBS_DIR / f"{job_id}.result.json"
        log_path = JOBS_DIR / f"{job_id}.log"

        argv = [
            self.python_exe, "-u", str(TRAIN_SCRIPT),
            "--task", task,
            "--name", policy_name,
            "--max_iterations", str(max_iterations),
            "--num_envs", str(int(num_envs)),
            "--headless", "--cpu",
            "--result_path", str(result_path),
        ]
        if from_checkpoint:
            argv += ["--from_checkpoint", from_checkpoint]
        if cmd_vx:
            argv += ["--cmd_vx_range", str(cmd_vx[0]), str(cmd_vx[1])]
        if cmd_vy:
            argv += ["--cmd_vy_range", str(cmd_vy[0]), str(cmd_vy[1])]
        if cmd_yaw:
            argv += ["--cmd_yaw_range", str(cmd_yaw[0]), str(cmd_yaw[1])]

        # Exactly what a human would type, modulo the interpreter path and
        # `-u` — this string is what the web UI already showed as a preview
        # before Start was clicked; nothing here should surprise it.
        display_command = "python " + " ".join(argv[2:])

        log_f = open(log_path, "w")
        proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT)

        job = TrainingJob(
            id=job_id, policy_name=policy_name, task=task, command=display_command,
            log_path=str(log_path), result_path=str(result_path), started_at=time.time(),
        )
        self.jobs[job_id] = job
        self._procs[job_id] = proc
        self._log_files[job_id] = log_f
        return job_id

    # ---- polling (call once per sim tick — cheap, non-blocking) ----

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
                job.status = "done"
                newly_done.append(job)
            except Exception as e:  # noqa: BLE001 - report to the UI, don't crash the sim loop
                job.status = "failed"
                job.error = f"training process exited cleanly but its result file was unreadable: {e}"
        return newly_done

    def status(self) -> List[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.started_at)]
