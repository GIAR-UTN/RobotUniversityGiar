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


# ---- turning a raw job .log into something a human (or the info popup) can read ----

_ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
_STAT_RES = {
    "noise_std": re.compile(r"Mean action noise std:\s*([-\d.]+)"),
    "reward": re.compile(r"Mean reward:\s*([-\d.]+)"),
    "episode_length": re.compile(r"Mean episode length:\s*([-\d.]+)"),
}
_TERM_RE = re.compile(r"Mean episode rew_(\w+):\s*([-\d.]+)")
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
                current["terms"][m.group(1)] = float(m.group(2))
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
    simulator: str = "genesis"  # "genesis" | "isaacgym" — which Simulator backend actually
                                 # trained this policy (see legged_gym/simulator/). Kaggle jobs
                                 # use isaacgym: Genesis's GPU JIT needs Volta+ (sm_70+) hardware
                                 # Kaggle's free-tier P100 (Pascal, sm_60) doesn't have, while
                                 # Isaac Gym's PhysX GPU pipeline runs on Pascal fine (see
                                 # HANDOFF_kaggle_cloud_gpu.md). Recorded here (not derived from
                                 # `backend`) because it's what actually determines sim2sim risk
                                 # for a policy trained under it — surfaced in meta.json so the
                                 # UI can flag it (see finalize_policy()).

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
                 max_minutes: Optional[float] = None, backend: str = "local") -> dict:
        """Estimated (iterations, seconds) for a job on the given backend,
        from THAT backend's own completed-job history — pooled across tasks
        (dominated by robot/obs/action space size, not which reward
        function is being trained, so cost-per-iteration is comparable
        across tasks), but never pooled ACROSS backends: a local-CPU run
        and a Kaggle Isaac-Gym-GPU run are different throughput regimes
        entirely (real numbers this session: local CPU ran single-digit
        iterations/sec on g1; the Kaggle GPU smoke test did 5 iterations
        with 16 envs in ~254s wall-clock including Isaac Gym's own ~3-4min
        per-job bootstrap) — mixing them would corrupt both estimates.
        History entries with no "backend" key predate this distinction and
        are treated as "local" (every job used to run there).

        Works with either or both of max_iterations/max_minutes, mirroring
        the actual job's own 'whichever hits first' semantics (see
        web_train.py's chunked learn() loop) — if both are given, whichever
        resolves to fewer seconds wins. This is always an estimate, not a
        promise: per-iteration cost varies with machine load (and, for
        Kaggle, with whatever GPU that session happens to get), so a
        wall-clock budget may stop a run a bit short of or past the
        iteration count shown here — it still stops on time; the iteration
        count just moves. Returns basis='none' (no invented number) when
        there's no history yet for this backend — see system_info()'s
        suggested_num_envs for a sizing starting point in that case."""
        num_envs = max(1, int(num_envs))
        none_result = {"basis": "none", "samples": 0, "seconds": None, "iterations": None}
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
        matters for a policy passed via swap_experiment.py's `--policy
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
                         simulator: str = "genesis") -> None:
        """`train_checkpoint` is the raw rsl_rl checkpoint to resume PPO
        from (see finalize_policy()'s docstring for how a fresh training
        job gets one). Pass None (the --policy CLI path, via
        swap_experiment.py) to fall back to guessing it from `checkpoint`'s
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
        engines."""
        self.policy_sources[name] = {
            "task": task, "checkpoint": checkpoint,
            "train_checkpoint": train_checkpoint or self._train_checkpoint_from_export(checkpoint),
            "simulator": simulator,
        }

    def finalize_policy(self, name: str, task: str, checkpoint: str,
                         train_checkpoint: Optional[str],
                         job: Optional["TrainingJob"] = None) -> str:
        """Called once a training job finishes (see swap_experiment.py's
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
        dest_checkpoint = dest_dir / "checkpoint.pt"
        shutil.copyfile(checkpoint, dest_checkpoint)
        dest_train_checkpoint = None
        if train_checkpoint and os.path.isfile(train_checkpoint):
            dest_train_checkpoint = dest_dir / "train_checkpoint.pt"
            shutil.copyfile(train_checkpoint, dest_train_checkpoint)

        meta = {"task": task, "created_at": time.time(), "trained_via": "control web",
                "simulator": job.simulator if job is not None else "genesis"}
        if job is not None:
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
        they'd survive that. This is what lets swap_experiment.py's
        startup re-offer every previously-trained policy instead of just
        whatever --policy flags happened to be typed that time.

        Returns name -> {"task", "checkpoint", "train_checkpoint"} for
        every folder with a checkpoint.pt, skipping names in `exclude`
        (already loaded a different way, e.g. via --policy) and skipping
        (with nothing raised — this must never crash startup) anything
        without a readable meta.json giving its task, since loading a
        checkpoint from the wrong task/observation-space would crash
        load_policy() rather than just fail to appear."""
        found = {}
        if not POLICIES_DIR.is_dir():
            return found
        for entry in sorted(POLICIES_DIR.iterdir()):
            name = entry.name
            if name in exclude or not entry.is_dir():
                continue
            checkpoint = entry / "checkpoint.pt"
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
        from legged_gym.control import fusion
        from legged_gym.utils import task_registry
        all_tasks = sorted(task_registry.task_classes.keys())
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
        change."""
        from legged_gym.utils import task_registry
        try:
            env_cfg, _ = task_registry.get_cfgs(name=task)
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
               backend: str = "local") -> str:
        if backend not in ("local", "kaggle"):
            raise ValueError(f"unknown backend '{backend}' — must be 'local' or 'kaggle'")
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
        if reward_scale_overrides:
            # Validated here, not left to web_train.py's subprocess exit
            # code — a typo'd term name should be a clear error in the
            # panel immediately, not a job that dies 10s later with
            # "exited with code 2, see the log".
            known = self.task_defaults(task)["reward_scales"]
            unknown = [k for k in reward_scale_overrides if k not in known]
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
        if backend == "kaggle":
            if not kaggle_backend.kaggle_credentials_available():
                raise ValueError(
                    "no Kaggle credentials found at ~/.kaggle/kaggle.json — the Kaggle backend "
                    "isn't set up on this machine")
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

        train_flags = list(shared_flags)
        if from_checkpoint and backend == "local":
            # Kaggle jobs can't take this local absolute path directly — the
            # kernel has no access to this machine's filesystem. KaggleRunner
            # uploads the same file as a private Kaggle Dataset and adds its
            # own --from_checkpoint pointing at the /kaggle/input/ mount
            # (see the backend=="kaggle" branch below).
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
            backend=backend,
            simulator="isaacgym" if backend == "kaggle" else "genesis",
        )

        if backend == "kaggle":
            # The rugiar equivalent of what the kernel runs — same "show the
            # real command" spirit as the local job below, minus the
            # base-checkpoint upload/mount step rugiar's own --backend
            # kaggle handles the same way TrainingManager does here.
            job.command = "rugiar train --backend kaggle " + " ".join(rugiar_flags)
            runner = kaggle_backend.KaggleRunner(
                job_id=job_id, train_flags=train_flags,
                result_path=result_path, log_path=log_path,
                base_checkpoint_path=Path(from_checkpoint) if from_checkpoint else None,
            )
            # kaggle_kernel_slug stays None until poll() sees runner.kernel_ref
            # populated (set inside the background thread once its push succeeds).
            self._kaggle_runners[job_id] = runner
            self.jobs[job_id] = job
            runner.start()
            return job_id

        argv = [
            self.python_exe, "-u", str(TRAIN_SCRIPT),
            "--headless", "--cpu",
            "--result_path", str(result_path),
            "--progress_path", str(progress_path),
        ] + train_flags

        # The rugiar CLI equivalent of this run — this string is what the
        # web UI already showed as a preview before Start was clicked
        # (see web/app.js's updateCommandPreview), and is what a user would
        # paste into a terminal to reproduce it, not the raw web_train.py
        # subprocess invocation actually exec'd below.
        job.command = "rugiar train " + " ".join(rugiar_flags)

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
            if job.backend == "kaggle":
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
                            "backend": "kaggle",
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
                job.error = f"web_train.py exited with code {rc} — see {job.log_path}"
                continue
            try:
                with open(job.result_path) as f:
                    result = json.load(f)
                job.policy_path = result["policy_path"]
                job.train_checkpoint_path = result.get("train_checkpoint_path")
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
                        "backend": "local",
                    })
                    self._save_history()
            except Exception as e:  # noqa: BLE001 - report to the UI, don't crash the sim loop
                job.status = "failed"
                job.error = f"training process exited cleanly but its result file was unreadable: {e}"
        return newly_done

    def status(self) -> List[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.started_at)]
