"""
KaggleRunner — the Kaggle-backed counterpart to TrainingManager's local
subprocess.Popen path (see training.py's TrainingManager.start()). One
instance per Kaggle training job, owns exactly one background thread that
does ALL the network talking to Kaggle (kernels_push/status/output).

Why a thread, not an inline call: transport.py's docstring is explicit that
start_training()/the per-tick poll() run ON THE SIM LOOP'S OWN THREAD, once
per control step — nothing there may block on a network round-trip.
`kaggle kernels push` and polling `kernels status` are both HTTP calls, so
they can never run inline. This thread is the stand-in for what a local
job's OS subprocess already is: something that runs to completion in the
background and is polled cheaply. TrainingManager.poll()'s Kaggle branch
polls this the same cheap way it polls a local Popen — `Thread.is_alive()` —
never touching the network itself; all the actual Kaggle HTTP calls happen
only inside this thread's run().

The repo this pushes to Kaggle is cloned fresh from GitHub inside the kernel
itself (see _build_kernel_script) rather than uploaded as a Kaggle Dataset —
only possible because this repo is public (github.com/josetabuyo/
RobotUniversityGiar). That keeps `kernels_push` fast (a few KB of script, not a
multi-GB upload of checkpoints/venvs/logs), which matters because push still
runs on this background thread's timeline, not the sim thread's — but a slow
push would still delay this thread noticing failures, so keeping it fast is
worth doing anyway. It does mean a Kaggle job trains whatever is on the
remote branch's HEAD at push time, not uncommitted local changes.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

REPO_URL = "https://github.com/josetabuyo/RobotUniversityGiar.git"
DEFAULT_BRANCH = "main"
POLL_INTERVAL_S = 15  # how often this thread (never the sim thread) checks kernels_status
# Kaggle's own GPU session cap is ~12h; bail well before that so a stuck/orphaned
# kernel can't wedge this thread (and the "job never finishes" UI state) forever.
MAX_RUNTIME_S = 6 * 3600
# How long a freshly-uploaded Clone-from base dataset gets to reach
# dataset_status() == "ready" before giving up — see the dataset_create_new()
# call site in KaggleRunner._run() for why this is polled, not slept.
DATASET_READY_TIMEOUT_S = 120

TERMINAL_STATUSES = {"complete", "error", "cancel_acknowledged"}

# Kaggle's free tier keeps assigning a Tesla P100 (Pascal, compute capability
# sm_60) — not the T4 the UI/docs suggest is the default (machine_shape
# below doesn't seem reliably honored). Genesis's GPU backend needs Volta+
# (sm_70+, for the `warp.sync` intrinsic its JIT compiler emits) — Pascal
# doesn't have the silicon for that, full stop, no dependency version routes
# around it (see HANDOFF_kaggle_cloud_gpu.md). Isaac Gym's PhysX GPU
# pipeline has no such requirement — confirmed working on Kaggle's actual
# P100 this session (real g1 env created, GPU Pipeline: enabled, stepped
# cleanly) — so Kaggle jobs run Isaac Gym, not Genesis, despite every local
# job on this repo defaulting to Genesis. See _build_kernel_script.


def _status_name(status) -> str:
    """kernels_status() returns an ApiGetKernelSessionStatusResponse whose
    `.status` is a plain (non-str) kagglesdk enum (KernelWorkerStatus) —
    comparing it directly against a string, or putting it `in` a set of
    strings, is always False (learned the hard way: a real smoke-test job
    sat reporting 'running' in the UI forever because this comparison
    silently never matched, even though the kernel itself had long since
    finished). Always go through `.name.lower()` instead."""
    return status.name.lower()


def kaggle_credentials_available() -> bool:
    """Best-effort, never raises — mirrors TrainingManager.system_info()'s
    other 'cosmetic' probes. A missing/broken kaggle.json should just hide
    the Kaggle option in the UI, not crash the status panel."""
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def _build_kernel_script(train_flags: List[str], branch: str,
                          has_base_checkpoint: bool = False) -> str:
    """The actual Python program Kaggle executes. Bootstraps a Python 3.8
    venv with Isaac Gym installed, clones THIS repo fresh into it, and runs
    the exact same legged_gym/scripts/web_train.py the control web already
    uses for local jobs (see training.py's TrainingManager.start()) — no
    training logic is duplicated here, only the environment bootstrap a
    fresh Kaggle session needs.

    Why Isaac Gym, not Genesis (this repo's default everywhere else): Kaggle's
    free tier keeps assigning a Tesla P100 (Pascal, sm_60). Genesis's GPU
    backend needs Volta+ (sm_70+, for the `warp.sync` intrinsic its JIT
    compiler emits) — Pascal doesn't have the silicon for that, no dependency
    pin routes around it. Isaac Gym's PhysX GPU pipeline has no such
    requirement and was confirmed working end-to-end on Kaggle's actual P100
    this session (real g1 env created, GPU Pipeline: enabled, stepped
    cleanly) — see HANDOFF_kaggle_cloud_gpu.md.

    Why a whole second Python interpreter: legged_gym/__init__.py picks the
    simulator by PYTHON VERSION, not (for isaacgym) the SIMULATOR env var —
    it hardcodes SIMULATOR="isaacgym" for any interpreter <=3.8, and on
    Kaggle's default (>=3.10) interpreter, "isaacgym" isn't even an accepted
    value for SIMULATOR (raises ValueError). So the ONLY way to select Isaac
    Gym is to run web_train.py under a 3.8 interpreter — this bootstraps one,
    mirroring both a colleague's own proven Kaggle recipe
    (kaggle.com/code/jvillalba007/unitree-rl) and this repo's own
    switch_simulator.sh (which likewise gives Isaac Gym its own environment).

    Built via plain string concatenation, not str.format()/an f-string,
    because the generated code below is full of its own braces (f-strings,
    dict literals) that would collide with format placeholders. Dynamic
    values (branch, flags) are embedded with json.dumps() so they come out
    as safe Python literals regardless of quotes/spaces inside them."""
    lines = [
        "import json, os, shutil, subprocess, sys",
        "",
        # Deliberately OUTSIDE /kaggle/working — that directory is exactly
        # what kernels_output() downloads wholesale afterward (see
        # KaggleRunner._run()), and a full repo checkout (.git, logs/,
        # rsl_rl's TensorBoard scratch space, the isaacgym venv itself) in
        # there turned kernels_output into a multi-minute (or, for the venv,
        # multi-GB) download for a job whose result is three small files.
        # Only those three (copied out below) ever need to survive.
        'REPO_DIR = "/tmp/repo"',
        'VENV_DIR = "/tmp/isaacgym-venv"',
        'VENV_PY = VENV_DIR + "/bin/python"',
        'RESULT_PATH = "/kaggle/working/result.json"',
        'LOG_PATH = "/kaggle/working/train.log"',
        "",
    ] + ([
        # Fail fast, BEFORE the multi-minute venv/Isaac Gym bootstrap below,
        # if the Clone-from base dataset never actually got attached to
        # THIS kernel session — a real, reproduced Kaggle flake: this
        # runner already waits for dataset_status()=='ready' before pushing
        # (see KaggleRunner._run()), but a session can still start with no
        # visible checkpoint despite that (confirmed twice: two separate
        # jobs both bootstrapped for 4+ minutes only to hit FileNotFoundError
        # deep in torch's checkpoint loader).
        #
        # Deliberately NOT a fixed path like /kaggle/input/<slug>/train_checkpoint.pt
        # — confirmed via a real failed kernel's own log that kernels pushed
        # via the API get their dataset sources namespaced one level deeper
        # than that (/kaggle/input/datasets/<owner>/<slug>/...), and that
        # nesting isn't documented/guaranteed to stay put. Glob for the file
        # by name instead, at any depth under /kaggle/input — nothing else
        # Kaggle mounts there would produce a file with this exact name, so
        # this is robust to whatever nesting Kaggle uses today or changes to
        # later. Poll a short grace period in case of a few-second
        # propagation lag, then bail with a marker string
        # (BASE_CHECKPOINT_MISSING) KaggleRunner greps for in this kernel's
        # own captured log to distinguish "retry with a fresh session" from
        # a real training crash.
        "import glob, time as _time",
        "_deadline = _time.time() + 45",
        "BASE_CHECKPOINT_MOUNT = None",
        "while BASE_CHECKPOINT_MOUNT is None and _time.time() < _deadline:",
        "    _matches = glob.glob('/kaggle/input/**/train_checkpoint.pt', recursive=True)",
        "    if _matches:",
        "        BASE_CHECKPOINT_MOUNT = _matches[0]",
        "    else:",
        "        _time.sleep(3)",
        "if BASE_CHECKPOINT_MOUNT is None:",
        "    print('BASE_CHECKPOINT_MISSING: no train_checkpoint.pt found under /kaggle/input')",
        "    print('/kaggle/input contents:', "
        "os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else '<no /kaggle/input>')",
        "    raise SystemExit('BASE_CHECKPOINT_MISSING: no train_checkpoint.pt found under /kaggle/input')",
        "",
    ] if has_base_checkpoint else []) + [
        # Python 3.8 venv + the exact torch build Isaac Gym Preview 4's own
        # compiled bindings need (also the build proven to still ship sm_60
        # kernels, unlike current torch releases) -- confirmed this exact
        # sequence installs and imports cleanly on a real Kaggle kernel.
        'subprocess.run(["sudo", "apt", "install", "-y", "python3.8", "python3.8-venv", '
        '"python3.8-dev"], check=True)',
        'subprocess.run(["python3.8", "-m", "venv", VENV_DIR], check=True)',
        'subprocess.run([VENV_PY, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)',
        'subprocess.run([VENV_PY, "-m", "pip", "install", "-q", '
        '"torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1", '
        '"--index-url", "https://download.pytorch.org/whl/cu121"], check=True)',
        "",
        # Isaac Gym itself isn't on PyPI -- NVIDIA ships it as a tarball.
        'subprocess.run(["wget", "-q", "-O", "/tmp/isaac-gym-preview-4.tar.gz", '
        '"https://developer.nvidia.com/isaac-gym-preview-4"], check=True)',
        'subprocess.run(["tar", "-xzf", "/tmp/isaac-gym-preview-4.tar.gz", "-C", "/tmp"], check=True)',
        'subprocess.run([VENV_PY, "-m", "pip", "install", "-q", "-e", '
        '"/tmp/isaacgym/python"], check=True)',
        "",
        "subprocess.run(["
        f'"git", "clone", "--depth", "1", "--branch", {json.dumps(branch)}, '
        f'{json.dumps(REPO_URL)}, REPO_DIR], check=True)',
        # pyproject.toml's isaacgym extra deliberately doesn't pin torch/
        # torchvision (a "+cu121" local-version pin isn't resolvable against
        # the default PyPI index -- confirmed the hard way) -- torch is
        # already installed above, from the index that actually has it.
        'subprocess.run([VENV_PY, "-m", "pip", "install", "-q", "-e", '
        'f"{REPO_DIR}[isaacgym]"], check=True)',
        "",
        "env = dict(os.environ)",
        # Not load-bearing for simulator SELECTION under this 3.8 venv (see
        # docstring above) -- set anyway in case anything downstream reads
        # it directly rather than relying on legged_gym/__init__.py.
        'env["SIMULATOR"] = "isaacgym"',
        "",
        # Isaac Gym's PhysX GPU pipeline runs fine on Pascal -- unlike
        # Genesis, there's no compute-capability floor to probe for. Just
        # confirm CUDA is actually present before requesting --gpu.
        "gpu_probe = subprocess.run([VENV_PY, '-c', "
        "'import torch; print(torch.cuda.is_available())'], "
        "capture_output=True, text=True)",
        "gpu_flag = ['--gpu'] if gpu_probe.stdout.strip() == 'True' else []",
        "if not gpu_flag:",
        "    print('CUDA not available on this Kaggle kernel -- training on CPU instead.')",
        "",
        # web_train.py's --headless already defaults to True; passed
        # explicitly anyway to match what a human would type, same as the
        # local-job command preview. BASE_CHECKPOINT_MOUNT (if a Clone-from
        # base was picked — see KaggleRunner._run()'s dataset upload) is
        # resolved above via glob, only inside THIS kernel, so it's appended
        # here rather than folded into train_flags upstream (training.py's
        # start() never sees it).
        'argv = [VENV_PY, "-u", "legged_gym/scripts/web_train.py",'
        ' "--headless", "--result_path", RESULT_PATH] + gpu_flag + '
        f"{json.dumps(train_flags)}"
        + (' + (["--from_checkpoint", BASE_CHECKPOINT_MOUNT] if BASE_CHECKPOINT_MOUNT else [])'
           if has_base_checkpoint else ""),
        "",
        'with open(LOG_PATH, "w") as log_f:',
        "    rc = subprocess.run(argv, cwd=REPO_DIR, env=env, stdout=log_f, "
        "stderr=subprocess.STDOUT).returncode",
        "if rc != 0:",
        '    raise SystemExit("web_train.py exited with code " + str(rc) + '
        '" -- see " + LOG_PATH)',
        "",
        "with open(RESULT_PATH) as f:",
        "    result = json.load(f)",
        "",
        "def _localize(p):",
        "    if not p:",
        "        return None",
        "    return p if os.path.isabs(p) else os.path.join(REPO_DIR, p)",
        "",
        'policy_src = _localize(result["policy_path"])',
        'policy_dst = "/kaggle/working/checkpoint.pt"',
        "shutil.copyfile(policy_src, policy_dst)",
        'result["policy_path"] = policy_dst',
        "",
        'train_ckpt_src = _localize(result.get("train_checkpoint_path"))',
        "if train_ckpt_src and os.path.isfile(train_ckpt_src):",
        '    train_ckpt_dst = "/kaggle/working/train_checkpoint.pt"',
        "    shutil.copyfile(train_ckpt_src, train_ckpt_dst)",
        '    result["train_checkpoint_path"] = train_ckpt_dst',
        "else:",
        '    result["train_checkpoint_path"] = None',
        "",
        'with open(RESULT_PATH, "w") as f:',
        "    json.dump(result, f)",
        "",
        'print("Done.")',
        "",
    ]
    return "\n".join(lines)


class KaggleRunner(threading.Thread):
    """One per Kaggle training job. `result_path`/`log_path` are the SAME
    paths TrainingManager already created for this job (see its start()) —
    on success this thread writes exactly the file TrainingManager.poll()
    already knows how to read for a finished job, just localized to files
    downloaded from Kaggle instead of ones a local subprocess wrote
    directly. On failure it writes nothing there and sets `.error` instead —
    poll()'s Kaggle branch checks that once this thread is no longer alive."""

    def __init__(self, job_id: str, train_flags: List[str],
                 result_path: Path, log_path: Path, branch: str = DEFAULT_BRANCH,
                 base_checkpoint_path: Optional[Path] = None):
        super().__init__(daemon=True, name=f"kaggle-train-{job_id}")
        self.job_id = job_id
        # Kaggle kernel slugs must be lowercase alphanumeric + hyphens.
        self.slug = f"legged-gym-ex-{job_id}"
        self.train_flags = train_flags
        self.result_path = result_path
        self.log_path = log_path
        self.branch = branch
        # Clone-from base's LOCAL train_checkpoint.pt (see
        # TrainingManager.start()'s backend=="kaggle" branch) — this thread
        # uploads it as a private Kaggle Dataset before pushing the kernel,
        # since the kernel has no access to this machine's filesystem. None
        # means "train from scratch", same as the local backend.
        self.base_checkpoint_path = base_checkpoint_path
        # Only set once base_checkpoint_path is actually uploaded — a
        # distinct dataset per job (not reused across jobs) so two
        # concurrent Kaggle jobs cloning different bases can never collide.
        self.dataset_slug = f"{self.slug}-base" if base_checkpoint_path else None
        # "<username>/<slug>" once push succeeds — the UI's "view on Kaggle" link.
        self.kernel_ref: Optional[str] = None
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 - must never crash this thread silently; report via .error instead
            self.error = str(e)

    def _kernel_log_has_marker(self, api, marker: str) -> bool:
        """Downloads this kernel's own captured stdout/stderr (works even on
        an 'error' status — kernels_output doesn't require 'complete') and
        checks for `marker`. Best-effort: a download failure here means
        "can't confirm", not "confirmed absent" — returns False either way,
        which is the safe default (skip the retry, report the plain error)."""
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                api.kernels_output(self.kernel_ref, path=out_dir, force=True, quiet=True)
                log_path = Path(out_dir) / f"{self.slug}.log"
                return log_path.is_file() and marker in log_path.read_text(errors="replace")
        except Exception:  # noqa: BLE001 - best-effort diagnostic, never fatal
            return False

    def _run(self) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        username = api.config_values.get("username")
        self.kernel_ref = f"{username}/{self.slug}"

        has_base_checkpoint = False
        if self.base_checkpoint_path is not None:
            dataset_ref = f"{username}/{self.dataset_slug}"
            with tempfile.TemporaryDirectory() as ds_tmp:
                ds_path = Path(ds_tmp)
                # dataset_create_new uploads every file in this directory —
                # keep it to exactly the one checkpoint, named the same
                # thing finalize_policy() always calls it locally, so the
                # kernel-side path below doesn't need to know the base
                # policy's own name.
                shutil.copyfile(self.base_checkpoint_path, ds_path / "train_checkpoint.pt")
                (ds_path / "dataset-metadata.json").write_text(json.dumps({
                    "title": self.dataset_slug,
                    "id": dataset_ref,
                    "licenses": [{"name": "CC0-1.0"}],
                }))
                api.dataset_create_new(str(ds_path), dir_mode="zip", quiet=True, convert_to_csv=False)
            # dataset_create_new() returns as soon as the upload finishes,
            # but Kaggle's own indexing lags behind that — confirmed the hard
            # way: a real job pushed the kernel 10s after upload and it ran
            # to completion (env bootstrap alone took ~4 minutes) only to
            # find /kaggle/input/<slug>/ empty, because the dataset hadn't
            # finished indexing when the kernel session actually mounted its
            # sources (kernels_push itself doesn't validate dataset_sources,
            # so the push succeeds either way — the failure only shows up
            # once web_train.py can't find the checkpoint file, ~4-5 minutes
            # and a wasted GPU-quota kernel run later). Poll dataset_status()
            # for "ready" instead of guessing a fixed delay.
            #
            # dataset_status() itself is flaky RIGHT after dataset_create_new()
            # returns — confirmed live: a real poll got "403 Client Error:
            # Forbidden" on its very first call, seconds after the upload
            # finished (the dataset's ACL/ownership hadn't propagated yet),
            # even though the exact same call against the exact same dataset
            # succeeded and returned "ready" moments later when retried by
            # hand. So a transient error here is expected noise, not a real
            # failure — only running out the deadline with no successful
            # "ready" is treated as one.
            deadline = time.time() + DATASET_READY_TIMEOUT_S
            ready = False
            last_status_error = None
            while time.time() < deadline:
                try:
                    if api.dataset_status(dataset_ref) == "ready":
                        ready = True
                        break
                except Exception as e:  # noqa: BLE001 - transient (403/network) right after upload, see above
                    last_status_error = str(e)
                time.sleep(3)
            if not ready:
                self.error = (
                    f"Kaggle dataset {dataset_ref} (the uploaded base checkpoint) never reached "
                    f"'ready' status within {DATASET_READY_TIMEOUT_S}s — not pushing a kernel that "
                    f"would just fail looking for it"
                    + (f" (last error: {last_status_error})" if last_status_error else ""))
                return
            has_base_checkpoint = True
            # dataset_status()=='ready' is itself the head of a longer
            # pipeline (upload -> process -> become the version a BRAND NEW
            # kernel session actually attaches) — a fixed grace pause here,
            # on top of the polling above, is cheap insurance against
            # exactly the failure BASE_CHECKPOINT_MOUNT's own in-kernel
            # retry (see _build_kernel_script) exists to catch.
            time.sleep(20)

        script = _build_kernel_script(self.train_flags, self.branch, has_base_checkpoint)
        # A Clone-from base gets one retry (a fresh kernel *session* against
        # the SAME already-ready dataset) if the specific, diagnosable
        # mount-miss flake above still slips through — see
        # _build_kernel_script's BASE_CHECKPOINT_MISSING check. A run with
        # no base checkpoint has nothing to retry for, so gets exactly one
        # attempt like before.
        max_attempts = 2 if has_base_checkpoint else 1
        status = None
        status_name = None
        mount_missing = False
        for attempt in range(1, max_attempts + 1):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                (tmp_path / "kernel.py").write_text(script)
                (tmp_path / "kernel-metadata.json").write_text(json.dumps({
                    "id": self.kernel_ref,
                    "title": self.slug,
                    "code_file": "kernel.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": True,
                    "enable_gpu": True,
                    # Request T4 x2 explicitly rather than leaving the accelerator
                    # to whatever Kaggle has free — a real run got handed a P100
                    # anyway (this value doesn't seem to be reliably honored by
                    # the API), which is why _build_kernel_script() also probes
                    # CUDA usability at runtime instead of trusting this request.
                    "machine_shape": "GPU_T4X2",
                    "enable_internet": True,
                    "dataset_sources": [f"{username}/{self.dataset_slug}"] if self.dataset_slug else [],
                    "competition_sources": [],
                    "kernel_sources": [],
                }))
                # Re-pushing the SAME kernel_ref (a retry) creates a new
                # version/session rather than colliding with the first —
                # exactly what "try attaching the dataset again" needs.
                api.kernels_push(str(tmp_path))

            deadline = time.time() + MAX_RUNTIME_S
            while True:
                if time.time() >= deadline:
                    self.error = f"timed out waiting for Kaggle kernel {self.kernel_ref} after {MAX_RUNTIME_S}s"
                    return
                time.sleep(POLL_INTERVAL_S)
                status = api.kernels_status(self.kernel_ref)
                status_name = _status_name(status.status)
                if status_name in TERMINAL_STATUSES:
                    break

            if status_name == "complete":
                break  # success — fall through to downloading the real result below

            mount_missing = has_base_checkpoint and self._kernel_log_has_marker(
                api, "BASE_CHECKPOINT_MISSING")
            if mount_missing and attempt < max_attempts:
                continue  # one retry: fresh kernel session, dataset already 'ready'
            self.error = status.failure_message or f"kernel finished with status '{status_name}'"
            if mount_missing:
                self.error += (
                    " — the Clone-from base checkpoint dataset never attached to the kernel "
                    "session, even after a retry (see BASE_CHECKPOINT_MISSING in its log on "
                    "kaggle.com); a transient Kaggle-side issue, not this job's training itself")
            return

        with tempfile.TemporaryDirectory() as out_dir:
            api.kernels_output(self.kernel_ref, path=out_dir, force=True, quiet=True)
            out_path = Path(out_dir)
            downloaded_result = out_path / "result.json"
            if not downloaded_result.is_file():
                self.error = "kernel completed but produced no result.json — see its log on kaggle.com"
                return
            with open(downloaded_result) as f:
                result = json.load(f)

            # Move (not leave in out_dir) the downloaded artifacts to a
            # stable job-scoped directory next to result_path — out_dir is a
            # tempdir that disappears the moment this `with` block exits,
            # but finalize_policy() (called later, from the sim thread once
            # poll() reports this job done) needs these files to still be on
            # disk when it gets around to copying them into policies/<name>/.
            dest_dir = self.result_path.parent / f"{self.job_id}_kaggle_output"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for key in ("policy_path", "train_checkpoint_path"):
                src_name = result.get(key)
                if not src_name:
                    result[key] = None
                    continue
                src = out_path / os.path.basename(src_name)
                if src.is_file():
                    dst = dest_dir / src.name
                    shutil.copyfile(src, dst)
                    result[key] = str(dst)
                else:
                    result[key] = None

            log_src = out_path / "train.log"
            if log_src.is_file():
                shutil.copyfile(log_src, self.log_path)

        with open(self.result_path, "w") as f:
            json.dump(result, f)
