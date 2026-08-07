#!/usr/bin/env python3
"""
Unit tests for legged_gym/control/kaggle_backend.py's Clone-from base
checkpoint mount-miss handling — a real, reproduced Kaggle flake: a
kernel session can start with an empty /kaggle/input/<dataset>/ even after
dataset_status() reported "ready" (confirmed twice on real jobs, see
KaggleRunner._run()'s own comments). Covers:

  - _build_kernel_script(): the in-kernel fail-fast/grace-poll/diagnostic
    check it emits when (and only when) a base checkpoint is being mounted.
  - KaggleRunner._run(): the one-retry-on-detected-mount-miss policy, via a
    fake KaggleApi (no real network/kaggle package needed) that simulates
    the exact failure sequence seen in production — first kernel push
    'errors' with a BASE_CHECKPOINT_MISSING marker in its log, second push
    (the retry) 'completes' — and asserts the run ultimately succeeds
    without surfacing an error, using exactly one retry (one extra
    kernels_push call).

Same package-stub trick as test_training_estimate.py/test_push_direction.py
— legged_gym/__init__.py unconditionally requires a SIMULATOR env var and
imports a physics backend, neither of which kaggle_backend.py itself needs
(its one real dependency, the `kaggle` package, is imported lazily inside
KaggleRunner._run() — stubbed out here with a fake instead of installed).

Run directly: python tests/test_kaggle_backend.py
"""
import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_package(dotted_name: str):
    if dotted_name in sys.modules:
        return
    module = types.ModuleType(dotted_name)
    module.__path__ = [str(ROOT / Path(*dotted_name.split(".")))]
    module.__package__ = dotted_name
    sys.modules[dotted_name] = module


for _pkg in ("legged_gym", "legged_gym.control"):
    _stub_package(_pkg)

import legged_gym.control.kaggle_backend as kaggle_backend
from legged_gym.control.kaggle_backend import KaggleRunner, _build_kernel_script


class TestBuildKernelScript(unittest.TestCase):
    def test_no_base_checkpoint_omits_the_mount_check_entirely(self):
        script = _build_kernel_script(["--task", "g1"], "main", False)
        ast.parse(script)  # must still be valid, runnable Python
        self.assertNotIn("BASE_CHECKPOINT_MISSING", script)

    def test_base_checkpoint_adds_a_fail_fast_mount_check(self):
        script = _build_kernel_script(["--task", "g1"], "main", True)
        ast.parse(script)
        self.assertIn("BASE_CHECKPOINT_MISSING", script)
        # Must glob for the checkpoint by name rather than assume a fixed
        # nesting depth under /kaggle/input — the whole point of this fix
        # (see HANDOFF_kaggle_clone_from_mount_bug.md): a real failed kernel
        # showed Kaggle nests API-pushed dataset sources one level deeper
        # (/kaggle/input/datasets/<owner>/<slug>/...) than the flat
        # /kaggle/input/<slug>/... path this code used to hardcode.
        self.assertIn("glob.glob('/kaggle/input/**/train_checkpoint.pt', recursive=True)", script)
        # The check must run before the multi-minute venv/Isaac Gym
        # bootstrap (the whole point is failing fast, not after wasting
        # GPU-minutes) — assert its ordering relative to a bootstrap
        # landmark line.
        self.assertLess(script.index("BASE_CHECKPOINT_MISSING"), script.index("isaac-gym-preview-4"))


class _FakeStatus:
    def __init__(self, name, failure_message=None):
        self.status = types.SimpleNamespace(name=name)
        self.failure_message = failure_message


class _FakeKaggleApi:
    """Simulates exactly the production failure sequence: the first
    kernels_push 'session' never sees the mounted dataset (status ERROR,
    its log carries the BASE_CHECKPOINT_MISSING marker _build_kernel_script
    would have printed), the retry's session succeeds (status COMPLETE with
    a real result.json)."""

    def __init__(self):
        self.config_values = {"username": "tester"}
        self.dataset_create_new_calls = 0
        self.kernels_push_calls = 0
        self.dataset_status_calls = 0

    def authenticate(self):
        pass

    def dataset_create_new(self, *a, **kw):
        self.dataset_create_new_calls += 1

    def dataset_status(self, ref):
        self.dataset_status_calls += 1
        return "ready"

    def kernels_push(self, path):
        self.kernels_push_calls += 1

    def kernels_status(self, ref):
        if self.kernels_push_calls == 1:
            return _FakeStatus("ERROR")
        return _FakeStatus("COMPLETE")

    def kernels_output(self, ref, path, force=True, quiet=True):
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if self.kernels_push_calls == 1:
            # What _build_kernel_script's fail-fast check would have
            # printed to the kernel's own captured stdout/stderr.
            (out / "legged-gym-ex-test123.log").write_text(
                "BASE_CHECKPOINT_MISSING: /kaggle/input/legged-gym-ex-test123-base/train_checkpoint.pt\n")
        else:
            (out / "result.json").write_text(json.dumps({
                "policy_path": "checkpoint.pt",
                "train_checkpoint_path": "train_checkpoint.pt",
            }))
            (out / "checkpoint.pt").write_bytes(b"fake exported policy")
            (out / "train_checkpoint.pt").write_bytes(b"fake raw checkpoint")
            (out / "train.log").write_text("fake training log\n")


class TestKaggleRunnerRetriesOnceOnMountMiss(unittest.TestCase):
    def setUp(self):
        # KaggleRunner._run() imports `from kaggle.api.kaggle_api_extended
        # import KaggleApi` lazily — stub that module tree with our fake
        # instead of requiring the real `kaggle` package/credentials.
        self._fake_api = _FakeKaggleApi()
        fake_module = types.ModuleType("kaggle.api.kaggle_api_extended")
        fake_module.KaggleApi = lambda: self._fake_api
        sys.modules["kaggle"] = types.ModuleType("kaggle")
        sys.modules["kaggle.api"] = types.ModuleType("kaggle.api")
        sys.modules["kaggle.api.kaggle_api_extended"] = fake_module

        # time.sleep is real inside kaggle_backend (POLL_INTERVAL_S=15,
        # plus the 20s post-'ready' grace pause) — patch the module's own
        # reference so every test run stays fast without changing the
        # module's actual polling *logic*, only its wall-clock cost.
        self._real_sleep = kaggle_backend.time.sleep
        kaggle_backend.time.sleep = lambda _s: None
        self.addCleanup(setattr, kaggle_backend.time, "sleep", self._real_sleep)

    def test_retries_once_and_succeeds_after_a_simulated_mount_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_checkpoint = tmp_path / "base_checkpoint.pt"
            base_checkpoint.write_bytes(b"fake base checkpoint")
            result_path = tmp_path / "jobs" / "test123.result.json"
            log_path = tmp_path / "jobs" / "test123.log"
            result_path.parent.mkdir(parents=True)

            runner = KaggleRunner(
                job_id="test123", train_flags=["--task", "g1"],
                result_path=result_path, log_path=log_path,
                base_checkpoint_path=base_checkpoint,
            )
            runner._run()

            self.assertIsNone(runner.error, f"unexpected error: {runner.error}")
            # Exactly one retry: two kernel sessions total, one dataset
            # upload (the SAME already-'ready' dataset is reused, not
            # re-uploaded for the retry).
            self.assertEqual(self._fake_api.kernels_push_calls, 2)
            self.assertEqual(self._fake_api.dataset_create_new_calls, 1)

            self.assertTrue(result_path.is_file())
            with open(result_path) as f:
                result = json.load(f)
            self.assertTrue(Path(result["policy_path"]).is_file())
            self.assertTrue(Path(result["train_checkpoint_path"]).is_file())

    def test_gives_up_with_a_clear_error_if_the_retry_also_mount_misses(self):
        class _AlwaysMissingApi(_FakeKaggleApi):
            def kernels_status(self, ref):
                return _FakeStatus("ERROR")  # every attempt errors, not just the first

            def kernels_output(self, ref, path, force=True, quiet=True):
                out = Path(path)
                out.mkdir(parents=True, exist_ok=True)
                (out / "legged-gym-ex-test456.log").write_text(
                    "BASE_CHECKPOINT_MISSING: /kaggle/input/legged-gym-ex-test456-base/train_checkpoint.pt\n")

        fake_api = _AlwaysMissingApi()
        sys.modules["kaggle.api.kaggle_api_extended"].KaggleApi = lambda: fake_api

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_checkpoint = tmp_path / "base_checkpoint.pt"
            base_checkpoint.write_bytes(b"fake base checkpoint")
            result_path = tmp_path / "jobs" / "test456.result.json"
            log_path = tmp_path / "jobs" / "test456.log"
            result_path.parent.mkdir(parents=True)

            runner = KaggleRunner(
                job_id="test456", train_flags=["--task", "g1"],
                result_path=result_path, log_path=log_path,
                base_checkpoint_path=base_checkpoint,
            )
            runner._run()

            self.assertIsNotNone(runner.error)
            self.assertIn("attach", runner.error)
            # Exactly one retry attempted (2 pushes total), then gives up —
            # must NOT loop forever re-pushing kernels on a truly-broken dataset.
            self.assertEqual(fake_api.kernels_push_calls, 2)
            self.assertFalse(result_path.is_file())

    def test_no_base_checkpoint_gets_a_single_attempt_no_retry_logic(self):
        class _SingleErrorApi(_FakeKaggleApi):
            def kernels_status(self, ref):
                return _FakeStatus("ERROR", failure_message="boom")

        fake_api = _SingleErrorApi()
        sys.modules["kaggle.api.kaggle_api_extended"].KaggleApi = lambda: fake_api

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result_path = tmp_path / "jobs" / "test789.result.json"
            log_path = tmp_path / "jobs" / "test789.log"
            result_path.parent.mkdir(parents=True)

            runner = KaggleRunner(
                job_id="test789", train_flags=["--task", "g1"],
                result_path=result_path, log_path=log_path,
                base_checkpoint_path=None,  # no Clone-from base -> nothing to retry for
            )
            runner._run()

            self.assertEqual(runner.error, "boom")
            self.assertEqual(fake_api.kernels_push_calls, 1)
            self.assertEqual(fake_api.dataset_create_new_calls, 0)


if __name__ == "__main__":
    unittest.main()
