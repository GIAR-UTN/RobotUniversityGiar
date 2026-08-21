#!/usr/bin/env python3
"""
Unit tests for TrainingManager (legged_gym/control/training.py) — pure
logic, no physics backend needed. Same package-stub trick as
tests/test_push_direction.py: legged_gym/__init__.py unconditionally
imports a physics backend as a side effect of package import, even though
nothing this test touches (legged_gym.control.training) actually needs one.

Covers: estimate() (time/iteration history arithmetic),
_train_checkpoint_from_export() (deriving a raw rsl_rl training checkpoint
from an exported/deployable one — see its own docstring for the real bug
this exists to prevent: passing the WRONG one to --from_checkpoint crashes
deep in torch's jit loader, which is exactly what happened the first time
this UI's "Clone from" was used for real), and _refresh_progress() (reading
web_train.py's mid-run progress file for a still-running job).

Run directly: python tests/test_training_estimate.py
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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

import legged_gym.control.training as training_mod
from legged_gym.control.training import (
    BACKENDS, REQUESTABLE_BACKENDS, TrainingBackend, TrainingManager, TrainingJob,
    _history_entry_backend_id,
)


def _manager_with_history(history):
    mgr = TrainingManager.__new__(TrainingManager)  # skip __init__'s JOBS_DIR/disk I/O
    mgr._history = history
    return mgr


class TestEstimate(unittest.TestCase):
    def test_no_history_returns_basis_none(self):
        mgr = _manager_with_history([])
        est = mgr.estimate(num_envs=64, max_iterations=100)
        self.assertEqual(est, {"basis": "none", "samples": 0, "seconds": None, "iterations": None})

    def test_neither_iterations_nor_minutes_returns_basis_none(self):
        mgr = _manager_with_history([{"elapsed_s": 100.0, "max_iterations": 50, "num_envs": 64}])
        est = mgr.estimate(num_envs=64)
        self.assertEqual(est["basis"], "none")

    def test_iterations_only(self):
        # rate = 100s / (50 iters * 64 envs) = 0.03125 s per (iter*env)
        mgr = _manager_with_history([{"elapsed_s": 100.0, "max_iterations": 50, "num_envs": 64}])
        est = mgr.estimate(num_envs=64, max_iterations=200)
        self.assertEqual(est["basis"], "measured")
        self.assertEqual(est["iterations"], 200)
        self.assertAlmostEqual(est["seconds"], 0.03125 * 200 * 64, places=2)

    def test_minutes_only_reverse_estimates_iterations(self):
        # Same rate as above; ask for a 10-minute (600s) budget at 64 envs.
        # iterations = budget_s / (rate * num_envs) = 600 / (0.03125*64) = 300
        mgr = _manager_with_history([{"elapsed_s": 100.0, "max_iterations": 50, "num_envs": 64}])
        est = mgr.estimate(num_envs=64, max_minutes=10)
        self.assertEqual(est["basis"], "measured")
        self.assertEqual(est["iterations"], 300)
        self.assertAlmostEqual(est["seconds"], 600.0, places=2)

    def test_both_set_whichever_is_smaller_wins(self):
        mgr = _manager_with_history([{"elapsed_s": 100.0, "max_iterations": 50, "num_envs": 64}])
        # 10 iterations costs far less time than a 10-minute budget -> iterations wins.
        est_iters_win = mgr.estimate(num_envs=64, max_iterations=10, max_minutes=10)
        self.assertEqual(est_iters_win["iterations"], 10)
        # A 1-second budget is far smaller than even 1 iteration's cost -> minutes wins.
        est_minutes_win = mgr.estimate(num_envs=64, max_iterations=10_000, max_minutes=1 / 60)
        self.assertLess(est_minutes_win["iterations"], 10_000)

    def test_median_rate_ignores_outliers(self):
        history = [
            {"elapsed_s": 10.0, "max_iterations": 10, "num_envs": 1},   # rate 1.0
            {"elapsed_s": 20.0, "max_iterations": 10, "num_envs": 1},   # rate 2.0
            {"elapsed_s": 3000.0, "max_iterations": 10, "num_envs": 1},  # rate 300.0 (outlier)
        ]
        mgr = _manager_with_history(history)
        est = mgr.estimate(num_envs=1, max_iterations=10)
        # median of [1.0, 2.0, 300.0] is 2.0 -> seconds = 2.0 * 10 * 1 = 20.0
        self.assertAlmostEqual(est["seconds"], 20.0, places=2)


def _pin_registries(mjlab_tasks, genesis_tasks):
    """Same helper as tests/test_training_backend_registry.py's -- pins what
    each registry probe reports so resolve_training_backend()/estimate()'s
    `task` disambiguation can be exercised without a real mjlab or Genesis
    venv importable in this pure-logic test process."""
    return mock.patch.multiple(
        training_mod,
        _mjlab_registered_tasks=mock.Mock(return_value=mjlab_tasks),
        _legged_gym_registered_tasks=mock.Mock(return_value=genesis_tasks),
    )


class TestEstimateBackendSimulatorGrouping(unittest.TestCase):
    """estimate()'s real point: local-genesis and local-mjlab history must
    never be pooled together even though both persist backend="local" (see
    TrainingBackend.job_backend's docstring) -- only `simulator` (recorded
    per job since this change) tells them apart. Task 1 of
    HANDOFF_mimic_motion_library_ux.md's ETA-calibration follow-up."""

    def _history(self):
        return [
            # genesis rate: 100 / (50*64) = 0.03125 s per (iter*env)
            {"task": "g1", "backend": "local", "simulator": "genesis",
             "max_iterations": 50, "num_envs": 64, "elapsed_s": 100.0},
            # mjlab rate: ~1.27 s per (iter*env), the real measured number
            # from HANDOFF's validation session -- deliberately NOT
            # hardcoded anywhere in estimate() itself, only in this fixture.
            {"task": "Rugiar-G1-Mimic", "backend": "local", "simulator": "mjlab",
             "max_iterations": 10, "num_envs": 8, "elapsed_s": 101.6},
        ]

    def test_genesis_and_mjlab_estimates_use_only_their_own_sample(self):
        mgr = _manager_with_history(self._history())
        with _pin_registries(None, {"g1"}):
            est_genesis = mgr.estimate(num_envs=64, max_iterations=100, backend="local", task="g1")
        with _pin_registries({"Rugiar-G1-Mimic"}, None):
            est_mjlab = mgr.estimate(num_envs=8, max_iterations=10, backend="local", task="Rugiar-G1-Mimic")
        self.assertEqual(est_genesis["basis"], "measured")
        self.assertEqual(est_mjlab["basis"], "measured")
        # Each bucket saw exactly its OWN history entry -- 2 total in
        # history, but pooling would make samples=2 for both.
        self.assertEqual(est_genesis["samples"], 1)
        self.assertEqual(est_mjlab["samples"], 1)
        self.assertAlmostEqual(est_genesis["seconds"], 0.03125 * 100 * 64, places=2)
        self.assertAlmostEqual(est_mjlab["seconds"], (101.6 / (10 * 8)) * 10 * 8, places=2)
        # The two regimes' per-unit rates are genuinely different (not
        # coincidentally similar, and not one masquerading as the other).
        self.assertNotAlmostEqual(est_genesis["seconds"] / (100 * 64), est_mjlab["seconds"] / (10 * 8), places=3)

    def test_no_task_falls_back_to_legacy_pooled_by_raw_backend(self):
        """Back-compat: a caller that doesn't pass `task` (every caller
        before this change) gets the OLD behavior -- everything with
        backend="local" pooled together, regardless of simulator."""
        mgr = _manager_with_history(self._history())
        est = mgr.estimate(num_envs=64, max_iterations=100, backend="local")
        self.assertEqual(est["basis"], "measured")
        self.assertEqual(est["samples"], 2)  # both entries, unfiltered by simulator

    def test_mjlab_with_no_history_yet_degrades_to_basis_none_not_genesis_numbers(self):
        """A fresh checkout with only Genesis history must never let an
        mjlab estimate silently borrow Genesis's rate."""
        mgr = _manager_with_history([self._history()[0]])  # genesis only
        with _pin_registries({"Rugiar-G1-Mimic"}, None):
            est_mjlab = mgr.estimate(num_envs=8, max_iterations=10, backend="local", task="Rugiar-G1-Mimic")
        self.assertEqual(est_mjlab, {"basis": "none", "samples": 0, "seconds": None, "iterations": None})

    def test_a_brand_new_backend_starts_feeding_its_own_bucket_automatically(self):
        """Extensibility claim: register a synthetic third backend (mirrors
        tests/test_training_backend_registry.py's 'pretend-gpu' pattern) and
        confirm estimate() buckets its history separately with ZERO changes
        to estimate()'s own code -- exactly what a future local-NVIDIA or
        second-cloud backend needs."""
        fake = TrainingBackend(
            id="local-pretend-gpu", requested_as="pretend-gpu", task_stack="genesis",
            job_backend="pretend", simulator="pretend-sim",
            command_prefix="rugiar train --backend pretend-gpu ",
            script=training_mod.TRAIN_SCRIPT,
        )
        history = self._history() + [
            {"task": "g1", "backend": "pretend", "simulator": "pretend-sim",
             "max_iterations": 5, "num_envs": 4, "elapsed_s": 2.0},  # rate: 0.1 s/(iter*env)
        ]
        mgr = _manager_with_history(history)
        with mock.patch.object(training_mod, "BACKENDS", BACKENDS + [fake]), \
                mock.patch.object(training_mod, "REQUESTABLE_BACKENDS", REQUESTABLE_BACKENDS + ("pretend-gpu",)), \
                _pin_registries(None, {"g1"}):
            est_pretend = mgr.estimate(num_envs=4, max_iterations=5, backend="pretend-gpu", task="g1")
            est_genesis = mgr.estimate(num_envs=64, max_iterations=100, backend="local", task="g1")
        self.assertEqual(est_pretend["basis"], "measured")
        self.assertEqual(est_pretend["samples"], 1)
        self.assertAlmostEqual(est_pretend["seconds"], 0.1 * 5 * 4, places=2)
        # The pretend backend's one sample never leaked into local-genesis's bucket.
        self.assertEqual(est_genesis["samples"], 1)


class TestHistoryEntryBackendId(unittest.TestCase):
    def test_resolves_known_pair(self):
        self.assertEqual(
            _history_entry_backend_id({"backend": "local", "simulator": "mjlab"}), "local-mjlab")

    def test_missing_simulator_backfills_as_genesis(self):
        """History entries written before this change (every local job
        predating mjlab training) have no "simulator" key at all -- and
        every one of them really was a Genesis run."""
        self.assertEqual(_history_entry_backend_id({"backend": "local"}), "local-genesis")

    def test_unknown_pair_returns_none(self):
        self.assertIsNone(_history_entry_backend_id({"backend": "local", "simulator": "nonsense"}))


class TestTrainCheckpointFromExport(unittest.TestCase):
    def test_none_export_path_returns_none(self):
        self.assertIsNone(TrainingManager._train_checkpoint_from_export(None))

    def test_finds_highest_iteration_sibling_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "Aug03_13-04-02_"
            export_dir = log_dir / "exported"
            export_dir.mkdir(parents=True)
            (export_dir / "policy_lstm_1.pt").write_bytes(b"fake export")
            for it in (0, 100, 1163, 950):
                (log_dir / f"model_{it}.pt").write_bytes(b"fake checkpoint")

            result = TrainingManager._train_checkpoint_from_export(str(export_dir / "policy_lstm_1.pt"))
            self.assertEqual(result, str(log_dir / "model_1163.pt"))

    def test_flat_copy_with_no_sibling_log_dir_returns_none(self):
        # Mirrors the real failure: rugiar_driver.py's --policy sources
        # point at flat copies under policies/*.pt (see docker-entrypoint.sh
        # / HANDOFF_control_web.md), not the original <log_dir>/exported/
        # structure — so there's no sibling model_*.pt to find, and this
        # must return None (safe, honest) rather than guessing wrong.
        with tempfile.TemporaryDirectory() as tmp:
            flat = Path(tmp) / "policies" / "crouch.pt"
            flat.parent.mkdir(parents=True)
            flat.write_bytes(b"fake export")
            self.assertIsNone(TrainingManager._train_checkpoint_from_export(str(flat)))

    def test_export_dir_with_no_checkpoints_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "Jul28_19-14-23_" / "exported"
            export_dir.mkdir(parents=True)
            # No model_*.pt siblings — e.g. an incomplete/failed run that
            # only ever wrote a TensorBoard events file.
            self.assertIsNone(
                TrainingManager._train_checkpoint_from_export(str(export_dir / "policy_lstm_1.pt")))

    def test_register_source_populates_train_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run"
            export_dir = log_dir / "exported"
            export_dir.mkdir(parents=True)
            (log_dir / "model_500.pt").write_bytes(b"fake checkpoint")

            mgr = TrainingManager.__new__(TrainingManager)
            mgr.policy_sources = {}
            mgr.register_source("croucher", task="g1", checkpoint=str(export_dir / "policy_lstm_1.pt"))
            self.assertEqual(mgr.policy_sources["croucher"]["train_checkpoint"], str(log_dir / "model_500.pt"))


class TestFinalizePolicyMotionFile(unittest.TestCase):
    """finalize_policy() must persist the job's exact --motion_file into the
    resulting policy's meta.json (Task 2 of HANDOFF_mimic_motion_library_ux.
    md's follow-ups) -- and must NOT invent a key when the job had none
    (any non-motion task)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_policies_dir = training_mod.POLICIES_DIR
        training_mod.POLICIES_DIR = Path(self._tmp.name)

    def tearDown(self):
        training_mod.POLICIES_DIR = self._orig_policies_dir
        self._tmp.cleanup()

    def _job(self, motion_file):
        return TrainingJob(
            id="abc", policy_name="dancer", task="Rugiar-G1-Mimic", command="rugiar train ...",
            log_path="/dev/null", result_path="/dev/null", progress_path="/dev/null",
            started_at=0.0, finished_at=1.0, max_iterations=3, max_minutes=None, num_envs=8,
            iterations_done=3, backend="local", simulator="mjlab", motion_file=motion_file,
        )

    def test_records_the_exact_motion_file_used(self):
        mgr = TrainingManager.__new__(TrainingManager)
        mgr.policy_sources = {}
        with tempfile.TemporaryDirectory() as src:
            checkpoint = Path(src) / "policy.onnx"
            checkpoint.write_bytes(b"fake onnx export")
            clip = "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"
            mgr.finalize_policy("dancer", "Rugiar-G1-Mimic", str(checkpoint), None, job=self._job(clip))

        with open(training_mod.POLICIES_DIR / "dancer" / "meta.json") as f:
            meta = json.load(f)
        self.assertEqual(meta["motion_file"], clip)

    def test_no_motion_file_on_a_non_motion_job_stays_none(self):
        mgr = TrainingManager.__new__(TrainingManager)
        mgr.policy_sources = {}
        with tempfile.TemporaryDirectory() as src:
            checkpoint = Path(src) / "policy.pt"
            checkpoint.write_bytes(b"fake checkpoint")
            mgr.finalize_policy("walker", "g1", str(checkpoint), None, job=self._job(None))

        with open(training_mod.POLICIES_DIR / "walker" / "meta.json") as f:
            meta = json.load(f)
        self.assertIsNone(meta["motion_file"])


class TestRefreshProgress(unittest.TestCase):
    def _job(self, progress_path):
        return TrainingJob(
            id="abc", policy_name="p", task="g1", command="cmd",
            log_path="/dev/null", result_path="/dev/null", progress_path=progress_path,
            started_at=0.0, max_iterations=None, max_minutes=1.0, num_envs=8,
        )

    def test_missing_progress_file_leaves_iterations_done_untouched(self):
        mgr = TrainingManager.__new__(TrainingManager)
        job = self._job("/nonexistent/path/progress.json")
        mgr._refresh_progress(job)
        self.assertIsNone(job.iterations_done)

    def test_reads_iterations_done_from_progress_file(self):
        mgr = TrainingManager.__new__(TrainingManager)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            path.write_text(json.dumps({"iterations_done": 40, "elapsed_s": 12.3, "updated_at": 1.0}))
            job = self._job(str(path))
            mgr._refresh_progress(job)
            self.assertEqual(job.iterations_done, 40)

    def test_malformed_progress_file_leaves_iterations_done_untouched(self):
        # web_train.py writes this file mid-run; a read racing a partial
        # write must degrade gracefully, not crash the sim-loop tick that
        # calls poll() -> _refresh_progress().
        mgr = TrainingManager.__new__(TrainingManager)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            path.write_text("{not valid json")
            job = self._job(str(path))
            job.iterations_done = 30  # a previous, valid read
            mgr._refresh_progress(job)
            self.assertEqual(job.iterations_done, 30)


if __name__ == "__main__":
    unittest.main()
