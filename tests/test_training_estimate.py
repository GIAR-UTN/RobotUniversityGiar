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

from legged_gym.control.training import TrainingManager, TrainingJob


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
