#!/usr/bin/env python3
"""
Unit test for TrainingManager.estimate() (legged_gym/control/training.py) —
pure history-based arithmetic, no physics backend needed. Same package-stub
trick as tests/test_push_direction.py: legged_gym/__init__.py unconditionally
imports a physics backend as a side effect of package import, even though
nothing this test touches (legged_gym.control.training) actually needs one.

Run directly: python tests/test_training_estimate.py
"""
import sys
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

from legged_gym.control.training import TrainingManager


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


if __name__ == "__main__":
    unittest.main()
