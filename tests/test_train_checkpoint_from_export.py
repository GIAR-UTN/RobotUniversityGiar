#!/usr/bin/env python3
"""
Unit tests for TrainingManager._train_checkpoint_from_export() (legged_gym/
control/training.py) — regression coverage for a real bug: a policy passed
to swap_experiment.py via `--policy name:policies/<name>/checkpoint.pt`
(rather than picked up through discover_local_policies(), which already
checks this directly) registered with train_checkpoint=None even though
policies/<name>/train_checkpoint.pt sat right next to it on disk, because
the only guess this method made was the OLDER `<log_dir>/exported/
policy_lstm_1.pt` -> `<log_dir>/model_<iter>.pt` convention. That silently
made every such policy un-fine-tunable and un-fusable. Same package-stub
trick as tests/test_delete_policy.py.

Run directly: python tests/test_train_checkpoint_from_export.py
"""
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

from legged_gym.control.training import TrainingManager


class TestTrainCheckpointFromExport(unittest.TestCase):
    def test_self_contained_policies_folder_resolves_the_sibling_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp) / "walk_gpu_c4_hient"
            policy_dir.mkdir()
            checkpoint = policy_dir / "checkpoint.pt"
            train_checkpoint = policy_dir / "train_checkpoint.pt"
            checkpoint.write_bytes(b"fake export")
            train_checkpoint.write_bytes(b"fake raw checkpoint")

            resolved = TrainingManager._train_checkpoint_from_export(str(checkpoint))

            self.assertEqual(resolved, str(train_checkpoint))

    def test_older_exported_log_dir_convention_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs" / "g1" / "some_run"
            exported_dir = log_dir / "exported"
            exported_dir.mkdir(parents=True)
            checkpoint = exported_dir / "policy_lstm_1.pt"
            checkpoint.write_bytes(b"fake export")
            (log_dir / "model_100.pt").write_bytes(b"older")
            newest = log_dir / "model_250.pt"
            newest.write_bytes(b"newest")

            resolved = TrainingManager._train_checkpoint_from_export(str(checkpoint))

            self.assertEqual(resolved, str(newest))

    def test_neither_convention_present_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "some_dir" / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fake - a fully external checkpoint like stable.pt")

            resolved = TrainingManager._train_checkpoint_from_export(str(checkpoint))

            self.assertIsNone(resolved)

    def test_none_input_returns_none(self):
        self.assertIsNone(TrainingManager._train_checkpoint_from_export(None))

    def test_sibling_file_takes_priority_over_the_older_convention(self):
        # A policies/<name>/ folder that ALSO happens to sit under a path shaped
        # like <log_dir>/exported/<file> — the sibling train_checkpoint.pt must win,
        # since it's the real file for THIS policy, not a guess.
        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp) / "exported"
            policy_dir.mkdir()
            checkpoint = policy_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"fake export")
            sibling = policy_dir / "train_checkpoint.pt"
            sibling.write_bytes(b"the real one")
            (Path(tmp) / "model_999.pt").write_bytes(b"decoy from the old convention")

            resolved = TrainingManager._train_checkpoint_from_export(str(checkpoint))

            self.assertEqual(resolved, str(sibling))


if __name__ == "__main__":
    unittest.main()
