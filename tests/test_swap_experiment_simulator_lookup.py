#!/usr/bin/env python3
"""
Regression test for a real bug: a policy passed to swap_experiment.py via
`--policy name:policies/<name>/checkpoint.pt` (a self-contained folder
finalize_policy() wrote) got registered with simulator="genesis" even when
its own meta.json said "isaacgym" (e.g. trained on Kaggle) — because
discover_local_policies(exclude=policy_paths.keys()) never even looks at a
name that was ALSO given via --policy, so the simulator lookup fell through
to a hardcoded default instead of reading the meta.json sitting right next
to the checkpoint. This fed a false "sources were trained on different
simulators" warning into the Fuse policies panel for any such policy pair.
See _sibling_meta_simulator() (legged_gym/scripts/swap_experiment.py).

Needs the real module (it does `from legged_gym.envs import *` at import
time), so this needs SIMULATOR set and a real simulator backend importable,
same as tests/test_control_transport.py's genesis-dependent tests.

Run directly: SIMULATOR=genesis python tests/test_swap_experiment_simulator_lookup.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legged_gym.scripts.swap_experiment import _sibling_meta_simulator


class TestSiblingMetaSimulator(unittest.TestCase):
    def test_reads_simulator_from_the_sibling_meta_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp) / "walk_gpu_c4_hient"
            policy_dir.mkdir()
            checkpoint = policy_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"fake")
            with open(policy_dir / "meta.json", "w") as f:
                json.dump({"task": "g1", "simulator": "isaacgym"}, f)

            self.assertEqual(_sibling_meta_simulator(str(checkpoint)), "isaacgym")

    def test_defaults_to_genesis_with_no_sibling_meta_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "stable.pt"
            checkpoint.write_bytes(b"fake - an external checkpoint with no meta.json at all")

            self.assertEqual(_sibling_meta_simulator(str(checkpoint)), "genesis")

    def test_defaults_to_genesis_on_malformed_meta_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp) / "broken"
            policy_dir.mkdir()
            checkpoint = policy_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"fake")
            (policy_dir / "meta.json").write_text("{not valid json")

            self.assertEqual(_sibling_meta_simulator(str(checkpoint)), "genesis")

    def test_defaults_to_genesis_when_meta_json_omits_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp) / "old_style"
            policy_dir.mkdir()
            checkpoint = policy_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"fake")
            with open(policy_dir / "meta.json", "w") as f:
                json.dump({"task": "g1"}, f)

            self.assertEqual(_sibling_meta_simulator(str(checkpoint)), "genesis")


if __name__ == "__main__":
    unittest.main()
