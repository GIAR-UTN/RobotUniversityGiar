#!/usr/bin/env python3
"""
Unit tests for TrainingManager.VARIABLE_REGISTRY (legged_gym/control/
training.py) — pure data/signature checks, no physics backend or
task_registry needed (same package-stub trick as test_training_estimate.py).

These encode the invariant the "Target variable" UI mechanism depends on
end to end (see app.js's composeTrainingParams()/refreshTargetReference()):
every VARIABLE_REGISTRY entry's "flag" must be the EXACT kwarg name
TrainingManager.start() and ControlService.start_training() both accept —
that's what lets the frontend send `{[meta.flag]: value}` generically
instead of hardcoding one variable's key. A registry entry added without
matching that wiring would silently no-op (the frontend would send a kwarg
neither method recognizes) rather than error, which is exactly the kind of
mistake worth catching here instead of on a wasted Kaggle GPU run.

Run directly: python tests/test_variable_registry.py
"""
import inspect
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
from legged_gym.control.service import ControlService

EXPECTED_KEYS = {"base_height", "lin_vel_z", "ang_vel_xy", "orientation_tilt"}
REQUIRED_META_FIELDS = {"label", "unit", "source", "flag", "config_attr", "range_attr", "note"}
VALID_SOURCES = {"sim_ground_truth", "sensor"}


class TestVariableRegistryStructure(unittest.TestCase):
    def test_has_exactly_the_expected_variables(self):
        # No new terms invented beyond what Live Telemetry actually shows
        # (base_height, lin_vel_z, ang_vel_xy, orientation_tilt) — a 5th key
        # here should be a deliberate addition, not a typo/duplicate.
        self.assertEqual(set(TrainingManager.VARIABLE_REGISTRY.keys()), EXPECTED_KEYS)

    def test_every_entry_has_all_required_fields(self):
        for key, meta in TrainingManager.VARIABLE_REGISTRY.items():
            missing = REQUIRED_META_FIELDS - meta.keys()
            self.assertFalse(missing, f"'{key}' is missing fields: {missing}")

    def test_every_entry_source_is_a_known_value(self):
        # app.js's renderVariableChrome() branches on exactly these two
        # strings ("sensor" -> "Real sensor", else -> "Simulator ground
        # truth") — an unrecognized value would silently mislabel a variable.
        for key, meta in TrainingManager.VARIABLE_REGISTRY.items():
            self.assertIn(meta["source"], VALID_SOURCES, f"'{key}' has an unrecognized source")

    def test_flag_equals_config_attr_by_convention(self):
        # Not a hard technical requirement, but every current entry follows
        # it and app.js's command-preview text assumes flag names read like
        # the config attribute they set — catches an entry that drifts from
        # this repo's own established naming convention.
        for key, meta in TrainingManager.VARIABLE_REGISTRY.items():
            self.assertEqual(meta["flag"], meta["config_attr"], f"'{key}': flag != config_attr")

    def test_no_variable_key_is_also_a_reward_scale_note(self):
        # REWARD_SCALE_NOTES documents WEIGHT-only terms explicitly kept out
        # of VARIABLE_REGISTRY (see its own comment in training.py) — a term
        # in both would mean the UI offers it as a fixed target AND
        # describes it as target-less, a direct self-contradiction.
        overlap = set(TrainingManager.VARIABLE_REGISTRY.keys()) & set(TrainingManager.REWARD_SCALE_NOTES.keys())
        self.assertEqual(overlap, set())


class TestVariableRegistryFlagsAreWiredThrough(unittest.TestCase):
    """Every registry entry's flag/config_attr must be an accepted kwarg on
    BOTH TrainingManager.start() (the actual CLI-flag emitter) and
    ControlService.start_training() (the RPC layer app.js's call()
    reaches) — the two-hop path a value takes from the "Target variable"
    dropdown down to web_train.py's argparse."""

    def test_start_accepts_every_registered_flag_as_a_kwarg(self):
        start_params = set(inspect.signature(TrainingManager.start).parameters)
        for key, meta in TrainingManager.VARIABLE_REGISTRY.items():
            self.assertIn(meta["flag"], start_params,
                          f"'{key}': TrainingManager.start() has no '{meta['flag']}' parameter")

    def test_start_training_rpc_accepts_every_registered_flag_as_a_kwarg(self):
        rpc_params = set(inspect.signature(ControlService.start_training).parameters)
        for key, meta in TrainingManager.VARIABLE_REGISTRY.items():
            self.assertIn(meta["flag"], rpc_params,
                          f"'{key}': ControlService.start_training() has no '{meta['flag']}' parameter")


if __name__ == "__main__":
    unittest.main()
