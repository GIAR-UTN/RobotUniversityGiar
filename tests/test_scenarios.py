#!/usr/bin/env python3
"""
Unit tests for legged_gym/utils/scenarios.py -- the Scenario registry that
replaced the old independent --ball/--race booleans with a single
--scenario={default,ball,race} choice, defaulting to 'default' (full admin)
when omitted (see legged_gym/utils/scenarios.py's module docstring).
Pure-Python module (no Genesis/mjlab import chain), so this runs under any
SIMULATOR value.

Run directly: python tests/test_scenarios.py
"""
import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legged_gym.utils.scenarios import (
    SCENARIOS, add_scenario_args, resolve_scenario, apply_scenario_to_env_cfg,
)
from legged_gym.utils.props import RACE_TRACK_LENGTH


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_scenario_args(parser)
    return parser.parse_args(argv)


class TestScenarioRegistry(unittest.TestCase):
    def test_registers_exactly_default_ball_and_race(self):
        self.assertEqual(set(SCENARIOS), {"default", "ball", "race"})

    def test_default_spawns_no_props(self):
        self.assertEqual(SCENARIOS["default"].spawn_props({}), [])

    def test_ball_spawns_one_ball_prop(self):
        props = SCENARIOS["ball"].spawn_props({})
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0]["name"], "ball")

    def test_race_spawns_start_and_finish_lines_at_default_track_length(self):
        options = dict(SCENARIOS["race"].default_options)
        props = SCENARIOS["race"].spawn_props(options)
        start = next(p for p in props if p["name"] == "race_start_line")
        finish = next(p for p in props if p["name"] == "race_finish_line")
        self.assertEqual(start["pos"][0], 0.0)
        self.assertEqual(finish["pos"][0], -RACE_TRACK_LENGTH)

    def test_race_default_options_include_track_length(self):
        self.assertEqual(SCENARIOS["race"].default_options, {"track_length": RACE_TRACK_LENGTH})

    def test_ready_button_visibility_and_default_armed_state_per_scenario(self):
        # default: full admin, sees everything, but doesn't self-arm (launching admin
        # mode shouldn't itself trigger a restart nobody asked for).
        self.assertTrue(SCENARIOS["default"].ready_button_visible)
        self.assertFalse(SCENARIOS["default"].ready_button_armed_by_default)
        # ball: available if you want it, not on by default.
        self.assertTrue(SCENARIOS["ball"].ready_button_visible)
        self.assertFalse(SCENARIOS["ball"].ready_button_armed_by_default)
        # race: visible and pre-armed -- a race run starts ready to go.
        self.assertTrue(SCENARIOS["race"].ready_button_visible)
        self.assertTrue(SCENARIOS["race"].ready_button_armed_by_default)


class TestAddScenarioArgs(unittest.TestCase):
    def test_no_scenario_flag_defaults_to_default_scenario(self):
        # An operator who passes nothing lands in the named 'default' (full-admin)
        # scenario, not an unnamed absence of one -- see add_scenario_args's docstring.
        cli = _parse([])
        self.assertEqual(cli.scenario, 'default')
        self.assertEqual(cli.scenario_option, [])

    def test_unknown_scenario_name_rejected_by_choices(self):
        with self.assertRaises(SystemExit):
            _parse(['--scenario', 'not-a-real-scenario'])

    def test_scenarios_are_mutually_exclusive_by_construction(self):
        # A single --scenario flag (not independent booleans) means there is no argv
        # spelling that selects two at once -- the last one given wins.
        cli = _parse(['--scenario', 'ball', '--scenario', 'race'])
        self.assertEqual(cli.scenario, 'race')


class TestResolveScenario(unittest.TestCase):
    def test_no_scenario_flag_resolves_to_the_default_scenario(self):
        scenario, options = resolve_scenario(_parse([]))
        self.assertEqual(scenario.name, 'default')
        self.assertEqual(options, {})

    def test_explicit_none_scenario_resolves_to_none_and_empty_options(self):
        # Not reachable from the CLI (add_scenario_args always sets a real default), but
        # still the correct answer for a caller that builds/passes its own Namespace.
        cli = _parse([])
        cli.scenario = None
        scenario, options = resolve_scenario(cli)
        self.assertIsNone(scenario)
        self.assertEqual(options, {})

    def test_scenario_with_no_overrides_uses_defaults(self):
        scenario, options = resolve_scenario(_parse(['--scenario', 'race']))
        self.assertEqual(scenario.name, 'race')
        self.assertEqual(options, {"track_length": RACE_TRACK_LENGTH})

    def test_scenario_option_overrides_default_and_coerces_numeric(self):
        cli = _parse(['--scenario', 'race', '--scenario-option', 'track_length=10'])
        scenario, options = resolve_scenario(cli)
        self.assertEqual(options["track_length"], 10)
        self.assertIsInstance(options["track_length"], int)

    def test_scenario_option_without_equals_raises(self):
        cli = _parse(['--scenario', 'race', '--scenario-option', 'track_length'])
        with self.assertRaises(ValueError):
            resolve_scenario(cli)

    def test_scenario_option_applies_even_without_an_explicit_scenario(self):
        # cli.scenario defaults to 'default' (not None), so a bare --scenario-option
        # with no --scenario still resolves against a real scenario ('default', which
        # has no default_options of its own) -- the override still lands, just with
        # nothing to merge over.
        cli = _parse(['--scenario-option', 'track_length=10'])
        scenario, options = resolve_scenario(cli)
        self.assertEqual(scenario.name, 'default')
        self.assertEqual(options, {"track_length": 10})


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeEnvCfg:
    """Minimal stand-in for the real env_cfg (a dataclass tree) -- only the
    attributes apply_scenario_to_env_cfg() actually touches. Per-instance
    (not nested classes) so mutating one test's env_cfg.props.list can't
    leak into another test via a shared class attribute."""
    def __init__(self):
        self.props = _Namespace(list=None)
        self.init_state = _Namespace(rot=[0.0, 0.0, 0.0, 1.0])
        self.env = _Namespace(fail_to_terminal_time_s=0.1)


class TestApplyScenarioToEnvCfg(unittest.TestCase):
    def test_none_scenario_is_a_noop(self):
        env_cfg = _FakeEnvCfg()
        original_rot = env_cfg.init_state.rot
        original_fail_hold = env_cfg.env.fail_to_terminal_time_s
        apply_scenario_to_env_cfg(env_cfg, None, {})
        self.assertIsNone(env_cfg.props.list)
        self.assertEqual(env_cfg.init_state.rot, original_rot)
        self.assertEqual(env_cfg.env.fail_to_terminal_time_s, original_fail_hold)

    def test_ball_sets_props_only(self):
        env_cfg = _FakeEnvCfg()
        original_rot = env_cfg.init_state.rot
        original_fail_hold = env_cfg.env.fail_to_terminal_time_s
        apply_scenario_to_env_cfg(env_cfg, SCENARIOS["ball"], {})
        self.assertEqual(len(env_cfg.props.list), 1)
        self.assertEqual(env_cfg.init_state.rot, original_rot)
        self.assertEqual(env_cfg.env.fail_to_terminal_time_s, original_fail_hold)

    def test_race_sets_props_rot_and_fail_hold_by_default(self):
        env_cfg = _FakeEnvCfg()
        scenario, options = resolve_scenario(_parse(['--scenario', 'race']))
        apply_scenario_to_env_cfg(env_cfg, scenario, options)
        self.assertTrue(len(env_cfg.props.list) > 0)
        self.assertEqual(env_cfg.init_state.rot, scenario.init_state_rot)
        self.assertEqual(env_cfg.env.fail_to_terminal_time_s, scenario.fail_to_terminal_time_s)

    def test_apply_fail_hold_false_skips_fail_hold_but_keeps_rot(self):
        # play.py's override_configs() never applied fail_to_terminal_time_s even for
        # the old --race (only the driver scripts did) -- apply_fail_hold=False
        # preserves that distinction for the unified scenario path too.
        env_cfg = _FakeEnvCfg()
        original_fail_hold = env_cfg.env.fail_to_terminal_time_s
        scenario, options = resolve_scenario(_parse(['--scenario', 'race']))
        apply_scenario_to_env_cfg(env_cfg, scenario, options, apply_fail_hold=False)
        self.assertEqual(env_cfg.init_state.rot, scenario.init_state_rot)
        self.assertEqual(env_cfg.env.fail_to_terminal_time_s, original_fail_hold)


if __name__ == "__main__":
    unittest.main()
