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
from legged_gym.utils.props import (
    RACE_TRACK_LENGTH, ROUGH_TERRAIN_TRACK_LENGTH, ROUGH_TERRAIN_MAX_STEP,
    ROUGH_TERRAIN_BASE_HEIGHT, ROUGH_TERRAIN_TILE_SIZE, ROUGH_TERRAIN_START_GAP,
    ROUGH_TERRAIN_HEIGHT_JITTER, ROUGH_TERRAIN_SPAWN_SETBACK,
    rough_terrain_tile_heights, rough_terrain_baseline_height,
)


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_scenario_args(parser)
    return parser.parse_args(argv)


class TestScenarioRegistry(unittest.TestCase):
    def test_registers_exactly_default_ball_race_and_rough_terrain(self):
        self.assertEqual(set(SCENARIOS), {"default", "ball", "race", "rough_terrain"})

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
        # rough_terrain: same as race -- a run starts ready to go.
        self.assertTrue(SCENARIOS["rough_terrain"].ready_button_visible)
        self.assertTrue(SCENARIOS["rough_terrain"].ready_button_armed_by_default)

    def test_rough_terrain_web_options_expose_the_height_curve(self):
        # web/app.js's roughTerrainHeightAtDistance() mirrors rough_terrain_baseline_
        # height() client-side using exactly these fields -- if any go missing, the
        # web UI silently stops reporting a fall's terrain height.
        options = dict(SCENARIOS["rough_terrain"].default_options)
        web_opts = SCENARIOS["rough_terrain"].web_options(options)
        for key in ("track_length", "start_gap", "max_step", "curve_k", "base_height"):
            self.assertIn(key, web_opts)

    def test_rough_terrain_spawns_start_and_finish_lines_at_default_track_length(self):
        options = dict(SCENARIOS["rough_terrain"].default_options)
        props = SCENARIOS["rough_terrain"].spawn_props(options)
        start = next(p for p in props if p["name"] == "rough_terrain_start_line")
        finish = next(p for p in props if p["name"] == "rough_terrain_finish_line")
        self.assertEqual(start["pos"][0], 0.0)
        self.assertEqual(finish["pos"][0], -ROUGH_TERRAIN_TRACK_LENGTH)

    def test_rough_terrain_default_options_include_track_length(self):
        self.assertEqual(SCENARIOS["rough_terrain"].default_options,
                          {"track_length": ROUGH_TERRAIN_TRACK_LENGTH})

    def test_rough_terrain_tiles_are_all_climbable_and_within_bounds(self):
        props = SCENARIOS["rough_terrain"].spawn_props(dict(SCENARIOS["rough_terrain"].default_options,
                                                              seed=7))
        tiles = [p for p in props if p["name"].startswith("rough_terrain_tile_")]
        self.assertGreater(len(tiles), 0)
        for tile in tiles:
            self.assertEqual(tile["shape"], "box")
            self.assertTrue(tile["fixed"])
            self.assertNotIn("collision", tile)  # Genesis default (True) -- must stay climbable.
            height = tile["size"][2]
            self.assertGreaterEqual(height, ROUGH_TERRAIN_BASE_HEIGHT)
            self.assertLessEqual(height, ROUGH_TERRAIN_BASE_HEIGHT + ROUGH_TERRAIN_MAX_STEP)

    def test_rough_terrain_no_tile_overlaps_the_start_gap(self):
        # The first tile must appear only past ROUGH_TERRAIN_START_GAP, not on top of
        # the start line -- a tile that close to spawn used to catch the robot's foot
        # before it took a single step ("el pie queda pegado").
        props = SCENARIOS["rough_terrain"].spawn_props(dict(SCENARIOS["rough_terrain"].default_options))
        tiles = [p for p in props if p["name"].startswith("rough_terrain_tile_")]
        for tile in tiles:
            tile_near_edge_x = tile["pos"][0] + tile["size"][0] / 2  # closest edge to x=0
            self.assertLessEqual(tile_near_edge_x, -ROUGH_TERRAIN_START_GAP + 1e-9)

    def test_rough_terrain_crossing_lines_are_flush(self):
        props = SCENARIOS["rough_terrain"].spawn_props(dict(SCENARIOS["rough_terrain"].default_options))
        start_line = next(p for p in props if p["name"] == "rough_terrain_start_line")
        finish_line = next(p for p in props if p["name"] == "rough_terrain_finish_line")
        self.assertFalse(start_line["collision"])
        self.assertFalse(finish_line["collision"])
        self.assertLess(start_line["size"][2], ROUGH_TERRAIN_BASE_HEIGHT)
        self.assertLess(finish_line["size"][2], ROUGH_TERRAIN_BASE_HEIGHT)

    def test_rough_terrain_signs_are_elevated_not_ground_painted(self):
        props = SCENARIOS["rough_terrain"].spawn_props(dict(SCENARIOS["rough_terrain"].default_options))
        start_board = next(p for p in props if p["name"] == "rough_terrain_start_sign_board")
        finish_board = next(p for p in props if p["name"] == "rough_terrain_finish_sign_board")
        self.assertGreater(start_board["pos"][2], 1.0)
        self.assertGreater(finish_board["pos"][2], 1.0)
        # The lettering is paint on the board's face, same as race's ground text.
        text_props = [p for p in props if "sign_text" in p["name"]]
        self.assertGreater(len(text_props), 0)
        self.assertTrue(all(p["collision"] is False for p in text_props))

    def test_rough_terrain_start_text_contrasts_with_its_white_board(self):
        # Regression: START's lettering used to default to the same near-white color
        # as its own board and was reported invisible ("quedó blanco sobre blanco").
        props = SCENARIOS["rough_terrain"].spawn_props(dict(SCENARIOS["rough_terrain"].default_options))
        board = next(p for p in props if p["name"] == "rough_terrain_start_sign_board")
        text_props = [p for p in props if "start_sign_text" in p["name"]]
        self.assertGreater(len(text_props), 0)
        for text_prop in text_props:
            self.assertNotEqual(text_prop["color"], board["color"])

    def test_rough_terrain_has_no_crash_mat(self):
        # Requested removal: "la colchoneta azul no debe estar para este escenario".
        props = SCENARIOS["rough_terrain"].spawn_props(dict(SCENARIOS["rough_terrain"].default_options))
        self.assertFalse(any("mat" in p["name"] for p in props))


class TestRoughTerrainTileHeights(unittest.TestCase):
    """Covers rough_terrain_tile_heights()'s actual design property: tiles at the SAME
    depth (row) look similar to each other (jittered a bounded 10% around that row's
    baseline, not drawn independently) while the baseline itself rises -- the rising
    floor is what makes the track harder, not tile-to-tile randomness (see the
    function's own docstring)."""

    def test_tiles_within_a_row_stay_within_the_jitter_bound(self):
        rows, cols, heights = rough_terrain_tile_heights(seed=123)
        for row in range(rows):
            baseline = rough_terrain_baseline_height(row / (rows - 1) if rows > 1 else 1.0)
            for col in range(cols):
                if baseline == 0.0:
                    self.assertEqual(heights[row][col], 0.0)
                    continue
                self.assertLessEqual(
                    abs(heights[row][col] - baseline), baseline * ROUGH_TERRAIN_HEIGHT_JITTER + 1e-9)

    def test_tiles_within_a_row_are_within_ten_percent_of_each_other(self):
        # The actual "parecidas... hasta un 10% diferencia" property requested: at any
        # given depth, no tile should read as wildly taller/shorter than its neighbors
        # across the lane.
        rows, cols, heights = rough_terrain_tile_heights(seed=123)
        for row in range(rows):
            row_heights = heights[row]
            tallest = max(row_heights)
            if tallest == 0.0:
                continue
            for h in row_heights:
                self.assertLessEqual((tallest - h) / tallest, 2 * ROUGH_TERRAIN_HEIGHT_JITTER + 1e-9)

    def test_heights_stay_within_bounds(self):
        rows, cols, heights = rough_terrain_tile_heights(seed=123)
        for row in range(rows):
            for col in range(cols):
                self.assertGreaterEqual(heights[row][col], 0.0)
                self.assertLessEqual(heights[row][col], ROUGH_TERRAIN_MAX_STEP)

    def test_start_row_is_flat(self):
        _rows, cols, heights = rough_terrain_tile_heights(seed=123)
        self.assertTrue(all(heights[0][col] == 0.0 for col in range(cols)))

    def test_baseline_is_non_decreasing_towards_the_finish(self):
        rows, _cols, _heights = rough_terrain_tile_heights(seed=123)
        baselines = [rough_terrain_baseline_height(row / (rows - 1)) for row in range(rows)]
        self.assertEqual(baselines, sorted(baselines))
        self.assertAlmostEqual(baselines[-1], ROUGH_TERRAIN_MAX_STEP)
        self.assertAlmostEqual(baselines[0], 0.0)

    def test_baseline_is_exponential_not_linear(self):
        # The whole point of the curve requested ("algo parecido a una exponencial")
        # is that it stays low for most of the track then spikes at the end -- the
        # midpoint should be well under half of ROUGH_TERRAIN_MAX_STEP, unlike a
        # linear ramp where it would be exactly half.
        self.assertLess(rough_terrain_baseline_height(0.5), ROUGH_TERRAIN_MAX_STEP * 0.25)
        self.assertGreater(rough_terrain_baseline_height(0.9), ROUGH_TERRAIN_MAX_STEP * 0.5)

    def test_same_seed_is_reproducible(self):
        _r1, _c1, heights1 = rough_terrain_tile_heights(seed=42)
        _r2, _c2, heights2 = rough_terrain_tile_heights(seed=42)
        self.assertEqual(heights1, heights2)

    def test_tile_count_tiles_the_lane_without_overflow(self):
        rows, cols, _heights = rough_terrain_tile_heights(
            track_length=ROUGH_TERRAIN_TRACK_LENGTH, lane_width=3.2, seed=1)
        self.assertEqual(rows, int((ROUGH_TERRAIN_TRACK_LENGTH - ROUGH_TERRAIN_START_GAP) // ROUGH_TERRAIN_TILE_SIZE))
        self.assertEqual(cols, int(3.2 // ROUGH_TERRAIN_TILE_SIZE))


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

    def test_rough_terrain_with_no_overrides_uses_defaults(self):
        scenario, options = resolve_scenario(_parse(['--scenario', 'rough_terrain']))
        self.assertEqual(scenario.name, 'rough_terrain')
        self.assertEqual(options, {"track_length": ROUGH_TERRAIN_TRACK_LENGTH})

    def test_rough_terrain_scenario_option_overrides_track_length_and_accepts_seed(self):
        # seed isn't in rough_terrain's own default_options (see scenarios.py) --
        # confirms a caller can still add it via --scenario-option, same override
        # mechanism track_length itself uses.
        cli = _parse(['--scenario', 'rough_terrain',
                      '--scenario-option', 'track_length=5',
                      '--scenario-option', 'seed=42'])
        scenario, options = resolve_scenario(cli)
        self.assertEqual(options["track_length"], 5)
        self.assertEqual(options["seed"], 42)
        props = scenario.spawn_props(options)
        finish = next(p for p in props if p["name"] == "rough_terrain_finish_line")
        self.assertEqual(finish["pos"][0], -5)

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
        self.init_state = _Namespace(rot=[0.0, 0.0, 0.0, 1.0], pos=[0.0, 0.0, 1.0])
        self.env = _Namespace(fail_to_terminal_time_s=0.1)


class TestApplyScenarioToEnvCfg(unittest.TestCase):
    def test_none_scenario_is_a_noop(self):
        env_cfg = _FakeEnvCfg()
        original_rot = env_cfg.init_state.rot
        original_pos = env_cfg.init_state.pos
        original_fail_hold = env_cfg.env.fail_to_terminal_time_s
        apply_scenario_to_env_cfg(env_cfg, None, {})
        self.assertIsNone(env_cfg.props.list)
        self.assertEqual(env_cfg.init_state.rot, original_rot)
        self.assertEqual(env_cfg.init_state.pos, original_pos)
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
        original_pos = list(env_cfg.init_state.pos)
        scenario, options = resolve_scenario(_parse(['--scenario', 'race']))
        apply_scenario_to_env_cfg(env_cfg, scenario, options)
        self.assertTrue(len(env_cfg.props.list) > 0)
        self.assertEqual(env_cfg.init_state.rot, scenario.init_state_rot)
        self.assertEqual(env_cfg.env.fail_to_terminal_time_s, scenario.fail_to_terminal_time_s)
        # race doesn't override spawn position -- unlike rough_terrain, it has no
        # setback of its own.
        self.assertEqual(env_cfg.init_state.pos, original_pos)

    def test_rough_terrain_sets_props_rot_fail_hold_and_spawn_setback_by_default(self):
        env_cfg = _FakeEnvCfg()
        original_pos = list(env_cfg.init_state.pos)
        scenario, options = resolve_scenario(_parse(['--scenario', 'rough_terrain']))
        apply_scenario_to_env_cfg(env_cfg, scenario, options)
        self.assertTrue(len(env_cfg.props.list) > 0)
        self.assertEqual(env_cfg.init_state.rot, scenario.init_state_rot)
        self.assertEqual(env_cfg.env.fail_to_terminal_time_s, scenario.fail_to_terminal_time_s)
        # Spawns a bit BEHIND the start line (+x, since the track runs along -x) --
        # not on it (0) and not past it (negative) -- requested: "el robot debería
        # empezar un poquito atrás de la línea blanca, no sobre ni por delante". Only
        # x should move; y/z stay whatever the task's own default already was.
        self.assertGreater(env_cfg.init_state.pos[0], original_pos[0])
        self.assertEqual(env_cfg.init_state.pos[0], original_pos[0] + ROUGH_TERRAIN_SPAWN_SETBACK)
        self.assertEqual(env_cfg.init_state.pos[1:], original_pos[1:])

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
