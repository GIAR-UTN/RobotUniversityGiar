"""Scenario registry: the one place `--scenario` (play.py, rugiar_driver.py,
rugiar_driver_target.py) gets its list of names, its default per-scenario
options, and the props/init-state it wires into env_cfg.

A scenario bundles what used to be two independent `--ball`/`--race`
booleans into one named, mutually-exclusive choice, each with its own
default options (overridable per-run via `--scenario-option KEY=VALUE`,
never by editing this file) and, on the web side, its own panel order (see
web/app.js's SCENARIO_DEFAULT_ORDERS).
"""

import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional

from legged_gym.utils.props import (
    default_ball_prop, default_race_props, RACE_SPAWN_ROT, RACE_TRACK_LENGTH, RACE_FAIL_HOLD_S,
    default_rough_terrain_props, ROUGH_TERRAIN_TRACK_LENGTH, ROUGH_TERRAIN_START_GAP,
    ROUGH_TERRAIN_MAX_STEP, ROUGH_TERRAIN_STEP_CURVE_K, ROUGH_TERRAIN_BASE_HEIGHT,
    ROUGH_TERRAIN_SPAWN_SETBACK,
)
from legged_gym.utils.competition_props import (
    default_factory_handling_props, default_factory_sorting_props,
    default_hospital_pharmacy_props, default_hospital_dispensing_props,
    default_hotel_reception_props, default_hotel_cleaning_props,
    default_warehouse_sorting_props, default_obstacle_course_props,
    OBSTACLE_COURSE_SPAWN_SETBACK,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    # Takes the merged options dict (defaults + --scenario-option overrides), returns the
    # env_cfg.props.list this scenario spawns.
    spawn_props: Callable[[dict], list]
    default_options: dict = field(default_factory=dict)
    # env_cfg.init_state.rot override, if this scenario needs the robot to spawn facing
    # a particular direction (e.g. race, facing down the track). None = don't touch it.
    init_state_rot: Optional[list] = None
    # [dx, dy, dz] ADDED to whatever env_cfg.init_state.pos already is (not a
    # replacement -- the task's own default z/height stays whatever it was, only the
    # scenario-relevant axes shift), if this scenario needs the robot to spawn
    # somewhere other than the task's default spawn point (e.g. rough_terrain,
    # a small +x setback so it starts short of the start line, not on/past it).
    # None = don't touch it.
    init_state_pos_offset: Optional[list] = None
    # env_cfg.env.fail_to_terminal_time_s override, if this scenario needs a fall to stay
    # on screen longer than the training default (e.g. race, so a crash is visible until
    # the operator hits Restart). None = don't touch it.
    fail_to_terminal_time_s: Optional[float] = None
    # Merged options -> the dict the /config route hands the web UI as "scenario_options"
    # (e.g. race's track_length, so the client can compute finish-line distance).
    web_options: Callable[[dict], dict] = lambda opts: {}
    # Whether the web UI's "321 Ready!" countdown/timer button shows at all for this
    # scenario, and whether it starts pre-armed (auto-triggers a restart + countdown on
    # page load, same as a manual click -- see web/app.js's initRaceMode()/armRace()).
    # The countdown/timer mechanism itself isn't race-specific (finish-line auto-detection
    # already no-ops when there's no track to measure against -- see finishRace()'s
    # raceTrackLength != null guard), so any scenario can opt into showing it.
    ready_button_visible: bool = False
    ready_button_armed_by_default: bool = False


# Computed once, not per-request -- default_obstacle_course_props() is deterministic
# (no seed/randomization, unlike default_rough_terrain_props()), so there's nothing to
# recompute on every 'obstacle_course' spawn_props()/web_options() call.
_OBSTACLE_COURSE_PROPS, _OBSTACLE_COURSE_LENGTH = default_obstacle_course_props()


SCENARIOS: dict = {
    "default": Scenario(
        name="default",
        # Full admin: no props, nothing about env_cfg touched -- same "nothing extra"
        # behavior "no --scenario at all" used to have, now a named, /config-visible
        # entry instead of a bare None. Sees everything (ready button visible), but
        # doesn't auto-arm it -- launching in admin mode shouldn't itself trigger a
        # restart nobody asked for.
        spawn_props=lambda opts: [],
        ready_button_visible=True,
    ),
    "ball": Scenario(
        name="ball",
        spawn_props=lambda opts: [default_ball_prop()],
        ready_button_visible=True,
    ),
    "race": Scenario(
        name="race",
        default_options={"track_length": RACE_TRACK_LENGTH},
        spawn_props=lambda opts: default_race_props(track_length=opts["track_length"]),
        init_state_rot=RACE_SPAWN_ROT,
        fail_to_terminal_time_s=RACE_FAIL_HOLD_S,
        web_options=lambda opts: {"track_length": opts["track_length"]},
        ready_button_visible=True,
        ready_button_armed_by_default=True,
    ),
    "rough_terrain": Scenario(
        name="rough_terrain",
        # Same corridor as 'race' (start/finish crossing lines this same track_length
        # apart), but the flat lane is replaced by a field of paver tiles that get
        # rougher towards the finish -- see default_rough_terrain_props()'s docstring.
        default_options={"track_length": ROUGH_TERRAIN_TRACK_LENGTH},
        spawn_props=lambda opts: default_rough_terrain_props(
            track_length=opts["track_length"], seed=opts.get("seed")),
        init_state_rot=RACE_SPAWN_ROT,
        # A small +x setback so the robot starts a step short of the start line, not
        # standing on top of it (0.0) or already past it -- requested: "el robot
        # debería empezar un poquito atrás de la línea blanca, no sobre ni por delante".
        init_state_pos_offset=[ROUGH_TERRAIN_SPAWN_SETBACK, 0.0, 0.0],
        fail_to_terminal_time_s=RACE_FAIL_HOLD_S,
        # start_gap/max_step/curve_k/base_height let the web UI mirror
        # rough_terrain_baseline_height() client-side (see onRoughTerrainFall() in
        # web/app.js) to report the terrain height reached at the moment of a fall --
        # requested alongside distance/time ("indicando la distancia, la altura y el
        # tiempo que demoró").
        web_options=lambda opts: {
            "track_length": opts["track_length"],
            "start_gap": ROUGH_TERRAIN_START_GAP,
            "max_step": ROUGH_TERRAIN_MAX_STEP,
            "curve_k": ROUGH_TERRAIN_STEP_CURVE_K,
            "base_height": ROUGH_TERRAIN_BASE_HEIGHT,
        },
        ready_button_visible=True,
        ready_button_armed_by_default=True,
    ),
    # -- World Humanoid Robot Games (Beijing) 场景赛 scenarios -- static furniture
    # layouts for navigation/manipulation practice, digitized from the official
    # rulebooks (see legged_gym/utils/competition_props.py's module docstring for
    # sources). init_state_rot=RACE_SPAWN_ROT here (same as 'race') pairs with
    # each props function's own _rotate_yaw180() -- robot and furniture spin
    # together, 180 degrees from this repo's plain +x-facing default. Requested
    # directly: with the default facing, furniture sat between the robot and the
    # viser viewer's (currently fixed) default camera position, hiding the robot.
    "factory_handling": Scenario(
        name="factory_handling",
        spawn_props=lambda opts: default_factory_handling_props(),
        init_state_rot=RACE_SPAWN_ROT,
        ready_button_visible=True,
    ),
    "factory_sorting": Scenario(
        name="factory_sorting",
        spawn_props=lambda opts: default_factory_sorting_props(),
        init_state_rot=RACE_SPAWN_ROT,
        ready_button_visible=True,
    ),
    "hospital_pharmacy": Scenario(
        name="hospital_pharmacy",
        spawn_props=lambda opts: default_hospital_pharmacy_props(),
        init_state_rot=RACE_SPAWN_ROT,
        ready_button_visible=True,
    ),
    "hospital_dispensing": Scenario(
        name="hospital_dispensing",
        spawn_props=lambda opts: default_hospital_dispensing_props(),
        init_state_rot=RACE_SPAWN_ROT,
        ready_button_visible=True,
    ),
    "hotel_reception": Scenario(
        name="hotel_reception",
        spawn_props=lambda opts: default_hotel_reception_props(),
        init_state_rot=RACE_SPAWN_ROT,
        ready_button_visible=True,
    ),
    "hotel_cleaning": Scenario(
        name="hotel_cleaning",
        spawn_props=lambda opts: default_hotel_cleaning_props(),
        init_state_rot=RACE_SPAWN_ROT,
        ready_button_visible=True,
    ),
    "warehouse_sorting": Scenario(
        name="warehouse_sorting",
        spawn_props=lambda opts: default_warehouse_sorting_props(),
        init_state_rot=RACE_SPAWN_ROT,
        ready_button_visible=True,
    ),
    # -- The 100m-obstacle event: the harder ALTERNATIVE to plain 'race' (added
    # last, deliberately, once every task-arena scenario above was in) -- same
    # start/finish-line, -x-facing corridor convention as 'race'/'rough_terrain',
    # but with the event's own 10 real obstacles instead of an open lane or
    # randomized tiles. total_length isn't random (no seed, unlike
    # rough_terrain) so it's computed once at import time.
    "obstacle_course": Scenario(
        name="obstacle_course",
        spawn_props=lambda opts: _OBSTACLE_COURSE_PROPS,
        init_state_rot=RACE_SPAWN_ROT,
        # +x setback (same mechanism as rough_terrain's own) so the robot starts a
        # step short of the start line -- the first obstacle sits right at x=0 with
        # no clearance, and feet kept burying into its raised edge without this.
        init_state_pos_offset=[OBSTACLE_COURSE_SPAWN_SETBACK, 0.0, 0.0],
        fail_to_terminal_time_s=RACE_FAIL_HOLD_S,
        web_options=lambda opts: {"track_length": _OBSTACLE_COURSE_LENGTH},
        ready_button_visible=True,
        ready_button_armed_by_default=True,
    ),
}


def add_scenario_args(parser: argparse.ArgumentParser) -> None:
    """Adds --scenario/--scenario-option to `parser`. The single place these two flags
    are defined -- every CLI entry point (play.py via helpers.py, rugiar_driver.py,
    rugiar_driver_target.py) calls this instead of declaring its own --ball/--race.

    Defaults to 'default' (not None/absent) -- an operator who passes nothing gets the
    named admin scenario (everything visible, no props), not an unnamed absence of one."""
    parser.add_argument('--scenario', type=str, default='default', choices=sorted(SCENARIOS),
                         help="which named scenario's props/scenery/web-UI config to use "
                              "(Genesis/viser only, for now -- sim scenery, not part of "
                              "training: cfg.props.list stays empty unless a caller opts "
                              "in). 'default': no props, full admin web UI (everything "
                              "visible). 'ball': a physics-enabled ball prop. 'race': a "
                              "start line at the robot's spawn, a finish line down the "
                              "track, and a crash-mat wall to run into -- see "
                              "legged_gym/utils/props.py::default_race_props(). "
                              "'rough_terrain': same start/finish corridor as 'race', "
                              "but paved with tiles that get rougher towards the finish "
                              "-- see "
                              "legged_gym/utils/props.py::default_rough_terrain_props(). "
                              "'factory_handling'/'factory_sorting'/'hospital_pharmacy'/"
                              "'hospital_dispensing'/'hotel_reception'/'hotel_cleaning'/"
                              "'warehouse_sorting': task-arena furniture layouts digitized "
                              "from the World Humanoid Robot Games (Beijing) rulebooks -- "
                              "see legged_gym/utils/competition_props.py. "
                              "'obstacle_course': that same event's 100m-obstacle track, "
                              "the harder alternative to 'race' -- 10 real obstacles down "
                              "the same start/finish corridor -- see "
                              "legged_gym/utils/competition_props.py::"
                              "default_obstacle_course_props(). "
                              "See legged_gym/utils/scenarios.py::SCENARIOS.")
    parser.add_argument('--scenario-option', action='append', default=[], metavar='KEY=VALUE',
                         dest='scenario_option',
                         help="override one of the selected scenario's default options, e.g. "
                              "--scenario-option track_length=10 (repeatable). Ignored if "
                              "--scenario isn't set.")


def _coerce(value: str):
    """int/float-coerces a --scenario-option value when it looks numeric, else leaves it
    a string -- e.g. 'track_length=10' -> 10 (int), matching RACE_TRACK_LENGTH's own type
    family (float) closely enough for arithmetic, while a hypothetical string option
    stays a string untouched."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def resolve_scenario(cli: argparse.Namespace):
    """Returns (Scenario_or_None, merged_options_dict) for `cli.scenario`/`cli.scenario_option`
    (as set up by add_scenario_args). None, {} for a falsy cli.scenario -- in practice
    add_scenario_args's own 'default' default means the CLI path never hits this, but it's
    the correct answer for any caller that builds/passes a Namespace with scenario unset."""
    if not cli.scenario:
        return None, {}
    scenario = SCENARIOS[cli.scenario]
    options = dict(scenario.default_options)
    for raw in cli.scenario_option:
        key, sep, value = raw.partition('=')
        if not sep:
            raise ValueError(f"--scenario-option must be KEY=VALUE, got {raw!r}")
        options[key] = _coerce(value)
    return scenario, options


def apply_scenario_to_env_cfg(env_cfg, scenario: Optional[Scenario], options: dict,
                               apply_fail_hold: bool = True) -> None:
    """Wires a resolved scenario (see resolve_scenario) into env_cfg -- the one place
    cfg.props.list/init_state.rot/fail_to_terminal_time_s get scenario values, reused
    identically by play.py, rugiar_driver.py and rugiar_driver_target.py. No-op if
    `scenario` is None (no --scenario passed).

    `apply_fail_hold=False` skips the fail_to_terminal_time_s override -- play.py never
    applied it even for --race (only the driver scripts did, so a crash stays on screen
    for an operator to see); kept that way here to not change play.py's behavior."""
    if scenario is None:
        return
    env_cfg.props.list = scenario.spawn_props(options)
    if scenario.init_state_rot is not None:
        env_cfg.init_state.rot = scenario.init_state_rot
    if scenario.init_state_pos_offset is not None:
        env_cfg.init_state.pos = [
            p + d for p, d in zip(env_cfg.init_state.pos, scenario.init_state_pos_offset)
        ]
    if apply_fail_hold and scenario.fail_to_terminal_time_s is not None:
        env_cfg.env.fail_to_terminal_time_s = scenario.fail_to_terminal_time_s
