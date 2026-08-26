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
    if apply_fail_hold and scenario.fail_to_terminal_time_s is not None:
        env_cfg.env.fail_to_terminal_time_s = scenario.fail_to_terminal_time_s
