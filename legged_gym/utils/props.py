"""Shared prop presets for the 'ball'/'race' scenarios (see legged_gym/utils/scenarios.py,
selected via --scenario in play.py/rugiar_driver.py/rugiar_driver_target.py)."""

import math


def default_ball_prop():
    """A single physics-enabled sphere spawned just in front of the robot."""
    return {
        "name": "ball",
        "shape": "sphere",
        "size": 0.15,
        "mass": 1.5,
        "pos": [0.8, 0.0, 1.0],
        "restitution": 0.6,
        "friction": 0.8,
        "linear_damping": 1.5,  # 1/s decay rate on linear velocity -- viscous drag, so it coasts to a stop
        "color": [1.0, 0.1, 0.1, 1.0],
    }


# Default track length for the 'race' scenario -- the exact distance between the start crossing-line
# and the finish crossing-line (not the START/FINISH text, which is a separate
# legibility marking near each line): start at the robot's own spawn point
# (x=0), finish RACE_TRACK_LENGTH further along -x -- see RACE_SPAWN_ROT below
# for why -x. A clean, simple number by design (asked for "una distancia exacta
# y simple" -- 7m). An "Olympic sprint" style straight the robot is meant to be
# driven down with manual velocity commands until it runs into the mat.
RACE_TRACK_LENGTH = 7.0
# Total depth (along the direction of travel) the START/FINISH line-text
# stretches -- each letter's local "up" axis spans [0, 2], so the per-unit
# scale passed to _word_line_props is half of this. Kept modest so FINISH's
# own footprint doesn't eat into RACE_FINISH_GAP below.
RACE_LETTER_DEPTH = 1.0
# Clear gap between the FAR edge of the FINISH text and the mat's front face
# -- kept visually separate ("separada, no debajo") rather than the mat
# starting right where the text ends, but close ("acercar la palabra FINISH
# a la colchoneta") rather than the large gap this used to be.
RACE_FINISH_GAP = 0.6
# Width (across the lane) of the simple straight start/finish crossing-lines
# -- wide enough to sit under either word's full footprint.
RACE_CROSSING_LINE_WIDTH = 3.2
# Clear margin between each crossing-line's edge and the nearest edge of its
# START/FINISH text -- roughly one line-width, just enough that the two don't
# touch/overlap, per request ("un ancho de linea como separacion").
RACE_LINE_TEXT_MARGIN = 0.15
# [x,y,z,w] (gym quat convention, see legged_robot_config.py's init_state.rot)
# for a 180 degree yaw -- spawns the robot facing -x, i.e. facing straight
# down the race track towards the finish/mat, so a plain positive vx walks
# it there (no backwards walking / turning-in-place needed).
RACE_SPAWN_ROT = [0.0, 0.0, 1.0, 0.0]

# How long a fail state (fallen over / excessive contact force -- see
# legged_robot.py's check_termination()) is held before the env
# auto-resets, while the 'race' scenario is active. Normal training/demo default
# (env_cfg.env.fail_to_terminal_time_s) is 0.1s -- fine for training
# throughput, but it means the robot snaps back to spawn on the very next
# tick after going down, before anyone watching could actually see it fall
# or read the race result. An hour is "never happens on its own, only
# Restart resets it" without touching the underlying (int64 tick-counted,
# see legged_robot.py's fail_buf) mechanism's contract.
RACE_FAIL_HOLD_S = 3600.0

# A minimal block/stick font: each letter is a handful of straight strokes
# (horizontal, vertical, or diagonal) in a local [0, 1] x [0, 2] cell,
# rendered as thin flat boxes lying on the ground -- "escrito con lineas"
# instead of real glyph rendering, which Genesis/viser can't do for a
# physics prop. Only the letters START/FINISH actually need.
_LETTER_SEGMENTS = {
    "S": [((0, 2), (1, 2)), ((0, 1), (0, 2)), ((0, 1), (1, 1)), ((1, 0), (1, 1)), ((0, 0), (1, 0))],
    "T": [((0, 2), (1, 2)), ((0.5, 0), (0.5, 2))],
    "A": [((0, 0), (0, 2)), ((1, 0), (1, 2)), ((0, 2), (1, 2)), ((0, 1), (1, 1))],
    "R": [((0, 0), (0, 2)), ((0, 2), (1, 2)), ((1, 1), (1, 2)), ((0, 1), (1, 1)), ((0, 1), (1, 0))],
    "F": [((0, 0), (0, 2)), ((0, 2), (1, 2)), ((0, 1), (1, 1))],
    "I": [((0.3, 2), (0.7, 2)), ((0.5, 0), (0.5, 2)), ((0.3, 0), (0.7, 0))],
    "N": [((0, 0), (0, 2)), ((1, 0), (1, 2)), ((0, 2), (1, 0))],
    "H": [((0, 0), (0, 2)), ((1, 0), (1, 2)), ((0, 1), (1, 1))],
}


def _word_line_props(word, name_prefix, x0, color, forward_sign=-1.0,
                      letter_w=0.4, letter_h=RACE_LETTER_DEPTH / 2, gap=0.15, thickness=0.06,
                      mark_height=0.02, z=0.011):
    """Ground-level props spelling `word` out of straight-line strokes (see
    _LETTER_SEGMENTS), one flat rotated box per stroke -- a painted-line
    approximation of text for the race track markings.

    Layout: letters are laid out side by side across the lane (local x -> world
    y, centered on y=0), each letter's local "up" axis (0->2) stretched along
    the direction of travel (local y -> world x, scaled by `forward_sign` so
    the word reads correctly as the robot approaches down -x) -- the same
    convention real road-painted text uses (elongated in the direction of
    travel, since it's read obliquely while approaching, not from directly
    above).
    """
    letters = word.upper()
    total_width = len(letters) * letter_w + (len(letters) - 1) * gap
    y0 = -total_width / 2
    props = []
    for i, ch in enumerate(letters):
        cell_y0 = y0 + i * (letter_w + gap)
        for j, ((lx1, ly1), (lx2, ly2)) in enumerate(_LETTER_SEGMENTS.get(ch, [])):
            wy1, wy2 = cell_y0 + lx1 * letter_w, cell_y0 + lx2 * letter_w
            wx1, wx2 = x0 + forward_sign * ly1 * letter_h, x0 + forward_sign * ly2 * letter_h
            dx, dy = wx2 - wx1, wy2 - wy1
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            angle_deg = math.degrees(math.atan2(dy, dx))
            props.append({
                "name": f"{name_prefix}_{i}_{j}",
                "shape": "box",
                "size": [length, thickness, mark_height],
                "pos": [(wx1 + wx2) / 2, (wy1 + wy2) / 2, z],
                "euler": [0.0, 0.0, angle_deg],
                "fixed": True,
                # Ground-painted text, not a physical curb -- a raised box the
                # robot's foot can catch its edge on (even at a couple cm)
                # trips a live walking gait. See _crossing_line_prop() for the
                # same fix on the start/finish bars.
                "collision": False,
                "color": color,
            })
    return props


def _crossing_line_prop(name, x, color=(0.95, 0.95, 0.95, 1.0)):
    """A simple straight bar across the lane marking the exact start/finish
    crossing point -- distinct from the START/FINISH line-text, which is a
    nearby legibility marking, not the precise measured point. RACE_TRACK_LENGTH
    is measured between this prop at x=0 and this prop at x=-RACE_TRACK_LENGTH.
    """
    return {
        "name": name,
        "shape": "box",
        "size": [0.06, RACE_CROSSING_LINE_WIDTH, 0.02],
        "pos": [x, 0.0, 0.011],
        "fixed": True,
        # Paint on the ground, not a curb -- collision=False so a walking
        # gait can't catch a foot on its edge (see _word_line_props()'s
        # same fix, reported live as the robot's foot getting stuck on it).
        "collision": False,
        "color": list(color),
    }


def default_race_props(track_length=RACE_TRACK_LENGTH):
    """Static (fixed, non-physics) scenery for the 'race' scenario: a start
    crossing-line at the robot's spawn (x=0) with "START" line-text next to
    it, a finish crossing-line `track_length` further down -x with "FINISH"
    line-text next to it, and (separated from the finish text by a visible
    gap, not touching it) a big blue crash-mat wall to run into -- like an
    Olympic sprint track, sim/viser-only for now (see `default_ball_prop()`'s
    docstring convention: opt-in per driver/play invocation, never touched by
    training, since `cfg.props.list` stays empty unless a caller explicitly
    sets it).

    `track_length` defaults to RACE_TRACK_LENGTH but is overridable (see
    legged_gym/utils/scenarios.py's --scenario-option track_length=...).

    Pair with RACE_SPAWN_ROT (env_cfg.init_state.rot) so the robot actually
    faces -x, down the track, at spawn.

    Every entry is `fixed=True` -- static geometry, not dynamic rigid bodies
    like the ball prop, so they don't fall/get knocked away.
    """
    finish_x = -track_length
    mat_size = [1.0, 6.0, 1.8]  # [depth (x), width (y), height (z)] -- lying on its long
    # side (width along the ground, across the track), not standing on end: wide enough to
    # always be in the robot's path, still taller than the robot but not top-heavy-looking.

    # Text anchors sit a clear margin PAST their crossing-line (in the direction of
    # travel, -x) so the line and the text's nearest stroke don't touch/overlap.
    line_half_thickness = _crossing_line_prop("_", 0.0)["size"][0] / 2
    text_offset = line_half_thickness + RACE_LINE_TEXT_MARGIN
    start_text_x0 = 0.0 - text_offset
    finish_text_x0 = finish_x - text_offset

    finish_text_far_edge = finish_text_x0 - RACE_LETTER_DEPTH  # -x tip of the FINISH text's strokes
    mat_front_face = finish_text_far_edge - RACE_FINISH_GAP
    mat_x = mat_front_face - mat_size[0] / 2

    props = [
        _crossing_line_prop("race_start_line", x=0.0),
        _crossing_line_prop("race_finish_line", x=finish_x),
    ]
    props += _word_line_props("START", "race_start_text", x0=start_text_x0, color=[0.95, 0.95, 0.95, 1.0])
    props += _word_line_props("FINISH", "race_finish_text", x0=finish_text_x0, color=[1.0, 0.55, 0.0, 1.0])
    props.append({
        "name": "race_finish_mat",
        "shape": "box",
        "size": mat_size,
        "pos": [mat_x, 0.0, mat_size[2] / 2],
        "fixed": True,
        "color": [0.1, 0.3, 0.9, 1.0],
    })
    return props
