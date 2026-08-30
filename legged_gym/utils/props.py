"""Shared prop presets for the 'ball'/'race'/'rough_terrain' scenarios (see
legged_gym/utils/scenarios.py, selected via --scenario in
play.py/rugiar_driver.py/rugiar_driver_target.py)."""

import math
import random


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


def _crossing_line_prop(name, x, color=(0.95, 0.95, 0.95, 1.0), height=0.02):
    """A simple straight bar across the lane marking the exact start/finish
    crossing point -- distinct from the START/FINISH line-text, which is a
    nearby legibility marking, not the precise measured point. RACE_TRACK_LENGTH
    is measured between this prop at x=0 and this prop at x=-RACE_TRACK_LENGTH.

    `height` defaults to race's original 0.02 but can be thinned further (see
    default_rough_terrain_props(), which wants this to read as flush paint, not
    a raised strip, now that it sits right next to real climbable tiles rather
    than race's flat open lane).
    """
    return {
        "name": name,
        "shape": "box",
        "size": [0.06, RACE_CROSSING_LINE_WIDTH, height],
        "pos": [x, 0.0, height / 2 + 0.001],
        "fixed": True,
        # Paint on the ground, not a curb -- collision=False so a walking
        # gait can't catch a foot on its edge (see _word_line_props()'s
        # same fix, reported live as the robot's foot getting stuck on it).
        "collision": False,
        "color": list(color),
    }


def _word_sign_props(word, name_prefix, sign_x, z0, color, forward_offset=0.03,
                      letter_w=0.4, letter_h=None, gap=0.15, thickness=0.06, mark_height=0.02):
    """Same block-stroke lettering as `_word_line_props`, but painted onto a VERTICAL
    face (an elevated sign board) instead of the ground: each letter's local "up" axis
    (0->2) maps to world z (climbing the board) rather than world x (depth along the
    track), and stays at a fixed world x (`sign_x` + a small `forward_offset` so the
    paint sits just off the board's front face, not embedded in it). Letters are still
    spread across world y (the lane width), centered on y=0, exactly like the ground
    version.

    Rotating a stroke within the vertical (y, z) plane takes a rotation about world X
    (not Z, which is what rotates strokes flat on the ground) -- see the size/euler
    layout below: size=[mark_height, length, thickness] puts the stroke's long axis on
    local y (not local x), so an X-axis euler rotation carries it to any angle within
    the y-z plane, while local x (mark_height, the paint's protrusion off the board)
    is untouched by that rotation and stays pointing along world x as intended.
    """
    letter_h = RACE_LETTER_DEPTH / 2 if letter_h is None else letter_h
    letters = word.upper()
    total_width = len(letters) * letter_w + (len(letters) - 1) * gap
    y0 = -total_width / 2
    props = []
    for i, ch in enumerate(letters):
        cell_y0 = y0 + i * (letter_w + gap)
        for j, ((lx1, ly1), (lx2, ly2)) in enumerate(_LETTER_SEGMENTS.get(ch, [])):
            wy1, wy2 = cell_y0 + lx1 * letter_w, cell_y0 + lx2 * letter_w
            wz1, wz2 = z0 + ly1 * letter_h, z0 + ly2 * letter_h
            dy, dz = wy2 - wy1, wz2 - wz1
            length = math.hypot(dy, dz)
            if length < 1e-6:
                continue
            angle_deg = math.degrees(math.atan2(dz, dy))
            props.append({
                "name": f"{name_prefix}_{i}_{j}",
                "shape": "box",
                "size": [mark_height, length, thickness],
                "pos": [sign_x + forward_offset, (wy1 + wy2) / 2, (wz1 + wz2) / 2],
                "euler": [angle_deg, 0.0, 0.0],
                "fixed": True,
                # Paint on the board's face, not a physical relief -- see
                # _word_line_props()'s identical reasoning for the ground version.
                "collision": False,
                "color": color,
            })
    return props


def _sign_props(name_prefix, x, lane_width, word, word_color):
    """An elevated start/finish marker: two thin poles at the edges of the lane (out of
    the walking path down the center) holding up a board that spans between them, with
    `word` painted on its face via `_word_sign_props`. Replaces race's ground-painted
    START/FINISH text -- requested as "carteles elevados" (elevated signs) for the
    rough_terrain scenario, like a real obstacle-course start/finish gantry, rather than
    text a robot could be looking straight down at while stepping over broken ground.
    """
    pole_half = lane_width / 2 - ROUGH_TERRAIN_SIGN_POLE_THICKNESS / 2
    board_z0 = ROUGH_TERRAIN_SIGN_POLE_HEIGHT
    board_z_center = board_z0 + ROUGH_TERRAIN_SIGN_BOARD_HEIGHT / 2
    props = []
    for side, y in (("l", -pole_half), ("r", pole_half)):
        props.append({
            "name": f"{name_prefix}_pole_{side}",
            "shape": "box",
            "size": [ROUGH_TERRAIN_SIGN_POLE_THICKNESS, ROUGH_TERRAIN_SIGN_POLE_THICKNESS,
                     ROUGH_TERRAIN_SIGN_POLE_HEIGHT],
            "pos": [x, y, ROUGH_TERRAIN_SIGN_POLE_HEIGHT / 2],
            "fixed": True,
            "color": [0.15, 0.15, 0.15, 1.0],
        })
    props.append({
        "name": f"{name_prefix}_board",
        "shape": "box",
        "size": [ROUGH_TERRAIN_SIGN_BOARD_THICKNESS, lane_width, ROUGH_TERRAIN_SIGN_BOARD_HEIGHT],
        "pos": [x, 0.0, board_z_center],
        "fixed": True,
        "color": [0.95, 0.95, 0.95, 1.0],
    })
    # Letters climb from a small margin above the board's own bottom edge, sized to fit
    # within its height (2 local "up" units -> ROUGH_TERRAIN_SIGN_BOARD_HEIGHT).
    props += _word_sign_props(
        word, f"{name_prefix}_text", sign_x=x, z0=board_z0 + 0.05, color=word_color,
        letter_h=(ROUGH_TERRAIN_SIGN_BOARD_HEIGHT - 0.1) / 2)
    return props


# Square "laja"/pixel tile footprint -- a clean, simple paver size (roughly a big
# footstep wide) that also tiles RACE_CROSSING_LINE_WIDTH and RACE_TRACK_LENGTH cleanly
# enough to fill the lane without leaving slivers.
ROUGH_TERRAIN_TILE_SIZE = 0.35
# Every tile is at least this tall -- a real, solid slab even where the random walk
# below bottoms out at zero extra height, not a degenerate zero-height box.
ROUGH_TERRAIN_BASE_HEIGHT = 0.02
# The tallest a tile can ever get (base height + this) -- reached only right at the
# very end, per the exponential difficulty ramp below. A ~1.3m biped cannot climb a
# ~2m wall -- this is deliberately a literal wall by the end, not just "hard": the
# game is "how far did you get" precisely because finishing is meant to be
# impossible, and to bite well before the very last tile too (requested: "más alto,
# 2 metros al final... para que tenga más dificultad desde antes" -- the exponential
# curve's own shape means raising the ceiling also raises every earlier point on the
# curve, not just the final tile).
ROUGH_TERRAIN_MAX_STEP = 2.0
# How sharply the difficulty ramp curves -- see rough_terrain_baseline_height() below.
# Higher = flatter for longer, then a steeper last-minute spike towards
# ROUGH_TERRAIN_MAX_STEP ("algo parecido a una exponencial"). Lowered from 5.0 -- still
# clearly exponential (not linear), but the climb starts noticeably sooner so the
# challenge doesn't take as long to actually show up (requested: "que la rampa crezca
# un poco más rápido, un poco no más, para que no tarde tanto en enfrentar el
# desafío"). At the midpoint (frac=0.5) this roughly doubles the difficulty already
# reached vs. the old value (~12% of MAX_STEP vs ~8%).
ROUGH_TERRAIN_STEP_CURVE_K = 4.0
# Per-tile random variation around its row's baseline height, as a fraction of that
# baseline -- texture, not the difficulty mechanism itself (requested: tiles at the
# same depth should look "parecidas... hasta un 10% diferencia").
ROUGH_TERRAIN_HEIGHT_JITTER = 0.10
# Same corridor as the 'race' scenario, reused for parity (requested: "el callejón que
# llevaría hasta la meta del race") rather than inventing new geometry constants.
ROUGH_TERRAIN_TRACK_LENGTH = RACE_TRACK_LENGTH
ROUGH_TERRAIN_LANE_WIDTH = RACE_CROSSING_LINE_WIDTH
# How far behind the start line (+x, since the track runs along -x) the robot spawns
# -- a deliberately small setback so it starts a step short of the line, not standing
# on top of it (x=0) or already past it (negative x). Requested directly: "el robot
# debería empezar un poquito atrás de la línea blanca, no sobre ni por delante".
ROUGH_TERRAIN_SPAWN_SETBACK = 0.15
# Sign geometry -- poles tall enough to clear BOTH the robot's own height and the
# tallest tile the terrain ever reaches (ROUGH_TERRAIN_BASE_HEIGHT + MAX_STEP, right
# next to the finish sign) so the board reads as an overhead gantry above the terrain,
# not partly buried inside the final wall of tiles, plus a board with enough vertical
# room for the block-stroke lettering.
ROUGH_TERRAIN_SIGN_POLE_HEIGHT = 2.4
ROUGH_TERRAIN_SIGN_POLE_THICKNESS = 0.06
ROUGH_TERRAIN_SIGN_BOARD_HEIGHT = 0.5
ROUGH_TERRAIN_SIGN_BOARD_THICKNESS = 0.05
# Flat, tile-free clearance between the start crossing-line (x=0, where the robot
# spawns) and the first tile -- without this, the first tile's edge sits right under
# the robot's own starting stance, and a real collision box that close catches a foot
# before the robot has even taken a step (reported live: "el pie queda pegado").
# Tiles now only appear once you're already past this gap, not on top of the line.
ROUGH_TERRAIN_START_GAP = 0.4


def rough_terrain_baseline_height(frac):
    """The row's own target height at progress `frac` (0 at the start row, 1 at the
    finish row) along the track -- an exponential curve, near-flat for most of the
    track and then rocketing up right at the end, rather than a straight linear ramp
    -- requested directly ("algo parecido a una exponencial la altura random de las
    lajas"), and it reads better too: a robot should visibly be coping fine for a
    while before the terrain turns against it all at once, not sensing a steadily
    worsening slope from the first tile. Every tile in a row is jittered a little
    around this SAME baseline (see rough_terrain_tile_heights()) rather than drawn
    independently, so the rising floor itself is what reads as "getting harder", not
    per-tile randomness -- tiles at the same depth stay visibly similar to each other
    ("un poco más parejo... lajas que son parecidas en altura"), the height itself is
    what climbs.
    frac=0 -> 0, frac=1 -> ROUGH_TERRAIN_MAX_STEP, monotonically increasing between.
    """
    return ROUGH_TERRAIN_MAX_STEP * (math.exp(ROUGH_TERRAIN_STEP_CURVE_K * frac) - 1) / (
        math.exp(ROUGH_TERRAIN_STEP_CURVE_K) - 1)


def rough_terrain_tile_heights(track_length=ROUGH_TERRAIN_TRACK_LENGTH,
                                lane_width=ROUGH_TERRAIN_LANE_WIDTH, seed=None):
    """Returns (rows, cols, heights) where heights[row][col] is that tile's EXTRA
    height above ROUGH_TERRAIN_BASE_HEIGHT (0 at the easiest, up to
    ROUGH_TERRAIN_MAX_STEP at the hardest) -- row 0 is at the start line (x=0), the
    last row is at the finish line (x=-track_length).

    Each tile is the row's own rough_terrain_baseline_height(frac), jittered by up to
    +-ROUGH_TERRAIN_HEIGHT_JITTER (10%) independently per tile -- just enough
    per-paver texture that it doesn't look like a perfectly flat poured slab, without
    the wide swings a symmetric random WALK would produce (an earlier version of this
    generator used one, and neighboring tiles late in the track could differ by close
    to the full ROUGH_TERRAIN_MAX_STEP -- directly reported as not what was wanted:
    tiles at a given point in the track should look like each other, "hasta un 10%
    diferencia"). The anti-trip property ("que no se traben los pies") still holds --
    zero gaps (tiles abut, see default_rough_terrain_props()) plus a jitter that's a
    small fraction of the row's own height, not an unbounded step -- but the row-to-row
    RISE in the baseline itself is what makes the far end genuinely unclimbable, per
    "se pone cada vez más rugosa a medida que se acerca al final [...] tiene que ser
    imposible que llegue al final completo". `seed` makes a run reproducible (e.g. for
    judging a fixed course) but defaults to None -- genuinely random -- per the
    request ("un random de las alturas").
    """
    rng = random.Random(seed)
    cols = max(1, int(lane_width // ROUGH_TERRAIN_TILE_SIZE))
    # The tile field itself only spans track_length - ROUGH_TERRAIN_START_GAP -- see
    # default_rough_terrain_props()'s docstring on why the first tile is held back from
    # the start line.
    rows = max(1, int((track_length - ROUGH_TERRAIN_START_GAP) // ROUGH_TERRAIN_TILE_SIZE))
    heights = []
    for row in range(rows):
        frac = row / (rows - 1) if rows > 1 else 1.0
        baseline = rough_terrain_baseline_height(frac)
        row_heights = []
        for _col in range(cols):
            jitter = rng.uniform(-ROUGH_TERRAIN_HEIGHT_JITTER, ROUGH_TERRAIN_HEIGHT_JITTER)
            row_heights.append(min(max(baseline * (1.0 + jitter), 0.0), ROUGH_TERRAIN_MAX_STEP))
        heights.append(row_heights)
    return rows, cols, heights


def default_rough_terrain_props(track_length=ROUGH_TERRAIN_TRACK_LENGTH, seed=None):
    """Static scenery for the 'rough_terrain' scenario: the same start/finish crossing
    lines as 'race' (see default_race_props()), but the flat lane between them is
    replaced by a field of square paver tiles whose height gets rougher the closer
    they are to the finish -- "caminando sobre una superficie que se pone cada vez más
    rugosa a medida que se acerca al final, [...] quedando como un imposible llegar al
    final" -- and the ground-painted START/FINISH text is replaced by elevated
    overhead signs (see _sign_props()). Unlike race, there's no crash mat at the end:
    the terrain itself is what stops the robot well before it (explicitly requested --
    "la colchoneta azul no debe estar para este escenario" -- and there's nothing left
    for a mat to catch once ROUGH_TERRAIN_MAX_STEP's ~1m final tiles are genuinely
    unclimbable on their own).

    The scoring game this enables isn't "did you finish" -- it's built not to be
    reachable, see ROUGH_TERRAIN_MAX_STEP's docstring -- it's "how far did you get,
    how fast", scored the instant SafetyGovernor trips (web/app.js's
    onRoughTerrainFall(), reading status.safety_tripped) and held on screen briefly
    before auto-restarting for another attempt.

    `track_length`/`seed` are overridable via --scenario-option (seed isn't in this
    scenario's default_options, so it's None -- genuinely random -- unless a caller
    explicitly passes one).
    """
    finish_x = -track_length
    lane_width = ROUGH_TERRAIN_LANE_WIDTH

    props = [
        # Thinned to near-flush (see _crossing_line_prop's `height` param) -- these two
        # now sit right next to real climbable tiles rather than race's flat open lane,
        # so even a purely-cosmetic step here is worth avoiding.
        _crossing_line_prop("rough_terrain_start_line", x=0.0, height=0.005),
        _crossing_line_prop("rough_terrain_finish_line", x=finish_x, height=0.005),
    ]

    rows, cols, heights = rough_terrain_tile_heights(track_length=track_length, lane_width=lane_width, seed=seed)
    y0 = -(cols * ROUGH_TERRAIN_TILE_SIZE) / 2
    for row in range(rows):
        # Held back by ROUGH_TERRAIN_START_GAP -- the first tile appears only once
        # you're already past the start line, not straddling it (see the constant's
        # own docstring).
        tile_x = -ROUGH_TERRAIN_START_GAP - (row + 0.5) * ROUGH_TERRAIN_TILE_SIZE
        for col in range(cols):
            extra_h = heights[row][col]
            tile_h = ROUGH_TERRAIN_BASE_HEIGHT + extra_h
            tile_y = y0 + (col + 0.5) * ROUGH_TERRAIN_TILE_SIZE
            frac = extra_h / ROUGH_TERRAIN_MAX_STEP if ROUGH_TERRAIN_MAX_STEP > 0 else 0.0
            # Orange, deepening toward red as a tile gets harder -- a free visual read
            # of the difficulty ramp, not required for the physics.
            color = [1.0, 0.55 - 0.35 * frac, 0.0, 1.0]
            props.append({
                "name": f"rough_terrain_tile_{row}_{col}",
                "shape": "box",
                "size": [ROUGH_TERRAIN_TILE_SIZE, ROUGH_TERRAIN_TILE_SIZE, tile_h],
                "pos": [tile_x, tile_y, tile_h / 2],
                "fixed": True,
                # Unlike race's ground text/lines, these ARE meant to be physically
                # climbed -- collision stays on (Genesis's own default).
                "color": color,
            })

    # Both boards are white (see _sign_props) -- START's text needs a dark color to
    # actually read against it (white-on-white was reported invisible live); FINISH
    # already had enough contrast with orange, kept as-is.
    props += _sign_props("rough_terrain_start_sign", x=0.0, lane_width=lane_width,
                          word="START", word_color=[0.08, 0.08, 0.08, 1.0])
    props += _sign_props("rough_terrain_finish_sign", x=finish_x, lane_width=lane_width,
                          word="FINISH", word_color=[1.0, 0.55, 0.0, 1.0])
    return props


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
