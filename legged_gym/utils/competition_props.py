"""Prop presets for the World Humanoid Robot Games (Beijing, 世界人形机器人运动会)
competition scenarios -- digitized from the two official rulebook PDFs published by
the organizing committee at whrgoc.com ("2025世界人形机器人运动会比赛规则", first and
second batch). Each function below returns a list of fixed-box props approximating
that event's real arena furniture/obstacles, sized from the rulebook's own tables
(shelf/rack/table/box/cart dimensions are the official ones; Genesis prop shapes here
are limited to box/sphere -- see genesis_simulator.py::_create_props -- so anything
round or wedge-shaped in the real event, e.g. poles or ramps, is approximated with
thin boxes, not a claim of pixel-exact diagram reproduction).

Two families here:

- Seven task-arena scenarios (factory/hospital/hotel/warehouse) -- static furniture
  layouts for navigation/manipulation practice. Robot spawns facing -x (via
  RACE_SPAWN_ROT, same as 'race'), with furniture laid out at negative x/y -- i.e.
  the whole arena (robot + furniture) is spun 180 degrees from this repo's own
  default +x facing (see _rotate_yaw180() below for why: with the default facing,
  furniture placed "ahead" of the robot at +x sat between it and the viser
  camera's default fixed viewpoint, hiding the robot from view).
- One track scenario, 'obstacle_course' -- the 100m-obstacle event from the same
  rulebook, laid out exactly like 'race'/'rough_terrain' (start line at x=0, robot
  faces -x down the lane via RACE_SPAWN_ROT) since it's the harder, obstacle-laden
  ALTERNATIVE to that same straight-line race -- see legged_gym/utils/scenarios.py.
"""

import math

from legged_gym.utils.props import (
    RACE_CROSSING_LINE_WIDTH, _crossing_line_prop, _sign_props,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SHELF_COLOR = [0.55, 0.55, 0.6, 1.0]
_BOX_COLOR = [0.15, 0.45, 0.85, 1.0]
_TABLE_COLOR = [0.55, 0.4, 0.25, 1.0]
_CART_COLOR = [0.6, 0.55, 0.15, 1.0]
_FURNITURE_COLOR = [0.45, 0.32, 0.2, 1.0]
_DOOR_COLOR = [0.88, 0.88, 0.85, 1.0]
_SOFA_COLOR = [0.68, 0.6, 0.5, 1.0]
_BIN_COLOR = [0.25, 0.25, 0.25, 1.0]
_LUGGAGE_COLOR = [0.15, 0.15, 0.18, 1.0]


def _box(name, size, pos, color, euler=None, collision=True):
    """One fixed box prop -- the common shape every piece of furniture/obstacle
    below is built from (see module docstring on why: box/sphere are the only
    shapes genesis_simulator.py's _create_props() supports)."""
    prop = {
        "name": name,
        "shape": "box",
        "size": list(size),
        "pos": list(pos),
        "fixed": True,
        "collision": collision,
        "color": list(color),
    }
    if euler is not None:
        prop["euler"] = list(euler)
    return prop


def _rotate_yaw180(props):
    """Spins an entire prop list 180 degrees around the world origin (negates each
    prop's x/y position) -- pairs with RACE_SPAWN_ROT on the robot's own
    init_state_rot (see scenarios.py's 7 task-arena entries) so the ROBOT and its
    FURNITURE turn together, not just one of them. Every prop built here is an
    axis-aligned box, which looks identical after a 180-degree spin about its own
    center -- so only the prop's location needs to flip, not its own `euler`."""
    rotated = []
    for p in props:
        p = dict(p)
        pos = list(p["pos"])
        pos[0], pos[1] = -pos[0], -pos[1]
        p["pos"] = pos
        rotated.append(p)
    return rotated


def _ramp_prop(name, x_ground, x_top, y_center, width, thickness, height, color):
    """A thin inclined box climbing from ground level (z=0) at `x_ground` up to
    `height` at `x_top` -- the box/sphere-only prop system's approximation of a
    real wedge/ramp (see the 100m-obstacle course's slope obstacles below).
    `x_top` can be on either side of `x_ground` (handles the -x direction of
    travel the obstacle course runs in, same convention as 'race')."""
    dx = x_top - x_ground
    length = math.hypot(dx, height)
    angle_deg = -math.degrees(math.atan2(height, dx))
    return _box(name, [length, width, thickness],
                [(x_ground + x_top) / 2, y_center, height / 2], color,
                euler=[0.0, angle_deg, 0.0])


def _staircase_props(name_prefix, x0, y_center, width, num_steps, step_depth, step_height,
                      color, total_length=None):
    """An up-then-down staircase (ascending `num_steps` of `step_height` each, an
    optional flat plateau at the peak, then descending the same number back to
    ground) starting at `x0` and running in -x -- approximates the straight/spiral
    staircase obstacles (直线台阶/螺旋台阶), which the rulebook specifies as
    symmetric up+down step counts. Returns (props, next_x) so obstacle_course props
    can chain obstacles sequentially without hand-computed offsets."""
    props = []
    x = x0
    for i in range(num_steps):
        h = (i + 1) * step_height
        cx = x - step_depth / 2
        props.append(_box(f"{name_prefix}_up_{i}", [step_depth, width, h], [cx, y_center, h / 2], color))
        x -= step_depth
    covered = 2 * num_steps * step_depth
    flat_len = max(0.0, (total_length or covered) - covered)
    if flat_len > 0:
        h = num_steps * step_height
        cx = x - flat_len / 2
        props.append(_box(f"{name_prefix}_top", [flat_len, width, h], [cx, y_center, h / 2], color))
        x -= flat_len
    for i in range(num_steps):
        h = (num_steps - i) * step_height
        cx = x - step_depth / 2
        props.append(_box(f"{name_prefix}_down_{i}", [step_depth, width, h], [cx, y_center, h / 2], color))
        x -= step_depth
    return props, x


# ---------------------------------------------------------------------------
# 1. Factory scenario -- 工厂场景-物料搬运技能竞技 (material handling), 7m x 5m
# ---------------------------------------------------------------------------

def default_factory_handling_props():
    """Intake/exhaust-valve parts rack (5 tiers, 1.73m deep) plus the two line-side
    carts robots ferry material boxes between -- sizes from 表7 (first-batch rulebook,
    工厂场景-物料搬运技能竞技, p.25-27). Robot spawns at the event's own 起点/终点
    circle, facing the rack/carts."""
    return _rotate_yaw180([
        _box("factory_handling_rack", [1.73, 2.0, 1.68], [3.0, -1.3, 0.84], _SHELF_COLOR),
        _box("factory_handling_cart1", [1.3, 1.1, 0.34], [2.2, -0.2, 0.17], _CART_COLOR),
        _box("factory_handling_cart2", [1.3, 1.1, 0.34], [2.2, 1.0, 0.17], _CART_COLOR),
        _box("factory_handling_box_stack", [0.4, 0.3, 0.6], [2.2, 2.0, 0.3], _BOX_COLOR),
    ])


# ---------------------------------------------------------------------------
# 2. Factory scenario -- 工厂场景-物料整理技能竞技 (material organizing), 5m x 3m
# ---------------------------------------------------------------------------

def default_factory_sorting_props():
    """A single organizing workstation (长1m宽0.6m高0.7m) holding two material
    boxes -- 表8, p.27-28."""
    return _rotate_yaw180([
        _box("factory_sorting_station", [1.0, 0.6, 0.7], [1.5, 0.0, 0.35], _TABLE_COLOR),
        _box("factory_sorting_box1", [0.4, 0.3, 0.22], [1.3, -0.15, 0.71], _BOX_COLOR),
        _box("factory_sorting_box2", [0.4, 0.3, 0.22], [1.3, 0.15, 0.71], _BOX_COLOR),
    ])


# ---------------------------------------------------------------------------
# 3. Hospital scenario -- 医院场景-药品分拣技能竞技 (pharmacy sorting), 5m x 5m
# ---------------------------------------------------------------------------

def default_hospital_pharmacy_props():
    """8 drug shelves (长1m宽0.5m高2m each, 6 tiers) laid out per 图8: a row of 4
    along the far wall, two staggered rows of 2 closer in, plus the pick-up
    worktable (长1m宽0.5m高0.7m) -- 表9, p.29-30."""
    props = []
    for i, y in enumerate([-1.5, -0.5, 0.5, 1.5]):
        props.append(_box(f"pharmacy_shelf_back_{i}", [0.5, 1.0, 2.0], [3.5, y, 1.0], _SHELF_COLOR))
    for i, y in enumerate([-0.5, 0.5]):
        props.append(_box(f"pharmacy_shelf_mid_{i}", [0.5, 1.0, 2.0], [2.5, y, 1.0], _SHELF_COLOR))
        props.append(_box(f"pharmacy_shelf_near_{i}", [0.5, 1.0, 2.0], [2.0, y, 1.0], _SHELF_COLOR))
    props.append(_box("pharmacy_worktable", [1.0, 0.5, 0.7], [1.0, -1.0, 0.35], _TABLE_COLOR))
    return _rotate_yaw180(props)


# ---------------------------------------------------------------------------
# 4. Hospital scenario -- 医院场景-拆药分装技能竞技 (dispensing), 5m x 5m
# ---------------------------------------------------------------------------

def default_hospital_dispensing_props():
    """A single nursing worktable (长1m宽0.5m高0.7m) holding blister packs and
    dispensing boxes -- 表10, p.31-33."""
    return _rotate_yaw180([
        _box("dispensing_worktable", [1.0, 0.5, 0.7], [1.5, 0.0, 0.35], _TABLE_COLOR),
    ])


# ---------------------------------------------------------------------------
# 5. Hotel scenario -- 酒店场景-迎宾服务技能竞技 (reception/bellhop), 11m x 6m
# ---------------------------------------------------------------------------

def default_hotel_reception_props():
    """Luggage storage area, bellhop cart (长1.05m宽0.61m高1.86m), and a row of 5
    virtual room doors (贴纸 door-frame stickers, 宽1m×高1.2m) along the corridor
    wall -- 表11, p.33-34."""
    props = [
        _box("hotel_reception_luggage_rack", [1.2, 0.5, 1.2], [2.0, -2.0, 0.6], _LUGGAGE_COLOR),
        _box("hotel_reception_bell_cart", [1.05, 0.61, 1.86], [1.0, 0.0, 0.93], _CART_COLOR),
    ]
    for i, y in enumerate([-3.6, -1.8, 0.0, 1.8, 3.6]):
        props.append(_box(f"hotel_reception_door_{i}", [0.05, 1.0, 1.2], [5.0, y, 0.6],
                           _DOOR_COLOR, collision=False))
    return _rotate_yaw180(props)


# ---------------------------------------------------------------------------
# 6. Hotel scenario -- 酒店场景-清洁服务技能竞技 (housekeeping), 5m x 5m
# ---------------------------------------------------------------------------

def default_hotel_cleaning_props():
    """Guest-room furniture set -- door, sofa, bed, two nightstands, TV cabinet, and
    L-desk -- sizes from 表12, p.35-36. Loose litter items (cans/bottles/paper) the
    real event scatters on the furniture are left out here (props are for
    navigation/collision, not per-item manipulation)."""
    return _rotate_yaw180([
        _box("hotel_cleaning_door", [0.1, 1.0, 1.2], [2.5, 0.0, 0.6], _DOOR_COLOR),
        _box("hotel_cleaning_sofa", [0.8, 0.8, 0.8], [0.5, -1.5, 0.4], _SOFA_COLOR),
        _box("hotel_cleaning_bed", [2.0, 1.8, 0.6], [1.0, 1.5, 0.3], _FURNITURE_COLOR),
        _box("hotel_cleaning_nightstand1", [0.46, 0.45, 0.81], [0.2, 0.6, 0.4], _FURNITURE_COLOR),
        _box("hotel_cleaning_nightstand2", [0.46, 0.45, 0.81], [0.2, 2.4, 0.4], _FURNITURE_COLOR),
        _box("hotel_cleaning_tv_cabinet", [2.4, 0.3, 0.36], [2.0, -2.0, 0.18], _FURNITURE_COLOR),
        _box("hotel_cleaning_desk", [2.4, 0.6, 0.75], [2.0, 2.0, 0.375], _FURNITURE_COLOR),
        _box("hotel_cleaning_trash_bin", [0.3, 0.3, 0.5], [2.3, -0.8, 0.25], _BIN_COLOR),
    ])


# ---------------------------------------------------------------------------
# 7. Warehouse scenario -- 仓储中心场景-混料分拣技能竞技 (mixed-material sorting)
#    5m x 5m -- second-batch rulebook, 表6/图2, p.11-14
# ---------------------------------------------------------------------------

def default_warehouse_sorting_props():
    """Incoming-material station, sorting station (both 长2m宽0.6m高0.7m), and the
    wire-harness transfer cart (长1.27m宽0.62m高1.57m) -- 图2/表6."""
    return _rotate_yaw180([
        _box("warehouse_incoming_station", [0.6, 2.0, 0.7], [1.5, 0.0, 0.35], _TABLE_COLOR),
        _box("warehouse_sorting_station", [2.0, 0.6, 0.7], [1.0, -1.7, 0.35], _TABLE_COLOR),
        _box("warehouse_harness_cart", [0.62, 1.27, 1.57], [1.5, 1.8, 0.785], _CART_COLOR),
    ])


# ---------------------------------------------------------------------------
# 8. 100m-obstacle course -- 100米障碍, the harder ALTERNATIVE to 'race'
#    100m x 3m lane, 10 obstacles -- first-batch rulebook, 图3/表2, p.7-10
# ---------------------------------------------------------------------------

OBSTACLE_LANE_WIDTH = 3.0
_OBSTACLE_WIDTH = 2.4  # footprint width most individual obstacles are specced at
_OBSTACLE_GAP = 0.5    # clear gap left between consecutive obstacles
# Unlike 'rough_terrain' (which has ROUGH_TERRAIN_START_GAP before its first tile),
# the first obstacle here (gravel road) starts right at the start line (x=0) -- so the
# robot's own spawn needs to be the one pulled back instead, same +x setback mechanism
# as ROUGH_TERRAIN_SPAWN_SETBACK (see scenarios.py's 'obstacle_course' entry). Bigger
# than that one (0.15m) -- requested directly after its feet kept burying into the
# first obstacle's raised edge at the smaller setback rough_terrain gets away with.
OBSTACLE_COURSE_SPAWN_SETBACK = 0.3


def default_obstacle_course_props():
    """The 10 obstacles of the 100m-obstacle event, laid out sequentially down -x
    from the start line (x=0), same spawn/direction convention as 'race'/
    'rough_terrain'. Returns (props, total_length) -- total_length is the exact
    distance from the start line to the finish line, for the finish crossing-line
    and the web UI's distance readout (see scenarios.py's 'obstacle_course' entry).

    Ramps/staircases are approximated with box/sphere-only props (see
    _ramp_prop/_staircase_props) -- footprint, obstacle count and step
    counts/heights are the official ones (表2), full 3D wedge/spiral geometry is
    not (Genesis props here don't support arbitrary meshes).
    """
    props = []
    cursor = 0.0  # leading edge of the next obstacle (start of its footprint, x <= cursor)

    def advance(length):
        nonlocal cursor
        far = cursor - length
        cursor = far - _OBSTACLE_GAP
        return far

    # 1. 砾石路 gravel road, 3m x 2.4m
    length = 3.0
    far = advance(length)
    props.append(_box("obstacle_gravel_road", [length, _OBSTACLE_WIDTH, 0.05],
                       [far + length / 2, 0.0, 0.025], [0.55, 0.5, 0.45, 1.0]))

    # 2. 崎岖路 rugged road, 3m x 2.4m
    length = 3.0
    far = advance(length)
    props.append(_box("obstacle_rugged_road", [length, _OBSTACLE_WIDTH, 0.08],
                       [far + length / 2, 0.0, 0.04], [0.5, 0.35, 0.25, 1.0]))

    # 3. 直线台阶 straight stairs, 4m x 2.4m, 5 up + 5 down, 0.3m deep x 0.15m high steps
    length = 4.0
    far = advance(length)
    stair_props, _ = _staircase_props("obstacle_stairs_straight", far + length, 0.0,
                                       _OBSTACLE_WIDTH, num_steps=5, step_depth=0.3, step_height=0.15,
                                       color=[0.75, 0.65, 0.35, 1.0], total_length=length)
    props += stair_props

    # 4. 绕桩 weave poles, 7.4m x 2.4m, 5 poles Ø0.03m spaced 1.1m, staggered
    length = 7.4
    far = advance(length)
    pole_x0 = far + length
    for i in range(5):
        px = pole_x0 - i * 1.1
        py = 0.9 if i % 2 == 0 else -0.9
        props.append(_box(f"obstacle_weave_pole_{i}", [0.03, 0.03, 1.0], [px, py, 0.5],
                           [0.35, 0.25, 0.15, 1.0]))

    # 5. 连续斜坡 continuous slope, 7.4m x 2.4m, 15 deg, 0.3m humps (3 repeats)
    length = 7.4
    far = advance(length)
    hump_x = far + length
    hump_height = 0.3
    hump_run = hump_height / math.tan(math.radians(15))
    for h in range(3):
        peak_x = hump_x - hump_run
        props.append(_ramp_prop(f"obstacle_slope_cont_up_{h}", hump_x, peak_x, 0.0,
                                 _OBSTACLE_WIDTH, 0.05, hump_height, [0.8, 0.6, 0.25, 1.0]))
        valley_x = peak_x - hump_run
        # x_ground/x_top swapped vs. the "up" ramp above -- this side descends
        # FROM the peak (height already reached) back TO ground at valley_x.
        props.append(_ramp_prop(f"obstacle_slope_cont_down_{h}", valley_x, peak_x, 0.0,
                                 _OBSTACLE_WIDTH, 0.05, hump_height, [0.8, 0.6, 0.25, 1.0]))
        hump_x = valley_x

    # 6. 独木桥 balance beam, 3m long x 0.5m wide, deck 0.1m off the ground
    length = 3.0
    far = advance(length)
    props.append(_box("obstacle_balance_beam", [length, 0.5, 0.1], [far + length / 2, 0.0, 0.05],
                       [0.7, 0.5, 0.3, 1.0]))

    # 7. 螺旋台阶 spiral stairs, 5.04m x 3m, 6 up + 6 down, 0.15m steps
    length = 5.04
    far = advance(length)
    step_depth = length / 12
    stair_props, _ = _staircase_props("obstacle_stairs_spiral", far + length, 0.0,
                                       OBSTACLE_LANE_WIDTH, num_steps=6, step_depth=step_depth,
                                       step_height=0.15, color=[0.75, 0.65, 0.35, 1.0], total_length=length)
    props += stair_props

    # 8. 跨栏 hurdle, 2.4m long bar x 0.05m thick x 0.3m tall, spans the lane
    length = 0.05
    far = advance(length)
    props.append(_box("obstacle_hurdle", [0.05, 2.4, 0.3], [far + length / 2, 0.0, 0.15],
                       [0.9, 0.2, 0.2, 1.0]))

    # 9. 交叉斜坡 cross slope, 6m x 3m, 15 deg (approximated as one up/down ramp pair)
    length = 6.0
    far = advance(length)
    ramp_x0 = far + length
    peak_height = (length / 2) * math.tan(math.radians(15))
    peak_x = ramp_x0 - length / 2
    props.append(_ramp_prop("obstacle_cross_slope_up", ramp_x0, peak_x, 0.0,
                             OBSTACLE_LANE_WIDTH, 0.05, peak_height, [0.8, 0.55, 0.2, 1.0]))
    # x_ground/x_top swapped vs. the "up" ramp -- descends from the peak back to ground.
    props.append(_ramp_prop("obstacle_cross_slope_down", ramp_x0 - length, peak_x, 0.0,
                             OBSTACLE_LANE_WIDTH, 0.05, peak_height, [0.8, 0.55, 0.2, 1.0]))

    # 10. 对称斜坡 symmetric slope, 3m x 3m, 20 deg
    length = 3.0
    far = advance(length)
    ramp_x0 = far + length
    peak_height = (length / 2) * math.tan(math.radians(20))
    peak_x = ramp_x0 - length / 2
    props.append(_ramp_prop("obstacle_symmetric_slope_up", ramp_x0, peak_x, 0.0,
                             OBSTACLE_LANE_WIDTH, 0.05, peak_height, [0.85, 0.5, 0.15, 1.0]))
    # x_ground/x_top swapped vs. the "up" ramp -- descends from the peak back to ground.
    props.append(_ramp_prop("obstacle_symmetric_slope_down", ramp_x0 - length, peak_x, 0.0,
                             OBSTACLE_LANE_WIDTH, 0.05, peak_height, [0.85, 0.5, 0.15, 1.0]))

    total_length = -cursor - _OBSTACLE_GAP  # cursor already stepped past obstacle 10's trailing gap
    finish_x = -total_length

    props.append(_crossing_line_prop("obstacle_start_line", x=0.0))
    props.append(_crossing_line_prop("obstacle_finish_line", x=finish_x))
    props += _sign_props("obstacle_start_sign", x=0.0, lane_width=OBSTACLE_LANE_WIDTH,
                          word="START", word_color=[0.08, 0.08, 0.08, 1.0])
    props += _sign_props("obstacle_finish_sign", x=finish_x, lane_width=OBSTACLE_LANE_WIDTH,
                          word="FINISH", word_color=[1.0, 0.55, 0.0, 1.0])
    return props, total_length
