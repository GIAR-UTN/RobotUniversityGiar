"""WHRG / RUgiar scenarios (legged_gym/utils/scenarios.py) on the mjlab backend.

The scenario registry and its prop lists stay the single source of truth. The
Genesis drivers build them through genesis_simulator.py::_create_props(); this
module builds the SAME prop dicts into mjlab's MjSpec via SceneCfg.spec_fn, so
an arena looks and collides the same in both simulators.

Supported here: what the scenarios actually use, i.e. box and sphere props,
`fixed` or free, with optional `euler`, `color`, `collision`, `friction` and
`density`. Plus the robot's spawn pose (init_state_rot, init_state_pos_offset).

Also mirrored from the Genesis side:
- Free props (e.g. the ball) go back to their spawn pose on every reset,
  like genesis_simulator.py::_reset_props(). That is a `reset` event
  (reset_props).
- fail_to_terminal_time_s: a failure (fall) must HOLD that long before the
  episode ends, so the crash stays on screen (legged_robot.py's
  fail_buf > fail_to_terminal_time_s / dt). Every non-timeout termination is
  wrapped in held_termination().

Import note: `legged_gym.utils` can't be imported under .venv-mjlab. Its
__init__ pulls helpers.py, which needs this repo's vendored rsl_rl, and PyPI
rsl-rl-lib wins there (docs/mjlab_migration.md R1). props.py,
competition_props.py and scenarios.py themselves are pure Python, so
load_scenarios() loads them by file path under their real module names,
behind a bare stand-in `legged_gym.utils` package. It does that only when the
real package isn't importable, so a Genesis process is never affected.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
from typing import Optional

_UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"


def load_scenarios():
    """Returns the legged_gym.utils.scenarios module, importable under
    .venv-mjlab (see module docstring)."""
    name = "legged_gym.utils.scenarios"
    if name in sys.modules:
        return sys.modules[name]
    try:
        import legged_gym.utils.scenarios as mod  # noqa: F401 - the normal path (Genesis venv)
        return mod
    except Exception:
        pass
    # The failed import above may have left a half-initialized package behind.
    for stale in [k for k in sys.modules if k == "legged_gym.utils" or k.startswith("legged_gym.utils.")]:
        del sys.modules[stale]
    pkg = types.ModuleType("legged_gym.utils")
    pkg.__path__ = [str(_UTILS_DIR)]
    sys.modules["legged_gym.utils"] = pkg
    for sub in ("props", "competition_props", "scenarios"):
        full = f"legged_gym.utils.{sub}"
        spec = importlib.util.spec_from_file_location(full, _UTILS_DIR / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, sub, mod)
    return sys.modules[name]


def _euler_deg_to_quat_wxyz(euler):
    """Genesis morph `euler` is in degrees, with extrinsic x-y-z order
    (scipy 'xyz'). Returns the MuJoCo wxyz quaternion."""
    rx, ry, rz = (math.radians(a) / 2.0 for a in euler)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # R = Rz * Ry * Rx (extrinsic xyz)
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    return [w, x, y, z]


def legged_gym_quat_to_wxyz(rot):
    """legged_gym's init_state.rot is xyzw (Isaac Gym convention, also what
    RACE_SPAWN_ROT is written in); mjlab/MuJoCo use wxyz."""
    x, y, z, w = rot
    return (w, x, y, z)


def add_props_to_spec(spec, props: list) -> None:
    """Adds Genesis-style prop dicts to a MuJoCo MjSpec's worldbody."""
    import mujoco

    world = spec.worldbody
    for p in props:
        shape = p.get("shape", "sphere")
        pos = list(p.get("pos", [0.5, 0.0, 0.5]))
        fixed = p.get("fixed", False)
        quat = _euler_deg_to_quat_wxyz(p["euler"]) if p.get("euler") is not None else [1, 0, 0, 0]
        body = world.add_body(name=f"prop_{p['name']}", pos=pos, quat=quat)
        if not fixed:
            body.add_freejoint(name=f"prop_{p['name']}_free")
        geom_kw = {}
        if shape == "box":
            geom_kw["type"] = mujoco.mjtGeom.mjGEOM_BOX
            geom_kw["size"] = [s / 2.0 for s in p.get("size", [0.1, 0.1, 0.1])]  # MuJoCo: half extents
        elif shape == "sphere":
            r = p.get("size", 0.1)
            geom_kw["type"] = mujoco.mjtGeom.mjGEOM_SPHERE
            geom_kw["size"] = [r, 0, 0]
        else:
            raise ValueError(f"Unsupported prop shape: {shape}")
        color = p.get("color")
        if color is not None:
            rgba = list(color) + [1.0] * (4 - len(color))
            geom_kw["rgba"] = rgba
        if not p.get("collision", True):
            geom_kw["contype"] = 0
            geom_kw["conaffinity"] = 0
        if p.get("friction") is not None:
            geom_kw["friction"] = [float(p["friction"]), 0.005, 0.0001]
        if not fixed:
            geom_kw["density"] = float(p.get("density", 1000.0))
        body.add_geom(name=f"prop_{p['name']}_geom", **geom_kw)


def reset_props(env, env_ids, joint_names: tuple) -> None:
    """mjlab `reset` event: puts every free prop back at its spawn pose
    (model qpos0) with zero velocity. The props are plain spec bodies, not
    mjlab entities, so no built-in reset touches them."""
    import mujoco

    if not joint_names:
        return
    m = env.sim.mj_model
    if env_ids is None:
        env_ids = slice(None)
    for jname in joint_names:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qa, da = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        qpos0 = torch.as_tensor(m.qpos0[qa:qa + 7], dtype=torch.float32, device=env.device)
        env.sim.data.qpos[env_ids, qa:qa + 7] = qpos0
        env.sim.data.qvel[env_ids, da:da + 6] = 0.0


def held_termination(env, inner, hold_steps: int, key: str, **params):
    """Wraps a termination so that it fires only after its condition has
    held `hold_steps` consecutive steps (Genesis's fail_to_terminal_time_s).
    The per-env counter lives on the env, under its own key per term."""
    cond = inner(env, **params)
    attr = f"_rugiar_hold_{key}"
    buf = getattr(env, attr, None)
    if buf is None or buf.shape[0] != cond.shape[0]:
        buf = torch.zeros(cond.shape[0], dtype=torch.long, device=cond.device)
        setattr(env, attr, buf)
    buf.copy_(torch.where(cond, buf + 1, torch.zeros_like(buf)))
    fire = buf >= hold_steps
    buf[fire] = 0
    return fire


# Tasks whose PLAY session should open on a scenario other than 'default'
# when the operator didn't pass --scenario. The DWAQ task is the blind
# obstacle/stairs policy, so the project's obstacle course (flat lane plus
# its 10 props) is its default arena; mjlab's random terrain generator is
# not what the policy was trained against.
TASK_DEFAULT_SCENARIOS = {
    "Rugiar-G1Edu-Velocity-DWAQ": "obstacle_course",
}


def default_scenario_for_task(task: str, requested: Optional[str]) -> Optional[str]:
    """The scenario a driver should actually use: an explicit non-'default'
    --scenario always wins; otherwise a task in TASK_DEFAULT_SCENARIOS gets
    its own default. Tasks not in the map keep the parser's 'default'
    (no props), so this is a no-op for every existing task."""
    if requested and requested != "default":
        return requested
    return TASK_DEFAULT_SCENARIOS.get(task, requested)


def apply_scenario_to_mjlab_cfg(cfg, scenario_name: Optional[str], option_overrides: Optional[list] = None):
    """Wires a scenario into an mjlab ManagerBasedRlEnvCfg (a PLAY cfg,
    num_envs=1). Returns (scenario, options), or (None, {}) when there is no
    scenario.

    - The props become worldbody geoms (SceneCfg.spec_fn, chained after any
      spec_fn the task already has).
    - The robot's spawn pose takes init_state_rot / init_state_pos_offset.
      mjlab's own reset_base pose randomization (±0.5 m, any yaw) is zeroed,
      otherwise the robot would spawn facing a random way inside the arena.
    """
    if not scenario_name or scenario_name == "default":
        return None, {}
    sc = load_scenarios()
    ns = types.SimpleNamespace(scenario=scenario_name, scenario_option=list(option_overrides or []))
    scenario, options = sc.resolve_scenario(ns)
    props = scenario.spawn_props(options)

    # Every prop is another collider in this single-world PLAY scene. The flat
    # G1 task leaves cfg.sim.nconmax at None, whose heuristic allocates ~48
    # contacts -- not enough once the course props touch the robot ("broadphase
    # overflow - please increase nconmax to 49..."). Scale it with the prop
    # count, never lowering a value the task set explicitly.
    if props:
        cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 64 + 8 * len(props))

    prev_fn = cfg.scene.spec_fn

    def _spec_fn(spec):
        if prev_fn is not None:
            prev_fn(spec)
        add_props_to_spec(spec, props)

    cfg.scene.spec_fn = _spec_fn

    robot = cfg.scene.entities["robot"]
    if scenario.init_state_rot is not None:
        robot.init_state.rot = legged_gym_quat_to_wxyz(scenario.init_state_rot)
    if scenario.init_state_pos_offset is not None:
        robot.init_state.pos = tuple(a + b for a, b in zip(robot.init_state.pos, scenario.init_state_pos_offset))

    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

    free_joints = tuple(f"prop_{p['name']}_free" for p in props if not p.get("fixed", False))
    if free_joints:
        cfg.events["reset_props"] = EventTermCfg(func=reset_props, mode="reset",
                                                 params={"joint_names": free_joints})

    if scenario.fail_to_terminal_time_s:
        step_dt = cfg.decimation * cfg.sim.mujoco.timestep
        hold_steps = max(1, int(round(scenario.fail_to_terminal_time_s / step_dt)))
        for name, term in list(cfg.terminations.items()):
            if term.time_out:
                continue
            cfg.terminations[name] = TerminationTermCfg(
                func=held_termination, time_out=False,
                params={"inner": term.func, "hold_steps": hold_steps, "key": name, **term.params})

    reset_base = cfg.events.get("reset_base")
    if reset_base is not None and "pose_range" in reset_base.params:
        reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.01),
                                           "yaw": (0.0, 0.0)}
    return scenario, options
