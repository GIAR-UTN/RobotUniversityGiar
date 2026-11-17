"""Scenarios on the mjlab backend (legged_gym/control/mjlab_scenarios.py).

Checked against a REAL mjlab env:
- The props of a scenario end up in the MuJoCo model. Box size is converted
  to half extents, and fixed props have no joint.
- The spawn pose follows the scenario: race faces -x (RACE_SPAWN_ROT),
  converted xyzw -> wxyz. mjlab's random reset pose is disabled.
- A free prop (the ball) goes back to its spawn pose on reset.
- held_termination fires only after the condition held N steps (Genesis's
  fail_to_terminal_time_s).
- A relaunch keeps --scenario (the driver's _base_relaunch_argv).

Run with .venv-mjlab:

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest tests/test_mjlab_scenarios.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SIMULATOR", "mjlab")

import argparse
import types

import pytest

mjlab = pytest.importorskip("mjlab")

import mujoco  # noqa: E402
import torch  # noqa: E402

from legged_gym.control.mjlab_scenarios import (  # noqa: E402
    apply_scenario_to_mjlab_cfg, held_termination, load_scenarios,
)

TASK = "Rugiar-G1Edu-Velocity-UnitreeObs"


def _env(scenario, options=None):
    import mjlab_tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    apply_scenario_to_mjlab_cfg(cfg, scenario, options)
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def test_race_props_and_spawn_pose():
    sc = load_scenarios()
    scenario, opts = sc.resolve_scenario(argparse.Namespace(scenario="race", scenario_option=[]))
    props = scenario.spawn_props(opts)
    env = _env("race")
    try:
        m = env.sim.mj_model
        box = next(p for p in props if p.get("shape") == "box")
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"prop_{box['name']}_geom")
        assert g >= 0
        assert list(m.geom_size[g]) == pytest.approx([s / 2 for s in box["size"]])
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"prop_{box['name']}")
        assert m.body_jntnum[b] == 0  # fixed

        # Facing -x: the body x axis, rotated by the root quat, points to -x.
        # The spawn pose is applied by mjlab's reset events, not at build time.
        env.reset()
        q = env.scene["robot"].data.root_link_quat_w[0]
        w, x, y, z = q.tolist()
        fwd_x = 1 - 2 * (y * y + z * z)
        assert fwd_x < -0.9
        assert env.scene["robot"].data.root_link_pos_w[0, :2].abs().max() < 0.05
    finally:
        env.close()


def test_free_prop_resets_to_spawn():
    env = _env("ball")
    try:
        m = env.sim.mj_model
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "prop_ball_free")
        qa = int(m.jnt_qposadr[j])
        spawn = torch.as_tensor(m.qpos0[qa:qa + 3], dtype=torch.float32)
        env.sim.data.qpos[0, qa:qa + 3] = spawn + torch.tensor([3.0, -2.0, 0.0])
        env.reset()
        assert torch.allclose(env.sim.data.qpos[0, qa:qa + 3], spawn, atol=1e-5)
    finally:
        env.close()


def test_held_termination_waits_hold_steps():
    env = types.SimpleNamespace()
    state = {"cond": torch.tensor([True, False])}
    inner = lambda e: state["cond"]  # noqa: E731
    fired = [held_termination(env, inner, hold_steps=3, key="t").tolist() for _ in range(3)]
    assert fired == [[False, False], [False, False], [True, False]]
    # The counter restarts after firing and when the condition clears.
    assert held_termination(env, inner, hold_steps=3, key="t").tolist() == [False, False]
    state["cond"] = torch.tensor([False, False])
    held_termination(env, inner, hold_steps=3, key="t")
    assert env._rugiar_hold_t.tolist() == [0, 0]


def test_scenario_wraps_terminations_with_hold():
    import mjlab_tasks  # noqa: F401
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg(TASK, play=True)
    apply_scenario_to_mjlab_cfg(cfg, "race")
    fell = cfg.terminations["fell_over"]
    assert fell.func is held_termination and fell.params["hold_steps"] > 1
    assert cfg.terminations["time_out"].func is not held_termination


def test_relaunch_keeps_scenario():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "legged_gym/scripts/rugiar_driver_mjlab.py"
    spec = importlib.util.spec_from_file_location("rugiar_driver_mjlab_t", path)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    cli = argparse.Namespace(viser_port=9021, ramp_ticks=25, control_port=9017, token=None,
                             scenario="obstacle_course", scenario_option=["foo=1"],
                             cam_azimuth=None, cam_elevation=None, cam_distance=None)
    argv = drv._base_relaunch_argv(cli, "x/rugiar_driver_mjlab.py", "python", TASK)
    i = argv.index("--scenario")
    assert argv[i + 1] == "obstacle_course"
    assert argv[argv.index("--scenario-option") + 1] == "foo=1"
