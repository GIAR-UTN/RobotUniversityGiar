"""Rugiar-G1Edu-Velocity-DWAQ + the DWAQ blind-stairs policy, checked against
a REAL mjlab env.

The policy (policies/unitree_g1_29dof_velocity_stairs) is the first external
checkpoint whose contract differs from Rugiar-G1Edu-Velocity-UnitreeObs in
every structural way: 100 current observations (gait phase LAST and 4-dim),
a 5-frame term-major history as a second ONNX input, Isaac Lab joint order,
DWAQ home pose/gains and a flat base arena. Its default PLAY scenario is the
project's `obstacle_course`, not mjlab's random terrain generator.

Run with .venv-mjlab (conftest.py handles sys.path, R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_mjlab_dwaq.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SIMULATOR", "mjlab")

from pathlib import Path

import pytest

mjlab = pytest.importorskip("mjlab")

import mujoco  # noqa: E402
import torch  # noqa: E402

from legged_gym.control.mjlab_adapter import (  # noqa: E402
    MjlabVelocityAdapter, adapter_for_env,
)
from legged_gym.control.mjlab_scenarios import (  # noqa: E402
    apply_scenario_to_mjlab_cfg, default_scenario_for_task,
)

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "policies/unitree_g1_29dof_velocity_stairs/checkpoint.onnx"
TASK = "Rugiar-G1Edu-Velocity-DWAQ"


def _make_env(scenario=None):
    import mjlab_tasks  # noqa: F401 - registers the Rugiar-* tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    if scenario is not None:
        apply_scenario_to_mjlab_cfg(cfg, scenario)
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def test_task_contract():
    import mjlab_tasks  # noqa: F401
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg(TASK, play=True)
    assert list(cfg.observations["actor"].terms) == [
        "base_ang_vel", "projected_gravity", "command",
        "joint_pos", "joint_vel", "actions", "phase",
    ]
    assert cfg.observations["actor"].terms["phase"].params == {
        "period": 0.8, "offset": 0.5,
    }
    assert cfg.scene.entities["robot"].init_state.joint_pos["left_elbow_joint"] == 0.97
    assert cfg.actions["joint_pos"].scale == 0.25
    r = cfg.commands["twist"].ranges
    assert tuple(r.lin_vel_x) == (-0.6, 1.0)
    assert tuple(r.lin_vel_y) == (-0.4, 0.4)
    assert tuple(r.ang_vel_z) == (-1.57, 1.57)
    # The obstacle_course scenario's props are built on top of this plane.
    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert "terrain_scan" not in {s.name for s in (cfg.scene.sensors or ())}


def test_default_scenario_is_obstacle_course():
    assert default_scenario_for_task(TASK, "default") == "obstacle_course"
    # An explicit non-default choice still wins.
    assert default_scenario_for_task(TASK, "race") == "race"
    # Other tasks keep the no-props default.
    assert default_scenario_for_task("Rugiar-G1Edu-Velocity", "default") == "default"


def test_obstacle_course_props_build_on_dwaq_task():
    env = _make_env(scenario="obstacle_course")
    try:
        m = env.sim.mj_model
        g = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, "prop_obstacle_gravel_road_geom"
        )
        assert g >= 0
        env.reset()
        # Same -x-facing start convention as the other course scenarios.
        q = env.scene["robot"].data.root_link_quat_w[0]
        w, x, y, z = q.tolist()
        assert 1 - 2 * (y * y + z * z) < -0.9
    finally:
        env.close()


def test_actor_obs_is_100_and_velocity_adapter_picked():
    env = _make_env()
    try:
        adapter = adapter_for_env(env)
        assert isinstance(adapter, MjlabVelocityAdapter)
        assert adapter.get_observations().shape[-1] == 100
    finally:
        env.close()


def test_dwaq_policy_walks_on_flat():
    if not POLICY.is_file():
        pytest.skip(f"{POLICY} not installed")
    pytest.importorskip("onnxruntime")

    from legged_gym.control import load_policy

    env = _make_env()
    try:
        adapter = adapter_for_env(env)
        pol = load_policy("dwaq", str(POLICY), num_obs=100, hidden_size=0, num_envs=1)
        adapter.reset()
        pol.backend.reset()
        adapter.set_command(0.9, 0.0, 0.0)

        start = adapter.get_state().base_pos_xy.clone()
        obs = adapter.get_observations()
        min_height = float("inf")
        for _ in range(int(5.0 / env.step_dt)):
            with torch.no_grad():
                state = adapter.send_action(pol.backend.step(obs))
            obs = adapter.get_observations()
            min_height = min(min_height, float(state.base_height[0]))
            assert min_height > 0.5, "fell"
        assert float(torch.linalg.norm(adapter.get_state().base_pos_xy - start)) > 2.0
    finally:
        env.close()
