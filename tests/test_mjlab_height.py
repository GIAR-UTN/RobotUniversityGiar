"""Rugiar-G1Edu-Velocity-Height + agile's LEAPP velocity-height policy,
checked against a REAL mjlab env.

What is checked:
- the task contract: 4-wide command (vx, vy, wz, height), 12 leg actions,
  69-dim raw-state actor observation (command, ang_vel, root quat, joint
  pos/vel), LEAPP PD gains and default pose;
- the height channel is pinned by MjlabVelocityAdapter and reaches the
  command term (the same guarantee the plain velocity task has);
- the imported LEAPP ONNX (policies/unitree_g1_velocity_height_history)
  loads through LeappHistoryObsPolicy and actually walks, and the base
  height responds to a changed height command.

Run with .venv-mjlab (conftest.py handles sys.path, R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_mjlab_height.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SIMULATOR", "mjlab")

from pathlib import Path

import pytest

mjlab = pytest.importorskip("mjlab")

import torch  # noqa: E402

from legged_gym.control.mjlab_adapter import (  # noqa: E402
    MjlabVelocityAdapter, adapter_for_env,
)

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "policies/unitree_g1_velocity_height_history/checkpoint.onnx"
TASK = "Rugiar-G1Edu-Velocity-Height"


def _make_env():
    import mjlab_tasks  # noqa: F401 - registers the Rugiar-* tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def test_task_contract():
    import mjlab_tasks  # noqa: F401
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg(TASK, play=True)
    assert list(cfg.observations["actor"].terms) == [
        "command", "base_ang_vel", "root_quat_w", "joint_pos", "joint_vel",
    ]
    assert type(cfg.commands["twist"]).__name__ == "UniformVelocityHeightCommandCfg"
    assert tuple(cfg.commands["twist"].height_range) == (0.4, 0.72)
    assert cfg.commands["twist"].default_height == 0.72
    r = cfg.commands["twist"].ranges
    assert tuple(r.lin_vel_x) == (-0.5, 1.5)
    assert tuple(r.lin_vel_y) == (-0.5, 0.5)
    assert tuple(r.ang_vel_z) == (-1.0, 1.0)

    # 12 leg joints in the action term; the upper body is held at zero.
    assert cfg.actions["joint_pos"].scale == 1.0
    assert cfg.actions["joint_pos"].use_default_offset is False
    assert cfg.actions["joint_pos"].actuator_names == (
        ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint",
    )

    # LEAPP PD gains / default pose, not mjlab's armature-derived values.
    home = cfg.scene.entities["robot"].init_state.joint_pos
    assert home[".*_knee_joint"] == 0.3
    assert home[".*_hip_pitch_joint"] == -0.1
    assert home[".*_ankle_pitch_joint"] == -0.2
    assert home[".*_shoulder_pitch_joint"] == 0.0
    assert "track_base_height" in cfg.rewards


def test_adapter_exposes_and_pins_height():
    env = _make_env()
    try:
        adapter = adapter_for_env(env)
        assert isinstance(adapter, MjlabVelocityAdapter)
        assert len(adapter.command) == 4

        adapter.reset()
        adapter.set_command(0.4, -0.1, 0.2, height=0.55)
        assert adapter.command == pytest.approx((0.4, -0.1, 0.2, 0.55))
        term = env.command_manager.get_term("twist")
        assert float(term.height_command[0]) == pytest.approx(0.55)

        # Out-of-range height is clamped to the trained envelope.
        adapter.set_command(0.0, 0.0, 0.0, height=99.0)
        assert adapter.command[3] == pytest.approx(0.72)

        zero = torch.zeros(1, env.action_manager.total_action_dim)
        for _ in range(5):
            adapter.send_action(zero)
        assert float(term.height_command[0]) == pytest.approx(0.72)
        assert env.command_manager.get_command("twist").shape == (1, 4)
    finally:
        env.close()


def test_control_service_sets_and_reports_height():
    from legged_gym.control import (
        ControlService, PolicySupervisor, SafetyGovernor, damping_policy,
    )

    env = _make_env()
    try:
        adapter = adapter_for_env(env)
        supervisor = PolicySupervisor(
            {"damping": damping_policy(1, env.action_manager.total_action_dim)},
            active="damping", ramp_ticks=1,
        )
        service = ControlService(
            adapter, supervisor, SafetyGovernor(supervisor), task_name=TASK,
        )
        service.set_command(0.3, -0.1, 0.2, height=0.6)
        status = service.status()
        assert status["command"] == pytest.approx(
            {"vx": 0.3, "vy": -0.1, "yaw": 0.2, "height": 0.6}
        )
    finally:
        env.close()


def test_control_service_rejects_height_on_plain_velocity_adapter():
    from legged_gym.control import (
        ControlService, PolicySupervisor, SafetyGovernor, damping_policy,
    )

    class PlainAdapter:
        backend_name = "sim"
        capabilities = {"restart": True}
        command = (0.0, 0.0, 0.0)

        def set_command(self, vx, vy, yaw):
            self.called = (vx, vy, yaw)

    adapter = PlainAdapter()
    supervisor = PolicySupervisor(
        {"damping": damping_policy(1, 12)}, active="damping", ramp_ticks=1,
    )
    service = ControlService(adapter, supervisor, SafetyGovernor(supervisor))
    service.set_command(0.1, 0.0, 0.0)
    assert adapter.called == (0.1, 0.0, 0.0)
    with pytest.raises(NotImplementedError):
        service.set_command(0.1, 0.0, 0.0, height=0.6)


def test_policy_walks_and_height_responds():
    if not POLICY.is_file():
        pytest.skip(f"{POLICY} not installed")
    pytest.importorskip("onnxruntime")

    from legged_gym.control import load_policy

    env = _make_env()
    try:
        adapter = adapter_for_env(env)
        num_actions = env.action_manager.total_action_dim
        assert num_actions == 12
        pol = load_policy(
            "height", str(POLICY), num_obs=69, hidden_size=0,
            num_envs=1, num_actions=num_actions,
        )
        adapter.reset()
        pol.backend.reset()
        adapter.set_command(0.5, 0.0, 0.0, height=0.72)

        start = adapter.get_state().base_pos_xy.clone()
        obs = adapter.get_observations()
        min_height = float("inf")
        for _ in range(int(8.0 / env.step_dt)):
            with torch.no_grad():
                state = adapter.send_action(pol.backend.step(obs))
            obs = adapter.get_observations()
            min_height = min(min_height, float(state.base_height[0]))
            assert min_height > 0.5, "fell"
        assert float(torch.linalg.norm(adapter.get_state().base_pos_xy - start)) > 2.0

        # Height command response: stand at 0.55 m, then back to 0.72 m.
        heights = {}
        for target in (0.55, 0.72):
            adapter.set_command(0.0, 0.0, 0.0, height=target)
            samples = []
            for _ in range(int(3.0 / env.step_dt)):
                with torch.no_grad():
                    state = adapter.send_action(pol.backend.step(obs))
                obs = adapter.get_observations()
                samples.append(float(state.base_height[0]))
            heights[target] = sum(samples[-50:]) / len(samples[-50:])
        assert heights[0.55] < heights[0.72] - 0.02
    finally:
        env.close()
