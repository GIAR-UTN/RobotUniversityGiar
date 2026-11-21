"""MjlabVelocityAdapter + the G1-EDU velocity tasks (Rugiar-G1Edu-Velocity
and Rugiar-G1Edu-Velocity-UnitreeObs), checked against a REAL mjlab env.

What is checked:
- adapter_for_env() picks the velocity adapter for a `twist` task, and the
  adapter exposes the command surface that ControlService.status() and
  web/app.js look for.
- A manual command is clamped to the trained envelope times
  operator_speed_limit, and it stays PINNED across steps. mjlab's own
  resampling, heading and standing logic must not overwrite it. The
  command slots of the actor observation must carry it on the very next
  observation.
- Rugiar-G1Edu-Velocity removes base_lin_vel from the actor (96 values,
  real-deployable), and the UnitreeObs variant matches unitree_rl_mjlab's
  98-value contract.
- When unitree_rl_mjlab's published policy is installed under policies/,
  it walks: no falls over 10 s at 0.5 m/s, with forward progress.

Run with .venv-mjlab (conftest.py handles sys.path, R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_mjlab_velocity_adapter.py -q
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
UNITREE_POLICY = REPO / "policies/unitree_g1_29dof_velocity_v0/checkpoint.onnx"


def _make_env(task):
    import mjlab_tasks  # noqa: F401 - registers the Rugiar-* tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


@pytest.fixture(scope="module")
def env():
    e = _make_env("Rugiar-G1Edu-Velocity")
    yield e
    e.close()


@pytest.fixture(scope="module")
def adapter(env):
    return adapter_for_env(env)


def test_picks_velocity_adapter_with_command_surface(adapter):
    assert isinstance(adapter, MjlabVelocityAdapter)
    assert "motion" not in adapter.capabilities
    for attr in ("command", "set_command", "operator_speed_limit", "set_operator_speed_limit"):
        assert hasattr(adapter, attr), attr


def test_actor_obs_is_real_deployable(adapter, env):
    terms = list(env.cfg.observations["actor"].terms)
    assert "base_lin_vel" not in terms
    assert adapter.get_observations().shape[-1] == 96


def test_command_clamped_to_trained_envelope(adapter):
    r = adapter._term().cfg.ranges
    adapter.set_operator_speed_limit(1.0)
    adapter.set_command(99.0, -99.0, 99.0)
    assert adapter.command == (r.lin_vel_x[1], r.lin_vel_y[0], r.ang_vel_z[1])
    adapter.set_operator_speed_limit(0.5)
    assert adapter.command[0] == pytest.approx(r.lin_vel_x[1] * 0.5)
    adapter.set_operator_speed_limit(1.0)
    with pytest.raises(ValueError):
        adapter.set_operator_speed_limit(0.0)


def test_command_stays_pinned_and_reaches_obs(adapter, env):
    adapter.reset()
    adapter.set_command(0.4, -0.1, 0.2)
    zero = torch.zeros(1, env.action_manager.total_action_dim)
    # Long enough to cross mjlab's resampling window in the play cfg.
    for _ in range(int(12.0 / env.step_dt)):
        adapter.send_action(zero)
        if float(adapter.get_state().base_height[0]) < 0.3:
            adapter.reset()
    cmd = env.command_manager.get_command("twist")[0]
    assert torch.allclose(cmd, torch.tensor([0.4, -0.1, 0.2]), atol=1e-6)
    # Command is the actor group's last term (3 values).
    assert torch.allclose(adapter.get_observations()[0, -3:], torch.tensor([0.4, -0.1, 0.2]), atol=1e-6)
    assert adapter.get_state().commands[0].tolist() == pytest.approx([0.4, -0.1, 0.2])


def test_unitree_obs_variant_matches_published_contract():
    e = _make_env("Rugiar-G1Edu-Velocity-UnitreeObs")
    try:
        ad = adapter_for_env(e)
        assert list(e.cfg.observations["actor"].terms) == [
            "base_ang_vel", "projected_gravity", "command", "phase",
            "joint_pos", "joint_vel", "actions"]
        assert ad.get_observations().shape[-1] == 98

        if not UNITREE_POLICY.exists():
            pytest.skip(f"{UNITREE_POLICY} not installed")
        pytest.importorskip("onnxruntime")
        from legged_gym.control import load_policy

        pol = load_policy("unitree", str(UNITREE_POLICY), num_obs=98, hidden_size=0, num_envs=1)
        ad.reset()
        ad.set_command(0.5, 0.0, 0.0)
        start = ad.get_state().base_pos_xy.clone()
        obs = ad.get_observations()
        for _ in range(int(10.0 / e.step_dt)):
            with torch.no_grad():
                st = ad.send_action(pol.backend.step(obs))
            obs = ad.get_observations()
            assert float(st.base_height[0]) > 0.5, "fell"
        assert float(torch.linalg.norm(ad.get_state().base_pos_xy - start)) > 3.0
    finally:
        e.close()
