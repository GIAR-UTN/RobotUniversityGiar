"""Phase 5 of the mjlab migration: MjlabAdapter + rugiar_driver_mjlab.py.

Two things are checked here, both against a REAL mjlab env (no mocks):

1. MjlabAdapter satisfies the RobotAdapter contract the rest of
   legged_gym/control/ depends on — RobotState is fully populated with the
   right shapes, and the tracking-task-specific ABSENCES (no set_command /
   set_random_events / set_operator_speed_limit) are asserted explicitly,
   since that absence is exactly what ControlService.status() and
   web/app.js read to gray the Command HUD out (see MjlabAdapter's
   docstring, and docs/mjlab_migration.md R9).

2. The driver's control loop actually drives: the same
   ControlService/PolicySupervisor/SafetyGovernor stack the Genesis
   drivers use, ticking Javier's ONNX checkpoints and switching between
   them live — the headless smoke test's shape, as a repeatable test.

Run with .venv-mjlab (conftest.py handles sys.path, R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_mjlab_adapter_driver.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# legged_gym/__init__.py needs this to import at all; 'mjlab' is the value
# that skips the Genesis import (which would fail — no Genesis in this venv).
os.environ.setdefault("SIMULATOR", "mjlab")

from pathlib import Path

import pytest

mjlab = pytest.importorskip("mjlab")
pytest.importorskip("onnxruntime")

import torch  # noqa: E402

from legged_gym.control import (  # noqa: E402
    ControlService, PolicySupervisor, SafetyGovernor, damping_policy, load_policy,
)
from legged_gym.control.adapter import Lifecycle  # noqa: E402
from legged_gym.control.mjlab_adapter import MjlabAdapter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MOTION = REPO / "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"
CHECKPOINTS = {
    "javier_mjlab_dance1_subject2": REPO / "policies/javier_mjlab_dance1_subject2/checkpoint.onnx",
    "javier_mjlab_model_7000": REPO / "policies/javier_mjlab_model_7000/checkpoint.onnx",
}
NUM_OBS = 154
NUM_ACTIONS = 29


@pytest.fixture(scope="module")
def env():
    import mjlab_tasks  # noqa: F401 - registers Rugiar-G1-Mimic
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg("Rugiar-G1-Mimic", play=True)
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_file = str(MOTION)
    cfg.commands["motion"].debug_vis = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


@pytest.fixture(scope="module")
def adapter(env):
    return MjlabAdapter(env)


def test_backend_identity_and_capabilities(adapter):
    assert adapter.backend_name == "mjlab"
    # ControlService.status()['capabilities'] — web/app.js reads `restart`
    # to enable/disable the Restart button.
    assert adapter.capabilities == {"restart": True}
    assert adapter.num_envs == 1


def test_robot_state_is_fully_populated(adapter):
    state = adapter.get_state()
    assert state.dof_pos.shape == (1, NUM_ACTIONS)
    assert state.dof_vel.shape == (1, NUM_ACTIONS)
    assert state.default_dof_pos.shape == (1, NUM_ACTIONS)
    assert state.base_quat.shape == (1, 4)
    assert state.base_ang_vel.shape == (1, 3)
    assert state.base_lin_vel.shape == (1, 3)
    assert state.projected_gravity.shape == (1, 3)
    assert state.base_height.shape == (1,)
    assert state.commands.shape == (1, 3)
    assert isinstance(state.action_scale, float)
    assert state.lifecycle in (Lifecycle.READY, Lifecycle.ACTIVE)
    # Standing G1: gravity points down in the body frame (this is the exact
    # signal SafetyGovernor's fall threshold reads), pelvis ~0.78m up.
    assert state.projected_gravity[0, 2].item() < -0.9
    assert 0.5 < state.base_height[0].item() < 1.0


def test_no_velocity_command_surface(adapter):
    """A tracking task has no velocity command — these must be ABSENT, not
    stubbed, so ControlService raises NotImplementedError and status()
    omits the keys web/app.js gates the Command HUD on."""
    for missing in ("set_command", "set_operator_speed_limit", "set_random_events",
                    "set_episode_timeout", "command", "random_events", "operator_speed_limit"):
        assert not hasattr(adapter, missing), f"MjlabAdapter unexpectedly exposes {missing}"


def test_status_omits_command_keys(adapter):
    policies = {"damping": damping_policy(adapter.num_envs, NUM_ACTIONS)}
    supervisor = PolicySupervisor(policies, active="damping", ramp_ticks=1)
    service = ControlService(adapter, supervisor, SafetyGovernor(supervisor),
                             task_name="Rugiar-G1-Mimic")
    status = service.status()
    assert status["backend"] == "mjlab"
    assert status["current_task"] == "Rugiar-G1-Mimic"
    assert status["capabilities"] == {"restart": True}
    for key in ("command", "random_events", "operator_speed_limit", "episode_timeout_s"):
        assert key not in status, f"status() should not carry '{key}' on a tracking backend"
    # Telemetry still works — it's what the Live Telemetry panel renders.
    assert status["telemetry"]["base_height"]["value"] is not None
    assert status["telemetry"]["base_lin_vel"]["value"] is not None


def test_control_loop_drives_and_switches(env, adapter):
    """The headless smoke test's shape (see rugiar_driver_mjlab.py's
    run_headless_smoke_test): tick both of Javier's checkpoints through the
    real supervisor/safety stack and switch between them mid-rollout."""
    for path in CHECKPOINTS.values():
        assert path.exists(), f"missing checkpoint: {path}"

    policies = {
        name: load_policy(name, str(path), num_obs=NUM_OBS, hidden_size=0, num_envs=1)
        for name, path in CHECKPOINTS.items()
    }
    policies["damping"] = damping_policy(1, NUM_ACTIONS)
    first, second = list(CHECKPOINTS)

    supervisor = PolicySupervisor(policies, active=first, ramp_ticks=5)
    safety = SafetyGovernor(supervisor, damping_policy_name="damping")
    service = ControlService(adapter, supervisor, safety, task_name="Rugiar-G1-Mimic")

    adapter.reset()
    obs = adapter.get_observations()
    assert obs.shape == (1, NUM_OBS)

    for i in range(60):
        if i == 30:
            assert service.request_switch(second) is True
        action = service.tick(obs)
        assert action is not None and action.shape == (1, NUM_ACTIONS)
        assert not torch.isnan(action).any()
        adapter.send_action(action)
        obs = adapter.get_observations()

    assert supervisor.active_name == second, "live policy switch never confirmed"
    assert not safety.tripped, "the reference policy fell during a 60-step rollout"

    # Pause returns no action at all — the driver feeds zeros in that case.
    service.pause()
    assert service.tick(obs) is None
    service.resume()
    assert service.tick(obs) is not None


def test_driver_dispatches_families_by_registry():
    """rugiar_driver_mjlab._script_for_task must send mjlab tasks to itself
    and everything else back to the Genesis driver (which re-dispatches to
    the gaze one if needed) — the mjlab half of the family routing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rugiar_driver_mjlab", REPO / "legged_gym/scripts/rugiar_driver_mjlab.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    assert driver._script_for_task("Rugiar-G1-Mimic") == "rugiar_driver_mjlab.py"
    assert driver._script_for_task("g1") == "rugiar_driver.py"
    assert driver._script_for_task("g1_gaze") == "rugiar_driver.py"
    # meta.json task guard (a checkpoint trained for another task must not
    # be loaded silently) — same check rugiar_driver.py applies.
    assert driver._sibling_meta_task(
        str(CHECKPOINTS["javier_mjlab_dance1_subject2"])) == "Rugiar-G1-Mimic"
