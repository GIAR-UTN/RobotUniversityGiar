"""S2 (HANDOFF_mimic_motion_library_ux.md): rugiar_driver_mjlab.py must
survive zero local policies for its task instead of raising fatally --
starting a damping-only session (robot holds a neutral pose) so a
reference-motion clip's ghost overlay can still be previewed before
anything is trained against it.

Real components throughout, no mocks: a real mjlab env/MjlabAdapter (same
fixture shape as test_mjlab_adapter_driver.py), a real TrainingManager
pointed (via monkeypatch, same pattern as tests/test_rename_policy.py /
tests/test_fusion.py) at a genuinely empty temp policies/ directory rather
than this repo's real one -- discover_local_policies() then legitimately
finds nothing, it isn't told to pretend it found nothing. Exercises the
actual production function (_load_policies(), extracted from main() for
exactly this reason) plus a real PolicySupervisor/SafetyGovernor/
ControlService tick loop.

Run with .venv-mjlab (conftest.py handles sys.path, R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_mjlab_damping_only_session.py -q
"""
import argparse
import importlib.util
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SIMULATOR", "mjlab")

from pathlib import Path

import pytest

pytest.importorskip("mjlab")

import torch  # noqa: E402

from legged_gym.control.mjlab_adapter import MjlabAdapter  # noqa: E402
from legged_gym.control.safety import SafetyGovernor  # noqa: E402
from legged_gym.control.service import ControlService  # noqa: E402
from legged_gym.control.supervisor import PolicySupervisor  # noqa: E402
from legged_gym.control import training as training_mod  # noqa: E402
from legged_gym.control.training import TrainingManager  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MOTION = REPO / "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"
TASK = "Rugiar-G1-Mimic"


def _load_driver_module():
    spec = importlib.util.spec_from_file_location(
        "rugiar_driver_mjlab", REPO / "legged_gym/scripts/rugiar_driver_mjlab.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


@pytest.fixture(scope="module")
def driver():
    return _load_driver_module()


@pytest.fixture(scope="module")
def env(driver):
    e = driver.build_env(TASK, str(MOTION), "cpu", debug_vis=False)
    yield e
    e.close()


@pytest.fixture(scope="module")
def adapter(env):
    return MjlabAdapter(env)


@pytest.fixture
def empty_policies_dir(monkeypatch, tmp_path):
    """A real, genuinely empty policies/ directory -- not the repo's real
    one, so this test's result doesn't depend on whatever happens to be
    trained on this machine."""
    empty_dir = tmp_path / "policies"
    empty_dir.mkdir()
    monkeypatch.setattr(training_mod, "POLICIES_DIR", empty_dir)
    return empty_dir


def test_zero_local_policies_starts_damping_only(driver, adapter, empty_policies_dir):
    """_load_policies() (S2's fix) must not raise when no --policy specs
    were given and discover_local_policies() finds nothing for this task
    -- it must produce a damping-only policy set with 'damping' active,
    not the old fatal ValueError."""
    num_actions = adapter.env.action_manager.total_action_dim
    num_obs = adapter.get_observations().shape[-1]

    training = TrainingManager()
    assert training.discover_local_policies() == {}, "fixture didn't actually empty the policies dir"

    cli = argparse.Namespace(task=TASK, active=None)
    policies, active_name = driver._load_policies(cli, adapter, num_obs, num_actions, policy_paths={})

    assert active_name == "damping"
    assert set(policies) == {"damping"}


def test_damping_only_session_ticks_without_exception_and_stays_active(driver, adapter, empty_policies_dir):
    """The 'watch it walk' bar for S2: actually build the real
    PolicySupervisor/SafetyGovernor/ControlService stack around the
    damping-only policy set and step it, same shape as
    test_mjlab_adapter_driver.py's test_control_loop_drives_and_switches."""
    num_actions = adapter.env.action_manager.total_action_dim
    num_obs = adapter.get_observations().shape[-1]

    cli = argparse.Namespace(task=TASK, active=None)
    policies, active_name = driver._load_policies(cli, adapter, num_obs, num_actions, policy_paths={})

    supervisor = PolicySupervisor(policies, active=active_name, ramp_ticks=5)
    safety = SafetyGovernor(supervisor, damping_policy_name="damping")
    service = ControlService(adapter, supervisor, safety, task_name=TASK, motion_file=str(MOTION))

    adapter.reset()
    obs = adapter.get_observations()
    assert obs.shape == (1, num_obs)

    for _ in range(40):
        action = service.tick(obs)
        assert action is not None and action.shape == (1, num_actions)
        assert not torch.isnan(action).any()
        adapter.send_action(action)
        obs = adapter.get_observations()

    assert supervisor.active_name == "damping", "no other policy was ever offered -- must still be damping"
    assert not safety.tripped, "the robot fell just holding the default pose for 40 steps"
    status = service.status()
    assert status["active"] == "damping"
