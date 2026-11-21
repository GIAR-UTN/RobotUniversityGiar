"""OnnxHistoryObsPolicy + the deploy.yaml contract parser, without mjlab.

The DWAQ policy in policies/unitree_g1_29dof_velocity_stairs is the first
checkpoint in this repo whose ONNX has two inputs: `history` (5 term-major
frames) and `obs` (the current frame). These tests pin down the pieces that
used to make it load as an LSTM (wrong) or feed the wrong layout (also
wrong):

- deploy.yaml -> HistoryObsSpec (term order, history length, joint map);
- the wrapper's term-major history, joint remap, and reset re-seeding,
  checked against a manual reference implementation of the same contract;
- the clear errors for a missing deploy.yaml / a mismatched task.

Run in the main .venv:

    .venv/bin/python -m pytest tests/test_policy_history_obs.py -q
"""
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from legged_gym.control.policy import (  # noqa: E402
    HistoryObsSpec,
    OnnxHistoryObsPolicy,
    history_obs_spec_from_deploy_yaml,
    load_policy,
)

REPO = Path(__file__).resolve().parents[1]
POLICY_DIR = REPO / "policies/unitree_g1_29dof_velocity_stairs"
POLICY = POLICY_DIR / "checkpoint.onnx"
DEPLOY = POLICY_DIR / "deploy.yaml"

pytestmark = pytest.mark.skipif(
    not (POLICY.is_file() and DEPLOY.is_file()),
    reason="unitree_g1_29dof_velocity_stairs is not installed",
)

EXPECTED_JOINT_MAP = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)


def test_deploy_yaml_contract():
    spec = history_obs_spec_from_deploy_yaml(str(DEPLOY))
    assert spec is not None
    assert spec.history_length == 5
    assert spec.term_sizes == (3, 3, 3, 29, 29, 29, 4)
    assert spec.num_obs == 100
    assert spec.joint_slices == (9, 38, 67)
    assert spec.joint_map == EXPECTED_JOINT_MAP


def _reference_step(session, spec, frames, obs):
    current = obs.clone()
    map_t = torch.tensor(spec.joint_map, dtype=torch.long)
    for start in spec.joint_slices:
        n = len(spec.joint_map)
        current[:, start:start + n] = obs[:, start:start + n][:, map_t]
    if not frames:
        frames.extend([current.clone() for _ in range(spec.history_length)])
    else:
        frames.append(current.clone())
        frames.pop(0)
    ranges = []
    offset = 0
    for size in spec.term_sizes:
        ranges.append((offset, offset + size))
        offset += size
    history = torch.cat(
        [torch.cat([f[:, a:b] for f in frames], dim=1) for a, b in ranges], dim=1
    )
    (action_np,) = session.run(
        None,
        {
            "history": history.numpy().astype(np.float32),
            "obs": current.numpy().astype(np.float32),
        },
    )
    action = torch.from_numpy(action_np)
    env_action = torch.empty_like(action)
    env_action[:, map_t] = action
    return env_action


def test_wrapper_matches_reference_for_a_sequence():
    spec = history_obs_spec_from_deploy_yaml(str(DEPLOY))
    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])
    wrapper = OnnxHistoryObsPolicy(session, spec)

    torch.manual_seed(0)
    reference_frames = []
    for _ in range(4):
        obs = torch.randn(1, spec.num_obs)
        expected = _reference_step(session, spec, reference_frames, obs)
        actual = wrapper.step(obs)
        assert torch.allclose(actual, expected, atol=1e-5), (actual - expected).abs().max()


def test_reset_reseeds_history_with_current_obs():
    spec = history_obs_spec_from_deploy_yaml(str(DEPLOY))
    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])
    wrapper = OnnxHistoryObsPolicy(session, spec)
    obs = torch.randn(1, spec.num_obs)

    first = wrapper.step(obs)
    wrapper.step(torch.randn(1, spec.num_obs))
    wrapper.reset()
    again = wrapper.step(obs)
    assert torch.allclose(first, again, atol=1e-5)


def test_missing_deploy_yaml_is_a_clear_error(tmp_path):
    import shutil

    copy = tmp_path / "checkpoint.onnx"
    shutil.copy(POLICY, copy)
    with pytest.raises(RuntimeError, match="deploy.yaml"):
        load_policy("dwaq", str(copy), num_obs=100, hidden_size=0, num_envs=1)


def test_wrong_task_obs_is_a_clear_error():
    with pytest.raises(RuntimeError, match="100-dim current"):
        load_policy("dwaq", str(POLICY), num_obs=98, hidden_size=0, num_envs=1)


def test_obs_spec_is_current_frame_not_history():
    pol = load_policy("dwaq", str(POLICY), num_obs=100, hidden_size=0, num_envs=1)
    assert pol.obs_spec.num_obs == 100
    assert pol.backend.step(torch.zeros(1, 100)).shape == (1, 29)


def test_history_obs_spec_is_a_plain_dataclass():
    spec = HistoryObsSpec(history_length=2, term_sizes=(3, 4), joint_slices=(3,))
    assert spec.num_obs == 7
