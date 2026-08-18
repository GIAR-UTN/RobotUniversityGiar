"""Phase 4 of the mjlab migration -- the actual validation milestone: are
Javier Villalba's real trained checkpoints usable, closed-loop, against
this repo's own registered mjlab task? Drives each ONNX checkpoint
directly (onnxruntime, single [1,154]->[1,29] I/O, no external runner)
against Rugiar-G1-Mimic's play env, for 400 steps against the
dance1_subject2 motion, and checks it doesn't fall and tracks reasonably.

Thresholds have margin for MuJoCo Warp's non-determinism (see
docs/mjlab_migration.md R4) -- don't tighten these to the exact numbers
measured once. Run with .venv-mjlab (conftest.py handles sys.path, R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_javier_checkpoints_track.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import numpy as np
import pytest

mjlab = pytest.importorskip("mjlab")
onnxruntime = pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[1]
MOTION = REPO / "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"
STEPS = 400

CHECKPOINTS = {
    # (checkpoint path, max allowed terminations, max allowed mean body-pos error [m])
    "javier_mjlab_dance1_subject2": (
        REPO / "policies/javier_mjlab_dance1_subject2/checkpoint.onnx", 0, 0.15,
    ),
    "javier_mjlab_model_7000": (
        # Trained against an unknown motion (not necessarily dance1_subject2,
        # see docs/mjlab_migration.md R3) -- not a quality baseline, just
        # proving it runs closed-loop without erroring, with generous margin.
        REPO / "policies/javier_mjlab_model_7000/checkpoint.onnx", 20, 0.30,
    ),
}


def _rollout(checkpoint_path):
    import torch

    import mjlab_tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg("Rugiar-G1-Mimic", play=True)
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_file = str(MOTION)
    cfg.commands["motion"].debug_vis = False

    session = onnxruntime.InferenceSession(str(checkpoint_path))
    input_name = session.get_inputs()[0].name

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (1, 154)

        terminations = 0
        body_pos_errs = []
        motion_cmd = env.command_manager.get_term("motion")

        for _ in range(STEPS):
            actions = session.run(
                None, {input_name: obs["actor"].numpy().astype(np.float32)}
            )[0]
            obs, rew, term, trunc, extra = env.step(torch.from_numpy(actions))
            if bool(term.item()):
                terminations += 1
                obs, _ = env.reset()
            err = (motion_cmd.robot_body_pos_w - motion_cmd.body_pos_w).norm(dim=-1)
            body_pos_errs.append(err.mean().item())

        return terminations, float(np.mean(body_pos_errs))
    finally:
        env.close()


@pytest.mark.parametrize("name", list(CHECKPOINTS.keys()))
def test_checkpoint_tracks_closed_loop(name):
    checkpoint_path, max_terminations, max_mean_err = CHECKPOINTS[name]
    assert checkpoint_path.exists(), f"missing checkpoint: {checkpoint_path}"

    terminations, mean_err = _rollout(checkpoint_path)
    print(f"{name}: {STEPS} steps, terminations={terminations}, mean_body_pos_err={mean_err:.4f} m")

    assert terminations <= max_terminations, (
        f"{name} fell {terminations} times in {STEPS} steps (max allowed {max_terminations})"
    )
    assert mean_err <= max_mean_err, (
        f"{name} mean body-pos error {mean_err:.4f}m exceeds {max_mean_err}m"
    )
