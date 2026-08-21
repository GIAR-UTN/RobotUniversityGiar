"""Phase 1 of the mjlab migration: our raw_run/*.pkl -> mjlab .npz
converter (legged_gym/scripts/process_reference_motion_mjlab.py) produces
a file with the same schema as a known-good mjlab motion, and that file is
loadable/steppable in a real mjlab tracking env. Run with .venv-mjlab, -I:

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -I -m pytest \
        tests/test_process_reference_motion_mjlab.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import subprocess
import sys
from pathlib import Path

import pytest

mjlab = pytest.importorskip("mjlab")

REPO = Path(__file__).resolve().parents[1]
CONVERTER = REPO / "legged_gym/scripts/process_reference_motion_mjlab.py"
RAW_MOTION = "unitree_g1/raw_run/g1moves_B_DadDance.pkl"
OUT_DIR = REPO / "resources/reference_motion/unitree_g1/mjlab_run"
FIXTURE = OUT_DIR / "dance1_subject2.npz"


def test_converts_g1moves_clip_to_mjlab_schema(tmp_path):
    import numpy as np

    out_file = OUT_DIR / "g1moves_B_DadDance.npz"
    if out_file.exists():
        out_file.unlink()

    result = subprocess.run(
        [sys.executable, "-I", str(CONVERTER),
         "--motion-file", RAW_MOTION, "--motion-out-dir", "unitree_g1/mjlab_run"],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert out_file.exists()

    converted = np.load(out_file)
    fixture = np.load(FIXTURE)
    for key in fixture.files:
        assert key in converted.files, f"missing field: {key}"
        assert converted[key].shape[1:] == fixture[key].shape[1:], key
        assert converted[key].dtype == fixture[key].dtype, key
    assert converted["body_pos_w"].shape[0] > 1000  # ~2090 frames expected


def test_converted_clip_loads_and_steps_in_tracking_env():
    import torch

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    motion_path = OUT_DIR / "g1moves_B_DadDance.npz"
    assert motion_path.exists(), "run the converter test first"

    cfg = registry.load_env_cfg(
        "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation", play=True
    )
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_file = str(motion_path)
    cfg.commands["motion"].debug_vis = False

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (1, 154)
        for _ in range(50):
            obs, rew, term, trunc, extra = env.step(torch.zeros(1, 29))
        assert not bool(term.item())
    finally:
        env.close()
