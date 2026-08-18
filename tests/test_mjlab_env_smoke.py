"""Phase 0 of the mjlab migration (see docs/mjlab_migration.md): proves
mjlab builds a G1 tracking env on CPU on this machine and that the actor
observation is the 154-dim vector Javier Villalba's checkpoints expect.

Run with .venv-mjlab, NOT the main .venv, and with `-I` (isolated mode) to
avoid this repo's vendored rsl_rl/ shadowing PyPI rsl-rl-lib -- see R1 in
docs/mjlab_migration.md:

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -I -m pytest \
        tests/test_mjlab_env_smoke.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import pytest

mjlab = pytest.importorskip("mjlab")

REPO = Path(__file__).resolve().parents[1]
MOTION = REPO / "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"

EXPECTED_ACTOR_TERMS = [
    ("command", (58,)),  # ref joint_pos(29) + ref joint_vel(29)
    ("motion_anchor_ori_b", (6,)),  # first 2 cols of the anchor rotation matrix
    ("base_ang_vel", (3,)),
    ("joint_pos", (29,)),  # joint_pos_biased - default
    ("joint_vel", (29,)),
    ("actions", (29,)),
]  # == 154


def test_tracking_env_actor_obs_is_154():
    import torch

    import mjlab.tasks  # noqa: F401  (populates the registry)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    assert MOTION.exists(), f"missing fixture motion: {MOTION}"
    cfg = registry.load_env_cfg(
        "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation", play=True
    )
    cfg.scene.num_envs = 1
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(MOTION)
    motion_cmd.debug_vis = False

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (1, 154)
        assert obs["critic"].shape == (1, 286)
        om = env.observation_manager
        actual = list(
            zip(om.active_terms["actor"], om.group_obs_term_dim["actor"])
        )
        assert actual == EXPECTED_ACTOR_TERMS
        env.step(torch.zeros(1, 29))
    finally:
        env.close()
