"""Phase 3 of the mjlab migration: this repo's own `Rugiar-G1-Mimic` task
(mjlab_tasks/) registers correctly and produces the exact same 154-dim
actor observation as mjlab's stock
Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation -- which is the task
Javier Villalba's checkpoints were trained against (see
docs/mjlab_migration.md §0/§1). Registering our own task_id must not
change the observation contract at all. Run with .venv-mjlab (conftest.py
handles the sys.path ordering, see docs/mjlab_migration.md R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_mjlab_tasks_registration.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import pytest

mjlab = pytest.importorskip("mjlab")

REPO = Path(__file__).resolve().parents[1]
MOTION = REPO / "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"


def test_rugiar_g1_mimic_is_registered():
    import mjlab_tasks  # noqa: F401
    from mjlab.tasks import registry

    assert "Rugiar-G1-Mimic" in registry.list_tasks()


def test_rugiar_g1_mimic_actor_obs_matches_stock_tracking_task():
    import torch

    import mjlab_tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks import registry

    cfg = registry.load_env_cfg("Rugiar-G1-Mimic", play=True)
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_file = str(MOTION)
    cfg.commands["motion"].debug_vis = False

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (1, 154)
        om = env.observation_manager
        assert om.active_terms["actor"] == [
            "command", "motion_anchor_ori_b", "base_ang_vel",
            "joint_pos", "joint_vel", "actions",
        ]
        env.step(torch.zeros(1, 29))
    finally:
        env.close()
