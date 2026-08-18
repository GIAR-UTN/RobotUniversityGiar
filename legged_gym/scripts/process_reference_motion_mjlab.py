"""Converts one of this repo's raw reference-motion .pkl files (see
process_reference_motion.py's docstring for that schema: fps/root_pos/
root_rot xyzw/dof_pos) into the .npz format mjlab's MotionCommand expects
(joint_pos/joint_vel/body_{pos,quat,lin_vel,ang_vel}_w, quat wxyz).

Ported from third_party/unitree_rl_mjlab/csv_to_npz.py.reference — same
MotionLoader/run_sim structure, with the input side swapped from a 36-col
CSV to this repo's raw_run/*.pkl, and importing mjlab's OWN
unitree_g1_flat_tracking_env_cfg (not the unitree_rl_mjlab fork's copy —
see docs/mjlab_migration.md §0 for why). Run with .venv-mjlab, -I flag
(see docs/mjlab_migration.md R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -I \
        legged_gym/scripts/process_reference_motion_mjlab.py \
        --motion_file unitree_g1/raw_run/g1moves_B_DadDance.pkl \
        --motion_out_dir unitree_g1/mjlab_run
"""
import os
import pickle
import sys
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)

# Same schema our raw_run/*.pkl files already use — see
# process_reference_motion.py's docstring. Confirmed (docs/
# motion_imitation_integration.md) this joint order is identical to
# G1Flat29DofCommonCfg.dof_names, which is identical to the order below.
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Pickle files saved with NumPy >= 2.0 reference numpy._core, which
# doesn't exist in NumPy < 2.0 -- same shim process_reference_motion.py
# carries, kept here since this script also unpickles raw_run/*.pkl.
if not hasattr(np, "_core"):
    import numpy.core as _np_core
    for _attr in dir(_np_core):
        _mod_name = f"numpy._core.{_attr}"
        _submod = getattr(_np_core, _attr, None)
        if isinstance(_submod, type(sys)):
            sys.modules.setdefault(_mod_name, _submod)
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
    del _np_core, _attr, _mod_name, _submod


class MotionLoader:
    """Loads root_pos/root_rot(xyzw)/dof_pos from one of our raw_run pkls,
    resamples to output_fps, and computes velocities -- identical math to
    third_party/unitree_rl_mjlab/csv_to_npz.py.reference's MotionLoader,
    only _load_motion's source differs (pkl fields instead of a CSV)."""

    def __init__(
        self,
        motion_file: str,
        output_fps: float,
        device: torch.device | str,
    ):
        self.motion_file = motion_file
        self.output_fps = output_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        with open(self.motion_file, "rb") as f:
            data = pickle.load(f)
        self.input_fps = float(data["fps"])
        self.input_dt = 1.0 / self.input_fps
        root_pos = torch.from_numpy(np.asarray(data["root_pos"])).to(torch.float32)
        root_rot = torch.from_numpy(np.asarray(data["root_rot"])).to(torch.float32)
        dof_pos = torch.from_numpy(np.asarray(data["dof_pos"])).to(torch.float32)

        self.motion_base_poss_input = root_pos.to(self.device)
        # xyzw (this repo's convention) -> wxyz (mjlab's convention).
        self.motion_base_rots_input = root_rot[:, [3, 0, 1, 2]].to(self.device)
        self.motion_dof_poss_input = dof_pos.to(self.device)

        self.input_frames = root_pos.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, "
            f"input fps: {self.input_fps}, "
            f"output frames: {self.output_frames}, "
            f"output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return slerped

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots, self.output_dt
        )

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_sim(sim, scene, joint_names, input_file, output_fps, output_path):
    motion = MotionLoader(motion_file=input_file, output_fps=output_fps, device=sim.device)

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log: dict[str, Any] = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    scene.reset()
    print(f"\nConverting {motion.output_frames} frames from {input_file}...")

    pbar = tqdm(total=motion.output_frames, desc="Processing frames", unit="frame", ncols=100)
    file_saved = False
    while not file_saved:
        (
            (
                motion_base_pos, motion_base_rot,
                motion_base_lin_vel, motion_base_ang_vel,
                motion_dof_pos, motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())

        torch.testing.assert_close(robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0])
        torch.testing.assert_close(robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0])

        pbar.update(1)
        if reset_flag:
            file_saved = True
            pbar.close()
            print("\nStacking arrays and saving...")
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                      "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)
            np.savez(output_path, **log)
            print(f"Saved {output_path}")


def main(
    motion_file: str,
    motion_out_dir: str = "unitree_g1/mjlab_run",
    output_fps: float = 50.0,
    device: str = "cpu",
):
    """Convert a raw_run/*.pkl reference motion to mjlab's .npz format.

    Args:
        motion_file: Path relative to resources/reference_motion/, e.g.
            "unitree_g1/raw_run/g1moves_B_DadDance.pkl".
        motion_out_dir: Output subdir under resources/reference_motion/.
        output_fps: Resample rate; 50 matches mjlab's own bundled motions.
        device: "cpu" (this machine has no CUDA) or "cuda:0".
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    motion_root = os.path.join(repo_root, "resources", "reference_motion")
    input_path = os.path.join(motion_root, motion_file)
    output_dir = os.path.join(motion_root, motion_out_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.basename(motion_file).replace(".pkl", ".npz")
    output_path = os.path.join(output_dir, out_name)

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    run_sim(
        sim=sim,
        scene=scene,
        joint_names=G1_JOINT_NAMES,
        input_file=input_path,
        output_fps=output_fps,
        output_path=output_path,
    )


if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)
