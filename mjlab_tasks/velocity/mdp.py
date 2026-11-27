"""Observation terms that mjlab core doesn't ship but external G1 policies
were trained with."""
import torch


def phase(env, period: float, command_name: str) -> torch.Tensor:
    """Gait-phase clock [sin, cos] with `period` seconds, zeroed while the
    command is ~standing (|cmd| < 0.1). A verbatim port of unitree_rl_mjlab's
    src/tasks/velocity/mdp/observations.py::phase (Apache-2.0). Its G1
    velocity policy reads exactly this as observation slots 9-10."""
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    out = torch.zeros(env.num_envs, 2, device=env.device)
    out[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    out[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    standing = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    return torch.where(standing.unsqueeze(1), torch.zeros_like(out), out)


def gait_phase(env, period: float, offset: float = 0.5) -> torch.Tensor:
    """Per-leg gait-phase clock [sin_L, sin_R, cos_L, cos_R] with `period`
    seconds and a left/right `offset` (0.5 = alternating gait). No standing
    mask: the DWAQ blind-stairs policy (Rugiar-G1Edu-Velocity-DWAQ) was
    trained with the phase always running, and reads exactly these four
    values as the LAST four observation slots.

    Matches unitree_rl_lab / LeggedLab's DWAQ phase term
    (g1_dwaq_env.py::compute_current_observations: sin(phase) then
    cos(phase), each two legs) and deploy_mujoco2026's `gait_phase`
    observation (params period=0.8, offset=0.5)."""
    phase_left = (env.episode_length_buf * env.step_dt) % period / period
    phase_right = (phase_left + offset) % 1.0
    out = torch.zeros(env.num_envs, 4, device=env.device)
    out[:, 0] = torch.sin(phase_left * torch.pi * 2.0)
    out[:, 1] = torch.sin(phase_right * torch.pi * 2.0)
    out[:, 2] = torch.cos(phase_left * torch.pi * 2.0)
    out[:, 3] = torch.cos(phase_right * torch.pi * 2.0)
    return out


def joint_pos(env, asset_cfg) -> torch.Tensor:
    """Absolute joint positions (no default-pose subtraction).

    mjlab's own `joint_pos_rel` subtracts the entity's default pose, but the
    LEAPP velocity-height policy (Rugiar-G1Edu-Velocity-Height) does that
    subtraction *inside its own ONNX graph* from a fixed training default.
    Feeding it raw positions keeps the env's own reset pose free to differ
    without silently shifting the policy's observation."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(env, asset_cfg) -> torch.Tensor:
    """Absolute joint velocities (default joint velocity is zero, so this is
    numerically identical to `joint_vel_rel` today, but explicit for the same
    reason as `joint_pos` above)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def root_quat_w(env, asset_cfg) -> torch.Tensor:
    """Root orientation quaternion in world frame, wxyz (mjlab's convention).

    The LEAPP graph needs the quaternion itself (it computes projected
    gravity internally), so the actor observation carries it instead of the
    derived projected-gravity vector."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_quat_w


def track_base_height(env, command_name: str, std: float = 0.2) -> torch.Tensor:
    """Exponential reward for tracking the command term's target base height.

    The mjlab-native counterpart of agile's `track_base_height_exp_smooth`:
    same exp(-error^2 / std^2) shape, but reads the height command straight
    from our `UniformVelocityHeightCommand` and the simulated root height
    (no low-pass filtering — mjlab's command is already smooth while
    standing/walking)."""
    term = env.command_manager.get_term(command_name)
    target = term.height_command
    actual = env.scene["robot"].data.root_link_pos_w[:, 2]
    return torch.exp(-torch.square(target - actual) / std**2)
