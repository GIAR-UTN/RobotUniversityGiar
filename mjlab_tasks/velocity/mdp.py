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
