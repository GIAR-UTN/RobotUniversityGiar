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
