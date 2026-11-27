"""Velocity command term with a fourth, base-height channel.

The stock `UniformVelocityCommand` samples `[vx, vy, wz]`; the LEAPP
velocity-height policy (and any policy trained on this repo's
`Rugiar-G1Edu-Velocity-Height` task) also consumes a target base height.
This subclass keeps every bit of the stock command's resampling, heading,
standing and world-frame behavior, and adds:

  * a `height_command` [N] buffer sampled from `cfg.height_range`,
  * a 4-wide `command` property `[vx, vy, wz, height]` for the actor
    observation and the control-web adapter,
  * a `track_base_height`-compatible accessor (mjlab_tasks/velocity/mdp.py),
  * a viser GUI height slider alongside the stock velocity joystick.

The stock rewards and `MjlabVelocityAdapter` only ever index the first three
channels, so a 4-wide command is backward compatible with them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)


@dataclass(kw_only=True)
class UniformVelocityHeightCommandCfg(UniformVelocityCommandCfg):
    """Stock velocity command plus a target-height channel."""

    height_range: tuple[float, float] = (0.4, 0.72)
    """(min, max) target pelvis height in meters, sampled per resample."""

    default_height: float = 0.72
    """Height held by the term between resamples and at construction."""

    def build(self, env) -> "UniformVelocityHeightCommand":
        return UniformVelocityHeightCommand(self, env)


class UniformVelocityHeightCommand(UniformVelocityCommand):
    cfg: UniformVelocityHeightCommandCfg

    def __init__(self, cfg: UniformVelocityHeightCommandCfg, env):
        super().__init__(cfg, env)
        self.height_command = torch.full(
            (self.num_envs,), cfg.default_height, device=self.device
        )
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        # Set by create_gui() when the viser viewer is active.
        self._height_enabled = None
        self._height_slider = None

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.vel_command_b, self.height_command.unsqueeze(-1)], dim=-1
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids] = r.uniform_(*self.cfg.height_range)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_height"] += (
            torch.abs(self.height_command - self.robot.data.root_link_pos_w[:, 2])
            / max_command_step
        )

    # GUI.

    def create_gui(
        self,
        name: str,
        server,
        get_env_idx,
        on_change=None,
        request_action=None,
    ) -> None:
        super().create_gui(name, server, get_env_idx, on_change, request_action)
        lo, hi = self.cfg.height_range
        with server.gui.add_folder(f"{name.capitalize()} height"):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)
            slider = server.gui.add_slider(
                "height",
                min=lo,
                max=hi,
                step=0.01,
                initial_value=self.cfg.default_height,
            )
        self._height_enabled = enabled
        self._height_slider = slider

    def compute(self, dt, env_ids: torch.Tensor | None = None) -> None:
        super().compute(dt, env_ids)
        if (
            env_ids is None
            and self._height_enabled is not None
            and self._height_enabled.value
            and self._height_slider is not None
        ):
            assert self._joystick_get_env_idx is not None
            idx = self._joystick_get_env_idx()
            self.height_command[idx] = self._height_slider.value
