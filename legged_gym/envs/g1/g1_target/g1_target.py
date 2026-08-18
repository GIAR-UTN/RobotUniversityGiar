from legged_gym import *
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float


class G1Target(LeggedRobot):
    """G1 (12-DOF, legs only -- same body as the "g1" walking task), standing in
    place and learning to pitch/roll its torso to a sampled per-episode target,
    purely by articulating its legs (no stepping). Proof of concept for aiming
    the torso-mounted camera at a target without walking toward it -- see
    rugiar's plan "G1 target proof of concept" for the full context.
    """

    def _init_buffers(self):
        super()._init_buffers()
        self.pitch_target_range = self.cfg.rewards.behavior_params_range.pitch_target_range
        self.roll_target_range = self.cfg.rewards.behavior_params_range.roll_target_range
        self.pitch_target = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.roll_target = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self._resample_targets(torch.arange(self.num_envs, device=self.device))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._resample_targets(env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        env_ids = (self.episode_length_buf % int(
            self.cfg.rewards.behavior_params_range.resampling_time / self.dt
        ) == 0).nonzero(as_tuple=False).flatten()
        self._resample_targets(env_ids)

    def _resample_targets(self, env_ids):
        if len(env_ids) == 0:
            return
        self.pitch_target[env_ids, :] = torch_rand_float(
            self.pitch_target_range[0], self.pitch_target_range[1], (len(env_ids), 1), device=self.device)
        self.roll_target[env_ids, :] = torch_rand_float(
            self.roll_target_range[0], self.roll_target_range[1], (len(env_ids), 1), device=self.device)

    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.pitch_target,
            self.roll_target,
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.pitch_target,
            self.roll_target,
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands (always zero in this task)
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.  # pitch/roll target
        return noise_vec

    def _reward_alive(self):
        return 1.0

    def _reward_tracking_torso_pitch(self):
        pitch_error = torch.square(self.simulator.base_euler[:, 1] - self.pitch_target.squeeze(1))
        return torch.exp(-pitch_error / self.cfg.rewards.torso_pitch_tracking_sigma)

    def _reward_tracking_torso_roll(self):
        # Note: no hip_pos-style penalty on hip roll/yaw drift here (unlike g1.py's
        # walking task) -- reaching a nonzero roll target on this body is expected
        # to need hip_roll articulation, so penalizing that would fight the goal.
        roll_error = torch.square(self.simulator.base_euler[:, 0] - self.roll_target.squeeze(1))
        return torch.exp(-roll_error / self.cfg.rewards.torso_roll_tracking_sigma)
