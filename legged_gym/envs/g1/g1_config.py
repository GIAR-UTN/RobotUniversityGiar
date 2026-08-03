from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import G1Flat12DofCommonCfg


class G1RoughCfg(G1Flat12DofCommonCfg):
    class env(G1Flat12DofCommonCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

    class sim(G1Flat12DofCommonCfg.sim):
        substeps = 4  # was 1 — raised to avoid NaN contact-force blowups on early random-policy falls (CPU/Metal backend)

    class domain_rand(G1Flat12DofCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class rewards(G1Flat12DofCommonCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales(G1Flat12DofCommonCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.3  # was 0.15 — small step up, not a leap: episode length plateaued at
                         # ~73/1000 steps across two curriculum runs (see HANDOFF_stability_
                         # curriculum.md §6 hypothesis 3) with alive contributing very little
                         # relative to the penalty terms above. Doubling it is enough to move
                         # the needle if this is the bottleneck, without drowning out the other
                         # terms — re-check the next run's `Mean episode length` before pushing
                         # this further.
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18


class G1RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'


class G1CautiousCfg(G1RoughCfg):
    """Swap-experiment variant: same 12 actions / 47 obs, same PD gains, same default pose —
    only the reward weights differ, biasing toward a slower, more energy-conserving gait
    instead of G1RoughCfg's default. Intended to be TorchScript-swappable with the g1 policy
    at inference time, since observation/action shapes match exactly."""
    class rewards(G1RoughCfg.rewards):
        class scales(G1RoughCfg.rewards.scales):
            tracking_lin_vel = 0.5       # was 1.0 — less eager to chase commanded speed
            dof_acc = -2.5e-6            # was -2.5e-7 — 10x more averse to jerky joints
            dof_vel = -1e-2              # was -1e-3 — 10x more averse to fast joints
            action_rate = -0.1           # was -0.01 — 10x more averse to changing its mind
            torques = -0.0002            # not penalized at all in G1RoughCfg — added here


class G1CautiousCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_cautious'
