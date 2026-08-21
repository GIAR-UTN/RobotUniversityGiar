from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import G1Flat12DofCommonCfg


class G1TargetCfg(G1Flat12DofCommonCfg):
    class env(G1Flat12DofCommonCfg.env):
        # ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12)
        # + pitch_target(1) + roll_target(1) = 47; privileged adds base_lin_vel(3) = 50.
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

    class sim(G1Flat12DofCommonCfg.sim):
        substeps = 4  # matches g1_config.py -- avoids NaN contact-force blowups on early random-policy falls (CPU/Metal backend)

    class commands(G1Flat12DofCommonCfg.commands):
        heading_command = False  # this task never walks -- a heading-derived ang_vel_yaw would fight the zeroed range below
        class ranges(G1Flat12DofCommonCfg.commands.ranges):
            lin_vel_x = [0., 0.]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]
            heading = [0., 0.]

    class rewards(G1Flat12DofCommonCfg.rewards):
        soft_dof_pos_limit = 0.9
        torso_pitch_tracking_sigma = 0.1  # same magnitude as go2_wtw's euler_tracking_sigma
        torso_roll_tracking_sigma = 0.1
        # Shared "target-aware" contract (see the "G1 target" plan): the last 2 obs
        # slots are (pitch_target, roll_target), meaning "the orientation needed
        # to face the current target" -- sampled per-episode during training
        # (behavior_params_range below), overwritten each tick with a live
        # --ball bearing by rugiar_driver.py's driver loop when this flag is
        # set. Any future sibling task (different object/reward, same last-2-slots
        # meaning) can set this too and share the same driver injection path.
        target_aware = True

        class behavior_params_range:
            resampling_time = 5.0
            # Conservative first-pass ranges -- within what a standing G1 can plausibly
            # reach without falling. Widen once Phase-1 watching confirms it's stable.
            pitch_target_range = [-0.5, 0.3]
            roll_target_range = [-0.2, 0.2]

        # G1Flat12DofCommonCfg.rewards.scales == LeggedRobotCfg.rewards.scales (the
        # base defaults, mostly 0.0) -- NOT g1_config.py's G1RoughCfg walking scales,
        # since this subclasses the common mixin directly. So only what's listed below
        # is actually active; no need to separately zero out tracking_lin_vel/orientation/etc.
        class scales(G1Flat12DofCommonCfg.rewards.scales):
            tracking_torso_pitch = 0.6
            tracking_torso_roll = 0.6
            alive = 0.3
            dof_pos_limits = -5.0
            action_rate = -0.01
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            collision = -1.0

    class sensor(G1Flat12DofCommonCfg.sensor):
        add_rgb_camera = True
        class rgb_camera_config(G1Flat12DofCommonCfg.sensor.rgb_camera_config):
            # Exact D435 mount pose from the URDF's pelvis->torso_link (fixed,
            # xyz=(-0.00396,0,0.044)) composed with torso_link->d435_link (fixed,
            # xyz=(0.0576,0.0175,0.4299), pitch~=0.8308 rad down) -- see the "G1 target"
            # plan for the derivation. `forward` is (1,0,0) rotated by that pitch about Y.
            pos = (0.0537, 0.0175, 0.4739)
            forward = (0.6743, 0.0, -0.7385)


class G1TargetCfgPPO(LeggedRobotCfgPPO):
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
        experiment_name = 'g1_target'
