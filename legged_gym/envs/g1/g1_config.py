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
            alive = 0.15
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


class G1CrouchCfg(G1RoughCfg):
    """Fall-mitigation crouch: as low as it can *sustain* while staying stable, discovered
    by training rather than picked by hand, WHILE still responding to velocity commands
    like G1RoughCfg (meant to be switched to as a protective reflex, not a frozen statue).
    Same 12 actions / 47 obs / PD gains / default pose as G1RoughCfg (still TorchScript-
    swappable at inference).

    v3 — replaces v1 (0.6 fixed target, zero-pinned commands) and v2 (0.4 fixed target,
    commands+pushes on, which trained from scratch and never stabilized — see git history/
    README §5 for that failure). Both v1 and v2 tried to guess a good height number by hand;
    v3 doesn't:
      - `crouch_depth` (see `_reward_crouch_depth` in legged_robot.py) replaces
        `base_height` for this task: an OPEN-ENDED reward (-base_height, no target) instead
        of a squared-distance-to-a-guessed-number one. Alone it would collapse the robot to
        the ground; paired with termination/orientation/dof_pos_limits/alive, the emergent
        equilibrium is "as low as this policy can actually hold," which is exactly what we
        want and don't have to guess. `base_height`'s own scale is zeroed — the two rewards
        encode opposite goals for the same DOF and shouldn't both be active.
      - commands + domain_rand: unchanged from v2 (full G1RoughCfg command range, pushes
        on) — v2's failure was the aggressive unreachable target fighting these, not these
        themselves being too hard.
      - lin_vel_z scale -2.0 -> -1.0: a middle ground between G1RoughCfg's full anti-bounce
        penalty and v2's very permissive -0.3 — some allowance for descending without
        removing the penalty that (probably) helped stability.
    """
    class rewards(G1RoughCfg.rewards):
        crouch_depth_reference = 0.78  # stable's standing height — numerical zero-point only, not a target

        class scales(G1RoughCfg.rewards.scales):
            lin_vel_z = -1.0        # was -2.0 in G1RoughCfg / -0.3 in v2 — see class docstring
            base_height = 0.0       # was -10.0 — replaced by the open-ended crouch_depth below
            crouch_depth = 3.0      # default weight — train_package.py overrides this per-package (0 for
                                    # a stability-only package, positive otherwise) rather than this file
                                    # defining separate config classes per package; see its own docstring.


class G1CrouchCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 2000  # v1 was 1000 — v2's combined objective (deep crouch + commands) is harder
        run_name = ''
        experiment_name = 'g1_crouch'
