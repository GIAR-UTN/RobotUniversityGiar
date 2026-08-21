"""RL runner config for Rugiar-G1-Mimic -- identical hyperparameters to
mjlab's own `unitree_g1_tracking_ppo_runner_cfg()`
(mjlab/tasks/tracking/config/g1/rl_cfg.py), only `experiment_name`
differs, so this repo's own training runs land under their own log dir
instead of mixing into `g1_tracking` (which is also where Javier's
Kaggle-trained checkpoints logged, per docs/mjlab_migration.md §0)."""
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def rugiar_g1_mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_mjlab_mimic",
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
