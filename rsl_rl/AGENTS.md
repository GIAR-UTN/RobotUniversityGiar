# RSL-RL MODULE

**Generated:** 2025-04-03

## OVERVIEW

RL algorithms module with 8 PPO variants for legged robot locomotion.
Specialized implementations for sim-to-real transfer, terrain awareness, and motion priors.

## STRUCTURE

```
rsl_rl/
├── algorithms/     # PPO implementations (8 variants)
├── modules/        # Actor-critic networks per algorithm
├── runners/        # Training orchestration + registry
├── storage/        # Rollout buffers per algorithm
├── env/            # VecEnv abstract interface
└── utils/          # Runner registry, symmetry utils
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Base PPO | `algorithms/ppo.py` | Standard proximal policy optimization |
| Base runner | `runners/on_policy_runner.py` | Generic training loop |
| Actor-critic | `modules/actor_critic.py` | Base policy network |
| Base storage | `storage/rollout_storage.py` | Standard rollout buffer |
| Add new method | Create alg + module + runner + storage | Follow existing pattern |
| Runner selection | `utils/runner_registry.py` | Dynamic runner instantiation |

## ALGORITHM VARIANTS

| Variant | File | Key Feature |
|---------|------|-------------|
| PPO | `algorithms/ppo.py` | Base algorithm |
| PPO_TS | `algorithms/ppo_ts.py` | Teacher-Student for sim-to-real |
| PPO_EE | `algorithms/ppo_ee.py` | Explicit state estimator |
| PPO_CTS | `algorithms/ppo_cts.py` | Concurrent Teacher-Student |
| PPO_DreamWaQ | `algorithms/ppo_dreamwaq.py` | Implicit terrain imagination |
| PPO_TSDepth | `algorithms/ppo_ts_depth.py` | TS with depth encoder |
| PPO_AMP | `algorithms/ppo_amp.py` | Adversarial Motion Priors |
| PPO_CTS_AMP | `algorithms/ppo_cts_amp.py` | Combined CTS + AMP |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| PPO | Class | `algorithms/ppo.py` | Base RL algorithm |
| BaseAlgorithm | Class | `algorithms/base_algorithm.py` | Algorithm interface |
| ActorCritic | Class | `modules/actor_critic.py` | Policy network |
| OnPolicyRunner | Class | `runners/on_policy_runner.py` | Training orchestration |
| RolloutStorage | Class | `storage/rollout_storage.py` | Experience buffer |
| VecEnv | ABC | `env/vec_env.py` | Environment interface |
| runner_registry | Instance | `utils/runner_registry.py` | Runner factory |

## CONVENTIONS

**Algorithm Pattern**: Extend `BaseAlgorithm`, implement `init_storage()`, `update()`, `act()`.

**Runner Registration**: Add to `runners/__init__.py`:
```python
from .my_runner import MyRunner
runner_registry.register("MyRunner", MyRunner)
```

**Component Matching**: Algorithm + Module + Storage + Runner must be compatible.
Example: `PPO_TS` uses `ActorCriticTS`, `RolloutStorageTS`, `TSRunner`.

**Config Selection**: Runner class specified in env config's `runner_class_name` field.

## TRAINING FLOW

1. **Initialization**: Runner creates actor_critic, algorithm, storage
2. **Rollout Collection**: `learn()` calls `env.step()` for N steps, stores transitions
3. **Update**: `algorithm.update()` computes PPO loss, runs backprop
4. **Logging**: TensorBoard + optional WandB, model checkpointing

## ANTI-PATTERNS

1. **Storage Mismatch**: Using base `RolloutStorage` with `PPO_TS` (needs teacher/student buffers)
2. **Missing Registration**: New runner not added to `runners/__init__.py`
3. **Device Mismatch**: Tensors must all be on same device as runner

## ALGORITHM CONFIGURATION

### PPO Parameters (cfg.algorithm)
```python
value_loss_coef = 1.0          # Value function loss weight
use_clipped_value_loss = True  # Clip value updates
clip_param = 0.2               # PPO clipping epsilon
entropy_coef = 0.01            # Entropy bonus weight
num_learning_epochs = 5        # Gradient updates per iteration
num_mini_batches = 4           # Mini-batches per update
learning_rate = 1.e-3          # Adam learning rate
schedule = 'adaptive'          # 'adaptive' or 'fixed'
gamma = 0.99                   # Discount factor
lam = 0.95                     # GAE lambda
desired_kl = 0.01              # Target KL divergence
max_grad_norm = 1.0            # Gradient clipping
use_spo = False                # Simple Policy Optimization
```

### Policy Network (cfg.policy)
```python
init_noise_std = 1.0
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
activation = 'elu'  # 'elu', 'relu', 'selu', 'tanh'

# RNN (optional)
rnn_type = 'lstm'
rnn_hidden_size = 512
rnn_num_layers = 1
```

**What this looks like as a network:** the actor is a dense MLP — `Observations → 512 → 256 → 128 → Actions`, ELU between hidden layers. Every connection between two neurons is one trained weight; the diagram below draws each weight as a small color chip on its connection (color = sign, size/opacity = magnitude) instead of a plain line, so a layer's weight distribution reads at a glance. Dense layers are sampled (a handful of neurons per layer, dotted gap standing in for the rest) since drawing all 512/256 edges would be illegible; the action layer is shown in full since it's small (12 leg DOF on G1) and each output maps to a real joint. The weights pictured are synthetic/randomly seeded — this illustrates the *architecture*, not a specific checkpoint.

<img src="../docs/assets/policy_network.svg" alt="Policy network diagram: observations flowing through a 512-256-128 MLP to leg-joint actions, with weight chips colored by sign and sized by magnitude" width="100%">

A live version of this — reading weights from the currently loaded `train_checkpoint.pt` instead of synthesizing them — is a natural addition to the Web UI's Policy Info Dock (`web/app.js`, `legged_gym/control/policy.py`; see `legged_gym/control/ARCHITECTURE.md` §4).

### Runner Parameters (cfg.runner)
```python
num_steps_per_env = 24    # Rollout length
max_iterations = 1500     # Total training iterations
save_interval = 50        # Checkpoint frequency
experiment_name = 'test'  # Log directory name
resume = False            # Resume from checkpoint
load_run = -1             # Run ID to load (-1 = latest)
checkpoint = -1           # Checkpoint ID (-1 = latest)
```

### Method-Specific Configurations

**Teacher-Student (PPO_TS)**:
- `num_encoder_epochs`: History encoder training epochs
- `history_encoder_type`: Encoder architecture ('MLP' or 'TCN')
- `history_encoder_hidden_dims`: History network dimensions
- `privilege_encoder_hidden_dims`: Privilege network dimensions

**Explicit Estimator (PPO_EE)**:
- `estimator_target`: Target variables for estimation
- `estimator_hidden_dims`: Estimator network dimensions

**DreamWaQ (PPO_DreamWaQ)**:
- `depth_encoder_dims`: Depth encoder network dimensions
- `use_terrain_imagination`: Enable terrain imagination module
