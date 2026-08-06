# PROJECT KNOWLEDGE BASE

**Generated:** 2025-04-03
**Commit:** 0073932
**Branch:** dev

## OVERVIEW
RobotUniversityGiar (formerly LeggedGym-Ex) is a legged robot RL framework supporting Genesis, IsaacGym, and IsaacSim. Extends legged_gym with 10+ published methods (DeepMimic, AMP, Walk These Ways, etc.).

## STRUCTURE
```
RobotUniversityGiar/
├── legged_gym/          # Core framework (envs, scripts, utils, simulator)
├── rsl_rl/              # RL algorithms (PPO variants)
├── resources/           # Robot URDFs, meshes, reference motions
└── tests/               # Test scripts
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new robot | `legged_gym/envs/` | Extend LeggedRobot base class |
| Add RL method | `rsl_rl/` | Add PPO variant + runner + storage |
| Train policy | `legged_gym/scripts/train.py` | Entry point for training |
| Run inference | `legged_gym/scripts/play.py` | Load and run trained policy |
| Config system | `legged_gym/envs/base/legged_robot_config.py` | Nested class configs |
| Task registry | `legged_gym/utils/task_registry.py` | Factory for envs/algs |
| Terrain gen | `legged_gym/utils/terrain.py` | Heightfield/trimesh |
| Math utils | `legged_gym/utils/math_utils.py` | Quaternion ops, etc. |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| LeggedRobot | Class | `envs/base/legged_robot.py` | Base environment |
| BaseTask | Class | `envs/base/base_task.py` | Abstract task interface |
| TaskRegistry | Class | `utils/task_registry.py` | Env/alg factory |
| OnPolicyRunner | Class | `rsl_rl/runners/` | Training orchestration |
| PPO | Class | `rsl_rl/algorithms/ppo.py` | Base RL algorithm |
| Simulator | ABC | `simulator/simulator.py` | Simulator interface |

## CONVENTIONS

**Configuration Pattern**: Nested classes inheriting from `LeggedRobotCfg`/`LeggedRobotCfgPPO`. Example: `class GO2Cfg(LeggedRobotCfg)` with nested `class env`, `class rewards`, etc.

**Task Registration**: Register in `legged_gym/envs/__init__.py`: `task_registry.register("go2", GO2, GO2Cfg, GO2CfgPPO)`

**Simulator Selection**: Set `SIMULATOR` env var: `export SIMULATOR=genesis` or `isaaclab`

**Naming**: Task names follow `<robot>_<variant>` (e.g., `go2_ts`, `k1_amp`)

## ANTI-PATTERNS (THIS PROJECT)

1. **IsaacGym Reset Bug**: After `reset()`, call `simulator.forward()` once before reading rigid body states (see `g1_deepmimic.py:73`)
2. **Observation Changes**: Modifying `obs_buf` requires updating ALL `_reward_*` methods (see "[NOTE]: Must be adapted" comments)
3. **IsaacLab Tensor Device**: Domain randomization tensors must be on CPU for IsaacLab (`set_material_properties`, `set_masses`, `set_coms`)
4. **Terrain Constraints**: Cannot use `curriculum=True` with `selected=True` simultaneously
5. **Genesis XML**: Must provide XML file path when using Genesis simulator
6. **Heightfield Limitation**: Heightfield terrain not implemented for IsaacLabSimulator
7. **Genesis GPU Device Mismatch**: Genesis getters (`get_vel()`, `get_ang()`, `get_dofs_limit()`, etc.) may return CPU tensors even when the scene runs on CUDA. Additionally, Genesis's internal `sanitize_index` force-moves all indices to CPU, causing `scatter_` failures on CUDA buffers. The workaround is runtime monkey-patches in `genesis_simulator.py` plus explicit `.to(self._device)` on all getter results before PyTorch tensor operations.

## UNIQUE STYLES

- **Type Aliases**: `ObsBuf = Tensor`, `Action = Tensor`, `Reward = Tensor` in base classes
- **Config Assertions**: Extensive validation in `LeggedRobot.__init__()` to catch config errors early
- **Debug Flags**: `cfg.env.debug*` flags for visualization (height points, depth images, etc.)
- **Paper Caveats**: Comments like "code above can't result in same reward curve as paper" indicate known deviations

## COMMANDS

```bash
# Training
python -m legged_gym.scripts.train --task go2_ts --headless
python -m legged_gym.scripts.train --task go2 --num_envs 1000

# Inference
python -m legged_gym.scripts.play --task go2_ts --resume
python -m legged_gym.scripts.play --task go2 --use_joystick --joystick_type xbox

# Motion processing (DeepMimic/AMP)
python -m legged_gym.scripts.process_reference_motion --task g1_deepmimic

# Testing
python tests/test_all_tasks.py
python tests/test_all_tasks.py --tasks go2 g1 --iterations 3

# List all tasks
python tests/test_all_tasks.py --list
```

## INSTALLATION

### Prerequisites
- CPU: Intel Core i9 recommended
- GPU: RTX 3080 10GB+
- OS: Ubuntu 22.04
- Python: >=3.8
- Nvidia Driver: >=570

### IsaacGym (Python 3.8)
```bash
conda create -n lr_gym python=3.8
conda activate lr_gym
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# Download IsaacGym Preview 4, then:
git clone https://github.com/lupinjia/LeggedGym-Ex.git
cd LeggedGym-Ex && pip install -e ".[isaacgym]"
```

### Genesis (Python 3.10)
```bash
conda create -n lr_gen python=3.10
conda activate lr_gen
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/lupinjia/LeggedGym-Ex.git
cd LeggedGym-Ex && pip install -e ".[genesis]"
export SIMULATOR=genesis
```

### IsaacSim/IsaacLab (Python 3.11)
```bash
conda create -n lr_lab python=3.11
conda activate lr_lab
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# Install IsaacLab v2.3.2
git clone https://github.com/lupinjia/LeggedGym-Ex.git
cd LeggedGym-Ex && pip install -e ".[isaaclab]"
export SIMULATOR=isaaclab
```

## TASK REFERENCE

| Task | Robot | Method | Paper |
|------|-------|--------|-------|
| go2 | Go2 | Basic | - |
| go2_wtw | Go2 | Walk These Ways | [2212.03238](https://arxiv.org/abs/2212.03238) |
| go2_ts | Go2 | Teacher-Student | [2010.11251](https://arxiv.org/abs/2010.11251) |
| go2_ee | Go2 | Explicit Estimator | [2202.05481](https://arxiv.org/abs/2202.05481) |
| go2_cts | Go2 | Concurrent TS | [CTS](https://clearlab-sustech.github.io/concurrentTS/) |
| go2_dreamwaq | Go2 | DreamWaQ | [2301.10602](https://arxiv.org/abs/2301.10602) |
| go2_cat | Go2 | Constraints as Terms | [CaT](https://constraints-as-terminations.github.io/) |
| go2_nav | Go2 | Navigation | - |
| g1 | G1 | Basic | - |
| g1_deepmimic | G1 | DeepMimic | [1804.02717](https://arxiv.org/abs/1804.02717) |
| k1 | K1 | Basic | - |
| k1_amp | K1 | AMP | [2104.02180](https://arxiv.org/abs/2104.02180) |
| tron1pf | TRON1 | Basic | - |
| tron1pf_ee | TRON1 | Explicit Estimator | - |

## KEY CONFIGURATION PARAMETERS

### Environment
- `num_envs`: Parallel environments (default: 4096)
- `num_observations`: Observation dimension
- `num_actions`: Action dimension (default: 12)
- `episode_length_s`: Episode length (default: 20s)

### Terrain
- `mesh_type`: 'plane', 'heightfield', 'trimesh'
- `curriculum`: Enable terrain curriculum
- `measure_heights`: Include height measurements

### Control
- `control_type`: 'P' (position), 'V' (velocity), 'T' (torque)
- `stiffness`/`damping`: PD gains
- `dt`: Control frequency (default: 0.02 = 50Hz)
- `decimation`: Sim steps per control (default: 4)

### Rewards
- `tracking_sigma`: Tracking reward parameter
- `scales`: Dict of reward weights

### Domain Randomization
- `randomize_friction`: [0.5, 1.25]
- `randomize_base_mass`: [-1, 1] kg added
- `push_robots`: Random velocity perturbations

### PPO Training
- `num_steps_per_env`: Rollout length (default: 24)
- `max_iterations`: Training iterations (default: 1500)
- `learning_rate`: Default 1e-3
- `clip_param`: PPO clip (default: 0.2)

Full reference: https://leggedgym-ex-doc.readthedocs.io/en/latest/developer_guide/parameter_reference/legged_robot_config.html

## DEPLOYMENT

### Sim2Sim Testing
Install [go2_deploy](https://github.com/lupinjia/go2_deploy) for MuJoCo sim2sim:
```bash
./go2_deploy simple_rl
```

### Real Robot Deployment
1. Remove `base_lin_vel` from observations (unavailable on real robot)
2. Train: `python -m legged_gym.scripts.train --task=go2 --headless`
3. Export: `python -m legged_gym.scripts.play --task=go2`
4. Find JIT model in `logs/experiment_name/exported/`
5. Deploy using [go2_deploy](https://github.com/lupinjia/go2_deploy)

## NOTES

- Multi-simulator support: Same code runs on Genesis/IsaacGym/IsaacLab via `SIMULATOR` env var
- 24+ registered tasks across 5 robot types (GO2, G1, K1, TRON1PF, TRON1SF)
- 8 PPO algorithm variants (TS, EE, CTS, AMP, DreamWaQ, etc.)
- Reference: External docs at https://genesis-lr-doc.readthedocs.io/en/latest/

## FORK ADDITIONS: unified control web (`legged_gym/control/`, `web/`)

Networked control layer this fork adds on top of upstream legged_gym — lets a browser (or
eventually an LLM) drive policy switching, training, and telemetry over WebSocket, one call
surface shared identically by sim and real hardware. See `README.md` §4/§5 for the narrative
version; this section is the quick map + the rules that keep it from rotting.

| Component | Location | Role |
|---|---|---|
| `RobotAdapter` (Protocol) | `control/adapter.py` | Sim/real-symmetric state+action interface. `SimAdapter` — working, tested. `RealAdapter` — ported but untested, no hardware available. |
| `Policy` | `control/policy.py` | Wraps a TorchScript/ONNX checkpoint; `load_policy()` auto-detects export convention (this fork's LSTM `(action,h,c)` vs. unitree_rl_gym's stateless) and format (`.pt`/`.onnx`). |
| `PolicySupervisor` | `control/supervisor.py` | Owns every loaded policy, the active/pending switch, and the cross-fade ramp between them. `rename_policy()`/`add_policy()`/`remove_policy()` are the mutation surface. |
| `SafetyGovernor` | `control/safety.py` | The ONLY component that decides "is it safe to switch right now." Forces the `damping` fallback on a fall/NaN/estop and keeps forcing it until `safety.reset()`. |
| `Selector` (Protocol) | `control/selector.py` | Autonomous switch proposals. Only impl (`TiltRecoverySelector`) has no hysteresis — do not wire it into a live demo without adding one; it will fight a human's manual switch. |
| `TrainingManager` | `control/training.py` | Launches/polls web-triggered training jobs as subprocesses; owns each `policies/<name>/` folder (checkpoint + raw + meta.json) and the clone-from catalog. |
| `ControlService` | `control/service.py` | **The single call surface** — `request_switch`/`status`/`pause`/`resume`/`estop`/`rename_policy`/`delete_policy`/`start_training`/... Human (viser/web buttons) and autonomous (`Selector`) callers use the exact same methods. |
| `ControlServer` | `control/transport.py` | FastAPI+uvicorn WebSocket wrapper around `ControlService` — JSON-RPC-shaped messages on `/ws`, ~10Hz status broadcast, also serves `web/` and `docs/` as static files. |
| Web UI | `web/index.html`, `web/app.js`, `web/keymap.json` | Plain HTML/JS/CSS, no build step — talks to `ControlService` only through `/ws`. |

**Rules — violating these has already caused real bugs once; don't reintroduce them:**

1. **Never bypass `SafetyGovernor`.** Only it confirms a policy switch; only `safety.reset()`
   clears a trip. No "force switch" shortcut anywhere upstream of it.
2. **Nothing outside `control/` touches `RobotAdapter`/`PolicySupervisor` directly** — always go
   through `ControlService`. This symmetry is what keeps the web UI, viser, and any future
   real-robot/LLM caller interchangeable.
3. **`ControlService` is only ever called from the sim loop's own thread.** `ControlServer`'s
   socket handlers enqueue `(websocket, msg)` on a `queue.Queue`; `drain_commands()` executes
   each one on the sim thread, once per tick. `estop` always runs first within a batch and is
   never blocked by other queued commands.
4. **New registered task vs. UI override — pick deliberately.** Register a new
   `task_registry` task only for a STRUCTURAL change: a new reward TERM/function, a new
   termination condition, an obs/action-space change, or a different robot asset. A pure
   reward-*weight* variant belongs in the Create Policy panel as a clone-from + reward-scale
   override on an existing task, not a new `G1...Cfg` class + `task_registry.register(...)`
   entry — `g1_cautious` was retired for being exactly that redundancy.
5. **No build-step frontend.** `web/`, like `docs/index.html`, stays plain HTML/JS/CSS — this
   is course material meant to be cloned and read, not a bundled SPA.
6. **Don't "finish" `RealAdapter` without hardware.** Its `NotImplementedError` sections carry
   exact porting instructions for when a real G1 + `unitree_sdk2py` exist — speculative
   completion would just be untested code wearing a tested costume.
7. **A policy's identity lives in more than one place.** `PolicySupervisor`'s dict key, its
   `policies/<name>/` folder + `TrainingManager`'s catalog entry, and the web UI's drag-order in
   `localStorage` all encode the same name. Always rename through
   `ControlService.rename_policy()` (composes the first two) — anything else desyncs them.

**Testing:** `tests/test_control_transport.py` is the transport smoke test (fake
`ControlService`, real WebSocket round-trip). `tests/test_delete_policy.py` and
`tests/test_rename_policy.py` unit-test the supervisor/training-manager/service layers directly
— no live server needed. Extend these when touching `control/`; a manual browser click-through
is not a substitute.
