# 🦿 RobotUniversityGiar — GIAR fork

> A working, from-scratch build log of teaching a Unitree G1 humanoid to walk in simulation on a laptop with no NVIDIA GPU, and of designing a control architecture that lets you switch between trained walking policies — from a web page, autonomously on the robot, or eventually from an LLM — without rewriting anything when you move from simulator to real hardware.

This README is written as course material, not a changelog. If you're a student (or just curious) starting from **zero** robotics/RL knowledge, read it top to bottom — every acronym is defined the first time it's used, and the [full didactic write-up](docs/index.html) goes even deeper on the fundamentals (motors, PD control, reinforcement learning, the whole training pipeline) with an interactive demo. This README focuses on **what this specific fork adds** on top of that: a real control architecture for switching policies, built incrementally and reviewed as it went.

---

## 0. Where this fork sits in the family tree

```
legged_gym (ETH Zürich, Robotic Systems Lab)
   │  the original RL training environment for legged robots
   ▼
unitree_rl_gym (Unitree Robotics)
   │  adapts legged_gym for Unitree's own robots (Go2, G1, H1, H1-2)
   │  trains in NVIDIA Isaac Gym, deploys to MuJoCo (sim2sim) and real hardware (sim2real)
   ▼
LeggedGym-Ex (lupinjia)
   │  ports the same training framework to run on Genesis (a physics engine that,
   │  unlike Isaac Gym, doesn't require Linux + an NVIDIA GPU — it runs on Apple Silicon)
   ▼
RobotUniversityGiar — this fork (josetabuyo/GIAR)
      adds legged_gym/control/: a backend-agnostic layer for switching between
      trained policies, supervised from a web UI or decided autonomously,
      designed to work identically in sim and (eventually) on the real robot
```

Every step above is a real, separate open-source project — see [UPSTREAM_README.md](UPSTREAM_README.md) for LeggedGym-Ex's own feature list, supported robots, and full acknowledgements. This file only covers what changed in this fork.

**Why this fork exists:** the goal driving all of this is to eventually let a person (or an LLM, see [§6](#6-roadmap-llm-interfacing)) tell the robot *what to do* in high-level terms — "walk carefully," "stand still," "switch to the trot gait" — and have the right trained policy take over, smoothly, whether the robot in question is a Genesis simulation on a laptop or an actual G1 standing in a lab. Getting there means solving the boring-but-critical plumbing first: how do you even switch which policy is driving the robot, safely, in a way that doesn't need to be re-invented for sim vs. real? That plumbing is what §3–§5 below are about.

---

## 1. The 90-second version, if you already know RL/robotics

- Trained a Unitree G1 (humanoid) walking policy from scratch in Genesis on an M1 Pro Mac (no CUDA), using this fork's own `g1` task — 1800 PPO iterations total.
- Also confirmed unitree_rl_gym's own **shipped pretrained G1 checkpoint** (`deploy/pre_train/g1/motion.pt`) is drop-in compatible with this fork's Genesis env (same URDF, same joint order, same PD gains) — it's dramatically more stable (~0.77-0.78m base height held for hundreds of steps) than anything trainable in a few minutes locally, and is used as the "stable" reference policy.
- Fine-tuned a second policy ("cautious") from that trained checkpoint under a reward that penalizes torque/joint-velocity much more heavily — a genuine derivative of the first policy, not an independent training run.
- Trained a fourth, genuinely new skill from scratch ("crouch"): a static squat instead of a walk — zero velocity commands, no gait to learn, so it converged in 1000 PPO iterations (~15 min on this Mac's CPU) vs. the 1800 iterations the from-scratch walk needed, and survived the demo's default random pushes without tripping the safety governor.
- Built `legged_gym/control/`: a small, backend-agnostic package (`RobotAdapter` / `PolicySupervisor` / `SafetyGovernor` / `Selector` / `ControlService`) that lets you load N policies, switch between them live with a smooth cross-fade instead of a hard cut, gate switches through a safety check, and drive all of it from either a human clicking a button or an autonomous rule/network — same call, same code path.
- Wired that up to a live demo: a `viser` (web-based 3D viewer) page with Restart / Pause / per-policy switch buttons and a live "active policy" label, running against Genesis.
- Left `deploy_real/real_adapter.py` as a carefully-ported but **explicitly untested** real-hardware adapter — this repo was built with no unitree_sdk2py installed and no physical robot attached, so real-hardware verification is the natural next step for whoever picks this up on actual hardware.

---

## 2. Setup

```bash
git clone https://github.com/josetabuyo/RobotUniversityGiar.git
cd RobotUniversityGiar
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision matplotlib tensorboard xlsxwriter pandas tqdm scipy pygame trimesh rich-argparse viser
pip install genesis-world warp-lang
pip install -e .
export SIMULATOR=genesis   # required — legged_gym refuses to import without this set
```

No GPU required. On Apple Silicon, Genesis will report `Running on [Apple M1/M2/...] with backend gs.metal` — if it silently falls back to CPU, training still works, just slower (this fork's own G1 training ran entirely on CPU; Genesis's Metal path was, at time of writing, inconsistent enough on macOS that we didn't depend on it).

### Docker Compose (recommended for reproducible runs)

```bash
# 1. Put your .pt checkpoints in ./policies/ (e.g. ./policies/motion.pt)
# 2. Copy .env.sample to .env and edit if needed
# 3. Build and run (works on any host arch — amd64 or arm64/Apple Silicon)
docker compose up --build
# then open http://localhost:9006  (viser viewer)
# and   http://localhost:9013  (unified control web)
```

The image builds and runs on any host architecture: on `linux/amd64` it installs the CUDA 12.8 (sm_120) build of PyTorch; everywhere else (e.g. `linux/arm64` under Colima/Docker Desktop on Apple Silicon) it keeps the generic CPU build, since NVIDIA doesn't publish CUDA wheels for non-amd64.

On a Linux host with an NVIDIA GPU and the `nvidia-container-runtime` installed, pass through the GPU with the `docker-compose.gpu.yml` overlay (not in the base file — Compose hard-fails container creation on hosts without a matching driver if the device reservation is unconditional):

```bash
GENESIS_BACKEND=cuda docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Key environment variables (set in `.env`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `GENESIS_BACKEND` | `cpu` | `cuda` for GPU (falls back to CPU if unavailable), `cpu` to force CPU-only |
| `ACTIVE_POLICY` | *(first alphabetically)* | Filename (without `.pt`) in `./policies/` to start active |
| `HEADLESS` | `0` | Set `1` for a smoke test without the browser viewer |
| `SPEED` | `0.35` | Playback speed multiplier (`1.0` = real-time 50 Hz) |
| `CONTROL_PORT` | `9013` | Port for the unified control WebSocket server |
| `VISER_PORT` | `9006` | Port for the viser 3D viewer |

The compose file mounts `./policies:/workspace/policies:ro` so checkpoint files are available inside the container without copying them into the image.

### Train a policy

```bash
python legged_gym/scripts/train.py --task=g1 --headless --cpu --num_envs=64 --max_iterations=1800
python legged_gym/scripts/play.py --task=g1 --headless --cpu --num_envs=1 --load_run=<run_name> --export_onnx
# play.py exports logs/g1/<run_name>/exported/policy_lstm_1.pt  (TorchScript) and
#                  logs/g1/<run_name>/exported/policy_lstm_1.onnx (ONNX, --export_onnx only)
# both are loadable directly by load_policy() / rugiar_driver.py — see §5a for sharing across a team.
```

**Before deleting any `logs/<task>/<run>/` directory**, copy its final raw checkpoint (`model_<N>.pt` — the one with optimizer + critic state, NOT the exported inference-only `policy_lstm_1.pt`) into `./checkpoints/<task>/`, git-tracked, if there's any chance you'll want to keep training that run later:

```bash
mkdir -p checkpoints/<task>
cp logs/<task>/<run_name>/model_<N>.pt checkpoints/<task>/model_<N>.pt
```

`logs/` is gitignored (training scratch — hundreds of intermediate checkpoints per run) and gets cleaned up; `checkpoints/` is not — it's the durable, resumable copy. Resume/fine-tune from it with `finetune_from_checkpoint.py`:

```bash
python legged_gym/scripts/finetune_from_checkpoint.py --task <task> \
    --from_checkpoint checkpoints/<task>/model_<N>.pt \
    --max_iterations <more> --headless --cpu --num_envs=64
```

### Fusing policies (model merging)

Combine 2+ already-trained local policies' weights into a new policy — no
further training involved. Available from the control web's Create Policy
panel ("⚛ Fuse policies…", right under "+ New policy…") or the `rugiar` CLI:

```bash
rugiar fuse --policies stable_home_made_3 stable_home_made_4 --name blended
# weighted (3:1:1 ratio), name it explicitly
rugiar fuse --policies base_a base_b base_c --weights 3 1 1 --name blended_v2
# permutation-aligned merge instead of naive averaging (works for LSTM/GRU too)
rugiar fuse --policies base_a base_b --method git_rebasin --name blended_rebasin
rugiar fuse --list_fusion_methods   # see every method this build knows about
```

The result is registered as a normal `./policies/<name>/` — fine-tunable via
`train --from_policy` and fusable again, same as anything trained through
this UI. Each source needs a `train_checkpoint.pt` (same requirement
Clone-from has), and all sources must be architecturally compatible (same
obs/action dims, hidden dims, and recurrent-or-not) — a mismatched *task*
label across sources is only a warning, not a hard stop, since two tasks can
share an identical network shape.

**Method today: weighted average** (a.k.a. model soup / SWA-style
interpolation) — an elementwise weighted sum of matching weights. It's cheap
and works reasonably well for closely related checkpoints (a fine-tune
lineage, or same-seed variants), but has no guarantee for independently-
trained policies: two networks trained from different random inits can
converge to functionally-equivalent but internally *permuted*
representations, and naively averaging permuted weights usually lands
between the two minima rather than a good spot near either.

**Method: Git Re-Basin (`--method git_rebasin`).** The fix for that —
solving for the hidden-unit permutation that best aligns every non-
reference source to the first one *before* averaging (Ainsworth et al.,
2022) — so the merge lands inside, rather than between, the sources' loss
basins. Works for both plain and recurrent (LSTM/GRU) actor/critic
policies — an RNN's own per-gate hidden-unit permutation symmetry is
aligned too, and chained into the downstream MLP's own alignment (see
`legged_gym/control/fusion.py`'s `rebasin_align()`).

### Run the policy-switching demo

```bash
python legged_gym/scripts/rugiar_driver.py \
    --policy stable:./policies/stable.pt \
    --policy cautious:./policies/cautious.pt \
    --policy scratch_wobbly:./policies/scratch_wobbly.pt \
    --policy undertrained_dummy:./policies/undertrained_dummy.pt \
    --policy crouch:./policies/crouch.pt \
    --active stable
# then open http://localhost:9006
```

`--headless` runs a short scripted smoke test instead (no browser needed) — useful for CI or a quick sanity check that everything still imports and steps correctly after a change.

Add `--control_port <PORT>` to also start the unified control web (see §4a below):

```bash
python legged_gym/scripts/rugiar_driver.py \
    --policy stable:./policies/stable.pt \
    --policy cautious:./policies/cautious.pt \
    --policy scratch_wobbly:./policies/scratch_wobbly.pt \
    --policy undertrained_dummy:./policies/undertrained_dummy.pt \
    --policy crouch:./policies/crouch.pt \
    --active stable --control_port 9013
# then open http://localhost:9013
```

---

## 3. The problem this fork's architecture solves

Say you have two trained policies for the same robot — say, a normal walk and a more cautious/careful one. The mechanically simple version of "switching" is: reassign which neural network gets called each control tick. That's *almost* the whole story, but three things go wrong if you stop there:

1. **The old policy's memory doesn't belong to the new policy.** These are recurrent (LSTM) networks — they carry a hidden state between ticks. Handing the new policy the old one's hidden state is like waking someone up mid-dream and expecting their memories to make sense; it must be reset.
2. **A sudden change in target joint angle is a sudden spike in torque**, because of how PD control works (`torque = Kp × (target − current) − Kd × velocity` — see the [full explainer](docs/index.html) for this from first principles). In simulation that just looks like a stumble. On a real robot, a hard cut is the kind of thing that can genuinely damage hardware or hurt someone standing nearby.
3. **"Where does the decision to switch come from" is a different question from "is this actually a safe moment to switch."** A human clicking a web button, an autonomous rule watching sensor data, and eventually an LLM reasoning about a task, all need the *same* answer to "can I switch right now" — you don't want three different, possibly inconsistent, implementations of that safety check scattered across three different callers.

`legged_gym/control/` is the answer to all three, and it's designed so the exact same code runs whether "the robot" is a Genesis simulation or a real G1.

---

## 4. Architecture

```
   Human (viser web UI)          Autonomous Selector           (future) LLM tool call
            │                            │                              │
            └──────────────┬─────────────┴──────────────────────────────┘
                            │  ALL call the same method:
                            │  ControlService.request_switch("cautious")
                            ▼
                  ┌───────────────────┐
                  │  ControlService    │   the one call surface — status(), pause(),
                  │                    │   resume(), request_switch(name), tick(obs)
                  └─────────┬──────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
   ┌───────────────┐ ┌──────────────┐ ┌───────────┐
   │PolicySupervisor│ │SafetyGovernor│ │  Selector │  (optional — autonomous mode only)
   │                │ │              │ │           │
   │ owns loaded    │ │ THE ONLY     │ │ proposes  │
   │ policies, does │◄┤ component    │ │ switches  │
   │ the cross-fade │ │ allowed to   │ │ (rule-    │
   │ (smooth swap,  │ │ say "yes,    │ │ based     │
   │ not a hard cut)│ │ switch now"  │ │ today,    │
   └───────┬────────┘ └──────────────┘ │ learned   │
           │                            │ later)    │
           ▼                            └───────────┘
   ┌────────────────┐
   │  RobotAdapter   │   the ONLY boundary that knows about physics/hardware
   └───┬────────┬────┘
       ▼        ▼
 ┌───────────┐ ┌──────────────┐
 │ SimAdapter │ │ RealAdapter  │  ← same interface, different backend
 │ (Genesis)  │ │ (DDS / G1)   │     (RealAdapter: see §5, untested)
 └───────────┘ └──────────────┘
```

**Why it's split this way, module by module:**

- **`adapter.py`** — `RobotAdapter` is a Python `Protocol` (structural interface) with `get_state()` / `send_action()` / `reset()`. `SimAdapter` implements it over a Genesis (or MuJoCo) `legged_gym` env. Nothing above this layer ever imports Genesis or a physics engine directly — it only ever sees a `RobotState` (joint positions/velocities, orientation, "am I upright" gravity signal) that looks the same regardless of source.
- **`policy.py`** — wraps a loaded network + its own hidden state behind a `.step(obs)` / `.reset()` interface, auto-detecting which of two real jit export conventions a `.pt` file uses (explicit hidden-state args, vs. Unitree's own convention of hidden state as an internal buffer — discovered the hard way while building this).
- **`supervisor.py`** — `PolicySupervisor` owns the loaded policies and does the actual swap: `request_switch(name)` just *records intent*; `confirm_pending_switch()` — called only by the safety governor — begins a linear cross-fade of the output action over N ticks, so the PD controller sees a gradually-moving target instead of a jump.
- **`safety.py`** — `SafetyGovernor` is the single place that decides "is this a safe instant to act on that pending switch," using the robot's own upright/fallen signal (`projected_gravity`, the same one `legged_robot.py` already uses to end a training episode when a robot falls over). It can also unilaterally hand control to a `damping` fallback skill (holds the default pose, no learned behavior) if something looks wrong — independent of who asked for what.
- **`selector.py`** — `Selector.propose(state) -> Optional[name]` is the pluggable seam for autonomous behavior. Today it's a simple threshold rule (`TiltRecoverySelector`). The 2025-2026 research direction for this specifically on Unitree G1 (see RPG, arXiv:2604.21355; SkillBlender, arXiv:2506.09366) is a small learned gating network doing continuous blending instead of discrete rule-based switching — that's a drop-in replacement behind the same one-method interface, not a redesign.
- **`service.py`** — `ControlService` is the call surface. Today, `viser`'s button callbacks call it in-process (`legged_gym/scripts/rugiar_driver.py`). The identical class, wrapped in a thin WebSocket/JSON-RPC layer, is what would let an external process (a real robot with no display attached, or a remote web app) drive the same thing later — the transport is a detail; the interface (`switch` / `status` / `pause` / `estop`) doesn't change.

### 4a. The unified control web

`legged_gym/control/transport.py` (`ControlServer`) wraps `ControlService` in a JSON-over-WebSocket transport (FastAPI + uvicorn), exposing the exact same five methods — `request_switch` / `status` / `pause` / `resume` / `estop` — to any external client at `ws://<host>:<port>/ws`. Started via `--control_port` on `rugiar_driver.py` (see above); a plain Python `websockets` client or `websocat` can drive it with no browser at all.

Unless `--headless` is also set, the same port also serves `web/index.html`: a single, build-step-free HTML/JS/CSS page (same philosophy as `docs/index.html` — no npm, no bundler, read it and run it) with three regions:

- **A tabbed view area** — Docs (this repo's didactic write-up, iframed), Simulator (the `viser` 3D viewer, iframed), and a Real-robot tab that's present but disabled until `ControlService.status()["backend"]` reports `"real"` instead of `"sim"` — the same panel and controls are meant to keep working once real hardware exists (see §5), only the view and backend change.
- **A persistent controls panel**, visible regardless of which tab is active: the 🟢/🟡/🔴 active-policy indicator (mirroring `viser`'s own label), one button per loaded policy, Pause/Resume, Restart, and a large E-STOP button — all driven purely by `status()` pushes over the same WebSocket, ~10 times a second.
- **Keyboard shortcuts**, defined in `web/keymap.json` (edit the file to change bindings — there's no in-page rebind UI in v1) and dispatched through the identical WebSocket send path the buttons use. They only fire while the controls panel — not the `viser` iframe — has DOM focus, because a cross-origin iframe cannot forward `keydown` events to the host page; the panel shows a visible hint and re-arms on click when that happens. The mouse E-STOP button is unaffected by this and always works, since it's a click rather than a keystroke — treat it as the primary stop mechanism, keyboard `Esc` as a convenience on top of it.

### Why not just adopt ROS 2 / `ros2_control`?

This is a legitimate question — `ros2_control`'s `controller_manager` solves almost exactly this problem (multiple named "controllers," switchable at runtime, backend-abstracted between sim and real), and there's real prior art doing exactly this for legged robots: [`legubiao/quadruped_ros2_control`](https://github.com/legubiao/quadruped_ros2_control) runs multiple controller types across MuJoCo, Gazebo, and a real Unitree Go2. It's worth reading if you want the "grown-up," ROS-ecosystem version of this idea.

For *this* fork, adopting full ROS 2 today would mean bolting a colcon workspace, DDS configuration, and a C++-adjacent toolchain onto a pure Python/PyTorch/Genesis project built for quick local iteration on a Mac — a lot of cost for a solo/small-team open-source project, for marginal benefit right now. So `legged_gym/control/` borrows the *pattern* (named, swappable, lifecycle-staged controllers behind an abstract hardware interface) without the dependency, and names its lifecycle states (`INACTIVE`/`READY`/`ACTIVE`/`FAULT`) to match `ros2_control`'s own vocabulary — on purpose, so a real `ros2_control` bridge later wouldn't require renaming anything.

---

## 5. Current status & known limitations

- **`SimAdapter`**: working, tested, is what the `viser` demo runs on.
- **`RealAdapter`** (`deploy_real/real_adapter.py`): ported carefully against unitree_rl_gym's own `deploy_real.py` (observation building, action → target-joint-position math, motor index mapping) — but the physical button-gated state machine (`zero_torque_state` → `move_to_default_pos` → `default_pos_state`) and the CRC/publish step are left as documented `NotImplementedError`s with exact porting instructions, because they cannot be written *or verified* without a real robot and unitree_sdk2py installed, neither of which existed in the environment this fork was built in. **Treat this file as a reviewed starting point, not proven code**, and re-verify every threshold in `safety.py` against your specific robot before trusting it near hardware.
- **`Selector`**: only the simple rule-based `TiltRecoverySelector` exists, and it has no hysteresis — it re-proposes every tick, so a live autonomous selector alongside a human operator will currently override a manual switch on the very next tick. A learned gating/blending network (the active 2025-2026 research direction — see §4) is the natural next step, and only requires implementing the same one-method `propose()` interface; a deadband/override-priority rule is the smaller near-term fix.
- **Networked transport + unified control web exist** (`legged_gym/control/transport.py`, `web/` — see §4a): a JSON-over-WebSocket bridge and a build-step-free browser UI, both driven purely through `ControlService`, nothing new bypasses it.
- **`ObsSpec` enforcement is a warning, not a hard stop**: `PolicySupervisor` checks the incoming observation's shape against each policy's declared spec and warns on mismatch, but doesn't refuse to proceed — every policy you load side-by-side today must genuinely share one observation space (which is true for `stable`/`cautious`/`damping` above, but won't automatically be true for an arbitrary new skill).
- **Episode-reset doesn't reset policy hidden states**: `SimAdapter.send_action()` ignores the env's own `dones` signal (used for RL training's episode termination). Fine for this demo — `SafetyGovernor` already reacts to a fall directly via `projected_gravity` — but a hidden state that should have been cleared on an env-internal reset currently isn't; worth fixing before using this for anything resembling an evaluation run.
- **GPU supported with workarounds**: Genesis on CUDA works via runtime monkey-patches in `genesis_simulator.py` that compensate for Genesis's internal `sanitize_index` CPU-forcing bug. CPU remains the primary tested path (this fork was originally built for Genesis on a GPU-less Mac), but GPU mode is functional.
- **`load_policy()` now accepts ONNX as well as TorchScript** (`policy.py`: `OnnxStatelessPolicy` / `OnnxExplicitStatePolicy`, auto-detected the same way the two jit conventions are, dispatched purely on the `.onnx`/`.pt` file extension — verified bit-identical output against this repo's own TorchScript export for a recurrent policy, see commit history). This closes the actual gap identified below: TorchScript-only vs. the ecosystem's real norm of "export both jit and onnx, pick per consumer" (confirmed against Isaac Lab's `exporter.py`, which exports both from every run). `docker-entrypoint.sh` auto-discovers `*.onnx` from `./policies/` exactly like `*.pt`.
- **External pretrained G1 policies still aren't automatically drop-in — but the remaining gap is obs/action-space compatibility, not file format.** Format is no longer the blocker (see above); every policy loaded side-by-side must still genuinely share this fork's `G1RoughCfg` observation encoding and action space (12 leg DOF, LSTM hidden_size=64 for the recurrent case). Checked against real public G1 releases:
  - **[NVIDIA GEAR-SONIC / GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)** (HF: `nvidia/GEAR-SONIC`) — real ONNX checkpoints, loadable format-wise now, but built around GR00T's own whole-body obs/action encoding, not `G1RoughCfg`'s — would need obs-layout verification (and likely retraining/fine-tuning against our layout) before it's safe to run.
  - **[hardware-pathon-ai/unitree-g1-phase1-locomotion](https://huggingface.co/hardware-pathon-ai/unitree-g1-phase1-locomotion)** — real MIT-licensed `.pt` weights, but only 15/29 DOF active (arms frozen) — action space still doesn't match.
  - **[mujocolab/g1_spinkick_example](https://github.com/mujocolab/g1_spinkick_example)** — real ONNX checkpoint, loadable format-wise, but it's a **full-body** (legs + arms + torso) policy against our **legs-only (12 DOF)** action space — this is a genuine action-space mismatch an adapter can't paper over. Reproducing this trick means extending `G1RoughCfg` to full-body DOF and retraining with trick-specific (motion-imitation) reward shaping in our own pipeline, not loading their checkpoint.
  - **ExBody2 / OmniH2O / HumanPlus / HOVER** — G1-relevant motion-imitation research with public code, but no confirmed public checkpoint release was found (verify each repo directly; this may change).
- **Reward-curve summaries aren't enough to trust a checkpoint — watch it.** A real incident: a crouch retrain (lower `base_height_target` + re-enabled commands/pushes, all in one fine-tune step, on a now-removed `g1_crouch` task) looked plausible enough from the training log to consider done, but measured directly in sim it was actually falling roughly once a second — the per-iteration summary didn't make that obvious, only stepping through it and counting real resets did. See `.claude/skills/rugiar/SKILL.md` ("Training a crouched-but-mobile policy") for the current, `g1`-only recipe that avoids this. Before trusting a new checkpoint (and especially before deleting the `logs/<task>/<run>/` directory it lives in — see below), watch it: `python legged_gym/scripts/play.py --task=<task> --load_run=<run> --ckpt=<N> --viewer=viser --viser_port=9006` opens any single checkpoint live in the browser, not just the final one (docs/index.html §14 has the full recipe and a gotcha about `exported/` getting overwritten on every review).
- **Never delete a `logs/<task>/<run>/` directory without archiving its final raw checkpoint first.** Lost one this way once — `rm -rf`'d a training run's log directory that held the only raw checkpoint (optimizer + critic state) worth resuming from, and the exported inference-only `.pt` in `policies/` wasn't a substitute. Convention now: copy anything worth keeping to `./checkpoints/<task>/` (git-tracked, unlike gitignored `logs/`) before any cleanup — see §2 "Train a policy".
- **`g1`'s survival time plateaus around ~73/1000 episode steps in short curriculum runs, unresolved.** Uncapped `entropy_coef` used to cause outright PPO exploration-noise runaway (`Mean action noise std` climbing monotonically the whole run instead of settling — watch that metric first in any job's log, it should trend down); lowering `entropy_coef` (now a real Create Policy field, not just a CLI flag) fixes the runaway, but two successive short runs at a low value (0.002 → 0.001) held `Mean episode length` essentially flat (73.28 → 72.88) even as noise fell and reward rose. Untried next steps, cheapest first: (1) nudge `entropy_coef` back up slightly (0.003–0.005) — a value chased too low can strand training in a local optimum instead of exploring past whatever causes the fall; (2) read the per-term `Mean episode rew_*` breakdown already sitting in the job logs for what's actually driving early termination, before spending more compute; (3) just budget more raw iterations (`G1RoughCfgPPO.runner.max_iterations=10000` is the task's real target — recent runs are ~37% of that).

### 5a. Team workflow: mixed OS, mixed hardware, shared training compute

This is a real, common pattern in the G1/humanoid-RL community, not a workaround specific to this fork: `mjlab` (the framework behind the spin-kick example above) explicitly requires an NVIDIA GPU for training and documents macOS as evaluation-only — i.e. train-on-a-GPU-box / run-anywhere is the norm, not an adaptation forced by this team's hardware mix.

Recommended split for a team with Mac/Linux/Windows laptops plus occasional access to a powerful (possibly remote/cloud) GPU server:

1. **Train** wherever the GPU is (Genesis already runs on CUDA today via `GENESIS_BACKEND=cuda`, with documented workarounds — see above). Training is the only step that needs a real GPU here.
2. **Export both formats** from that run — `play.py --export_onnx` now produces `policy_lstm_1.pt` *and* `policy_lstm_1.onnx` side by side (§2 "Train a policy"). This mirrors Isaac Lab's own convention rather than inventing a new one.
3. **Share the checkpoint** the way the wider community does — a shared HuggingFace model repo or a wandb artifact (this repo already logs to `wandb`, see `train.py --sync_wandb`) both work; git-lfs also works for a smaller team, but isn't required.
4. **Anyone on any OS runs it** — drop the `.pt` or `.onnx` file into `./policies/` (auto-discovered by `docker-entrypoint.sh`) or pass it directly via `rugiar_driver.py --policy <name>:<path>`. No GPU, no specific OS, and no format conversion needed on the receiving end — Genesis itself runs CPU-only on a GPU-less Mac exactly as it does today.

### 5b. Known follow-ups (Create Policy panel / task registry)

- **The Task dropdown doesn't explain itself.** A one-line note per task — why it's a registered task and not a UI override (§5b's rule) — would make that distinction visible without reading source.
- **Same-name training jobs silently collide.** If two jobs finish under the same policy name, the second `add_policy()` overwrites the first (last-write-wins) — no UI-level warning yet.
- **`num_envs` has no sanity ceiling** in the Create Policy form. A very large value on a CPU-only box is slow, not unsafe, but nothing warns about it.

---

## 6. Roadmap: LLM interfacing

The reason `ControlService` is deliberately a small, explicit set of methods (`request_switch(name)`, `status()`, `pause()`, `resume()`, `estop()`) rather than something more free-form is that this is exactly the shape an LLM tool-calling interface wants: a short list of named, well-typed actions with clear preconditions, sitting behind a safety layer that doesn't trust the caller's judgment about *when* it's safe to act. Wiring an LLM in — as a natural-language front-end that turns "be more careful" into `request_switch("cautious")`, or eventually as the `Selector` itself, proposing switches based on a much richer read of the situation than a tilt threshold — is intentionally left as a separate, later piece of work; today's job was making sure there's one clean, safe call surface for it to eventually call into, identically whether it's talking to a simulation or a real robot.

---

## 7. The full didactic write-up

For the from-zero explanation of everything this README assumes you already know — what a Unitree robot's motors actually are, what PD control and PPO and sim2sim/sim2real mean, walked through with real code from this repo and an interactive demo — see **[docs/index.html](docs/index.html)**. Its final section, **§14 "Working as a team"**, covers reviewing a training checkpoint live before trusting it (the mistake that section exists to prevent — see §5 below), the up-to-date Docker instructions, and which platform (Mac/Linux/Windows) is comfortable for which role.

---

## Credits & license

This fork sits on top of (in order): [legged_gym](https://github.com/leggedrobotics/legged_gym) (ETH Zürich Robotic Systems Lab), [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (Unitree Robotics), and [LeggedGym-Ex](https://github.com/lupinjia/LeggedGym-Ex) (lupinjia) — see [UPSTREAM_README.md](UPSTREAM_README.md) for the full acknowledgements list this fork inherits. Licensed under the same terms as upstream — see [LICENSE](LICENSE).

Within this fork, the Docker Compose setup (§2) — `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh`'s policy auto-discovery, and working CUDA passthrough — was contributed by [Ramiro R. C. (RawthiL)](https://github.com/RawthiL), not the original author: the first real external contribution to this repo, and the reason a Linux/Windows teammate doesn't need a native Python+Genesis setup at all.
