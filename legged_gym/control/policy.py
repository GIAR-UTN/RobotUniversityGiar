"""
Policy backends + the Policy wrapper PolicySupervisor actually holds.

Two jit export conventions exist in the wild for this repo's G1 TorchScript
checkpoints (this was discovered the hard way while building the swap demo —
see the commit history / README for the story):
  - "explicit state":  forward(obs, h, c) -> (action, h, c) — what this
    fork's own play.py exports (legged_gym/utils/helpers.py:PolicyExporterLSTM,
    once play.py is told to use it for a recurrent policy).
  - "internal state":  forward(obs) -> action, with hidden_state/cell_state
    kept as persistent buffers *inside* the jit module, updated in-place.
    This is the convention unitree_rl_gym's own shipped checkpoints use
    (deploy/pre_train/*/motion.pt).

A third format, ONNX, is the wider ecosystem's real interchange standard for
everything that isn't PyTorch-native (Isaac Lab exports both jit and onnx from
every run; community G1 releases like NVIDIA's GR00T-WholeBodyControl or
mjlab-trained tricks ship onnx only). `.onnx` checkpoints are handled the same
way: auto-detected from the file's declared inputs, wrapped behind the same
PolicyBackend protocol as the two jit conventions above, so PolicySupervisor
never needs to know which of the three it's holding. Two onnx shapes exist,
mirroring the jit split:
  - stateless: a single input tensor (obs) -> a single output (action).
  - explicit state: obs + N state input tensors -> action + N updated state
    tensors, in matching order (this repo's own onnx export uses this for its
    LSTM policies — see PolicyExporterLSTM.export()).
  - history: two inputs, `history` + `obs` (in either order, matched by
    name) -> a single action output. This is the DWAQ / dreamwaq-style
    contract: a 5-frame term-major history of the current observation is
    fed to an encoder alongside the current frame. Unlike explicit state,
    the history is not returned by the graph — the wrapper maintains it.

load_policy() auto-detects which of these shapes a given checkpoint file
is and wraps it so the rest of the codebase never needs to know the
difference.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Optional, Protocol

import numpy as np
import torch

try:
    import onnxruntime
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


class PolicyBackend(Protocol):
    def step(self, obs: torch.Tensor) -> torch.Tensor:
        ...

    def reset(self) -> None:
        ...


class ExplicitStatePolicy:
    """forward(obs, h, c) -> (action, h, c).

    device is CPU by default because that's all this fork's demo needed
    (Genesis ran on CPU/Metal, never CUDA, on the Mac this was built on) —
    but it's a real parameter, not an assumption baked into the tensor
    construction, so a GPU env just needs to pass device='cuda' here rather
    than hitting a silent CPU/GPU tensor mismatch inside step()."""

    def __init__(self, module, hidden_size: int, num_envs: int, device: str = "cpu"):
        self.module = module
        self.hidden_size = hidden_size
        self.num_envs = num_envs
        self.h = torch.zeros(1, num_envs, hidden_size, device=device)
        self.c = torch.zeros(1, num_envs, hidden_size, device=device)

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, self.h, self.c = self.module(obs.detach(), self.h, self.c)
        return action

    def reset(self) -> None:
        self.h.zero_()
        self.c.zero_()


class InternalStatePolicy:
    """forward(obs) -> action; hidden/cell state are buffers on the module
    itself. Batch size is fixed to whatever the checkpoint was exported
    with (unitree_rl_gym's own exports use batch=1)."""

    def __init__(self, module):
        self.module = module

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module(obs.detach())

    def reset(self) -> None:
        self.module.hidden_state.zero_()
        self.module.cell_state.zero_()


class StatelessPolicy:
    """forward(obs) -> action, no state at all — a plain (non-recurrent)
    MLP actor, exactly what helpers.py's plain PolicyExporter produces for
    any task whose actor_critic.is_recurrent is False (e.g. g1_deepmimic,
    g1_motion_vis — every g1-family policy before this one was recurrent,
    so this jit shape was never actually loaded through this control stack
    until g1_deepmimic's first real checkpoint surfaced it). Same forward()
    signature as InternalStatePolicy, but MUST NOT be confused with it: an
    InternalStatePolicy module owns hidden_state/cell_state buffers this
    class's module simply doesn't have — reset() here is a no-op rather
    than reaching for buffers that don't exist."""

    def __init__(self, module):
        self.module = module

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module(obs.detach())

    def reset(self) -> None:
        pass


def _fixed_dim(dim, fallback: int) -> int:
    """onnxruntime reports symbolic batch dims (e.g. 'batch') as non-int —
    fall back to 1 (this whole control stack runs num_envs=1, see
    rugiar_driver.py) rather than propagate a string into torch.zeros."""
    return dim if isinstance(dim, int) else fallback


@dataclasses.dataclass
class HistoryObsSpec:
    """The layout a two-input `history` + `obs` ONNX policy expects.

    `term_sizes` is the current-observation term layout in order (e.g. the
    DWAQ contract is [3, 3, 3, 29, 29, 29, 4]). `history_length` frames of
    that whole vector are flattened TERM-MAJOR: for each term, its
    `history_length` frames oldest -> newest, then the next term. That is
    what mjlab's ObservationManager and deploy_mujoco2026 both produce; a
    frame-major flattening is a different, incompatible layout.

    `joint_slices` are the start offsets (within the current 100-dim obs) of
    the terms that are per-joint arrays. `joint_map` is the policy's
    `joint_ids_map` from deploy.yaml: policy index -> env/SDK index. The
    wrapper uses it to turn the env's SDK-ordered joint obs into the
    policy's training order before feeding the graph, and to turn the
    policy's action back into SDK order before the env sees it. Empty means
    identity (the two orders already match, as with the v0 policy)."""
    history_length: int
    term_sizes: tuple[int, ...]
    joint_slices: tuple[int, ...] = ()
    joint_map: tuple[int, ...] = ()

    @property
    def num_obs(self) -> int:
        return sum(self.term_sizes)


_JOINT_TERM_NAMES = {
    "joint_pos", "joint_pos_rel", "joint_vel", "joint_vel_rel",
    "last_action", "actions",
}
_TERM_SIZES = {
    "base_ang_vel": 3,
    "base_ang_vel_body": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "velocity_command": 3,
    "keyboard_velocity_commands": 3,
    "command": 3,
    "gait_phase": 4,
}
_NON_TERM_KEYS = {"scale_first", "use_gym_history", "first_push_history", "depth"}


def history_obs_spec_from_deploy_yaml(path: str) -> Optional[HistoryObsSpec]:
    """Read the `history` contract out of the deploy.yaml that ships next to
    an external checkpoint (the same convention policies/<name>/ already
    uses for deploy.yaml). Returns None if the file is missing, isn't
    YAML/readable, or doesn't describe a history policy — callers turn that
    into a clear "this checkpoint needs its deploy.yaml" error instead of
    guessing a layout.

    Only the observation layout, history length and joint map are read;
    scales/clips/PD gains are the env task's business (see
    mjlab_tasks/velocity/g1_edu_dwaq_env_cfg.py)."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(cfg, dict):
        return None
    observations = cfg.get("observations")
    if not isinstance(observations, dict):
        return None

    default_joint_pos = cfg.get("default_joint_pos")
    joint_count = len(default_joint_pos) if isinstance(default_joint_pos, (list, tuple)) else 0
    if joint_count == 0:
        # Some exporters omit default_joint_pos; the per-joint scale list is
        # the same length, so use it as the fallback.
        for name, term in observations.items():
            if name in _JOINT_TERM_NAMES and isinstance(term, dict):
                scale = term.get("scale") or []
                if isinstance(scale, (list, tuple)) and len(scale) > 1:
                    joint_count = len(scale)
                    break

    term_sizes = []
    term_names = []
    history_lengths = []
    for name, term in observations.items():
        if name in _NON_TERM_KEYS or not isinstance(term, dict):
            continue
        scale = term.get("scale") or []
        if name == "gait_phase" and isinstance(scale, (list, tuple)) and len(scale) > 1:
            size = len(scale)
        elif name in _TERM_SIZES:
            size = _TERM_SIZES[name]
        elif name in _JOINT_TERM_NAMES and joint_count:
            size = joint_count
        elif isinstance(scale, (list, tuple)) and len(scale) > 1:
            size = len(scale)
        else:
            # Can't infer a dimension for this term; the layout is not one
            # this parser can be trusted with.
            return None
        term_sizes.append(size)
        term_names.append(name)
        term_history = term.get("history_length", 1)
        history_lengths.append(int(term_history) if term_history is not None else 1)

    if not term_sizes or not history_lengths:
        return None
    history_length = max(history_lengths)
    if history_length < 1 or any(n != history_length for n in history_lengths):
        # A single flat `history` input only makes sense when every term
        # shares one length. Mixed lengths would need per-term buffers.
        return None

    joint_slices = []
    offset = 0
    for name, size in zip(term_names, term_sizes):
        if name in _JOINT_TERM_NAMES and size == joint_count:
            joint_slices.append(offset)
        offset += size

    joint_map = cfg.get("joint_ids_map")
    if isinstance(joint_map, (list, tuple)) and joint_count and len(joint_map) == joint_count:
        if sorted(int(i) for i in joint_map) != list(range(joint_count)):
            return None
        joint_map = tuple(int(i) for i in joint_map)
    else:
        joint_map = ()

    return HistoryObsSpec(
        history_length=history_length,
        term_sizes=tuple(term_sizes),
        joint_slices=tuple(joint_slices),
        joint_map=joint_map,
    )


class OnnxStatelessPolicy:
    """Single input (obs) -> single output (action). No hidden state to
    carry, so reset() is a no-op — this is the common shape for MLP (non-
    recurrent) policies exported straight from Isaac Lab/mjlab/etc."""

    def __init__(self, session):
        self.session = session
        self.obs_name = session.get_inputs()[0].name
        self.out_name = session.get_outputs()[0].name

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        (action_np,) = self.session.run([self.out_name], {self.obs_name: obs_np})
        return torch.from_numpy(action_np).to(obs.device)

    def reset(self) -> None:
        pass


class OnnxExplicitStatePolicy:
    """obs + N state inputs -> action + N updated state outputs, in matching
    order — the onnx analogue of this file's ExplicitStatePolicy. Covers
    recurrent (LSTM/GRU) policies exported with hidden state as explicit
    tensor I/O rather than baked into the graph."""

    def __init__(self, session, num_envs: int):
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        self.session = session
        self.obs_name = inputs[0].name
        self.action_name = outputs[0].name
        self.state_in_names = [i.name for i in inputs[1:]]
        self.state_out_names = [o.name for o in outputs[1:]]
        self.state_shapes = [
            [_fixed_dim(d, num_envs if idx == 1 else 1) for idx, d in enumerate(i.shape)]
            for i in inputs[1:]
        ]
        self.states = [np.zeros(shape, dtype=np.float32) for shape in self.state_shapes]

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        feed = {self.obs_name: obs_np}
        feed.update(zip(self.state_in_names, self.states))
        outputs = self.session.run([self.action_name, *self.state_out_names], feed)
        action_np, self.states = outputs[0], outputs[1:]
        return torch.from_numpy(action_np).to(obs.device)

    def reset(self) -> None:
        self.states = [np.zeros(shape, dtype=np.float32) for shape in self.state_shapes]


class OnnxHistoryObsPolicy:
    """`history` + `obs` inputs -> action, with the history buffer owned by
    THIS wrapper (not returned by the graph, unlike OnnxExplicitStatePolicy).

    Each step it takes the env's current observation (SDK/mjlab joint order),
    remaps the per-joint slices into the policy's training order, appends
    that frame to a `history_length`-deep buffer, flattens the buffer
    term-major (see HistoryObsSpec), and feeds both `history` and the current
    `obs` to the graph. The returned action is remapped back into SDK order
    so the env's action term — and PolicySupervisor's cross-fade against
    other policies — always see the same joint order.

    reset() empties the buffer; the next step re-seeds all frames with the
    current observation, which is exactly what deploy_mujoco2026's
    ObservationTermCfg.reset() does."""

    def __init__(self, session, spec: HistoryObsSpec):
        self.session = session
        self.spec = spec
        inputs = session.get_inputs()
        names = [i.name for i in inputs]
        if "history" in names and "obs" in names:
            self.history_name, self.obs_name = "history", "obs"
        elif len(names) == 2:
            self.history_name, self.obs_name = names[0], names[1]
        else:
            raise ValueError(
                f"history-observation ONNX expects two inputs (history, obs), got {names}"
            )
        self.action_name = session.get_outputs()[0].name
        self._validate_shapes({i.name: i.shape for i in inputs})
        self.joint_map = (
            torch.as_tensor(spec.joint_map, dtype=torch.long) if spec.joint_map else None
        )
        self._frames: list[torch.Tensor] = []

    def _validate_shapes(self, shapes: dict) -> None:
        obs_shape = shapes[self.obs_name]
        history_shape = shapes[self.history_name]
        if isinstance(obs_shape[-1], int) and obs_shape[-1] != self.spec.num_obs:
            raise ValueError(
                f"ONNX '{self.obs_name}' input is {obs_shape[-1]}-dim but deploy.yaml "
                f"describes {self.spec.num_obs} observation values"
            )
        expected_history = self.spec.num_obs * self.spec.history_length
        if isinstance(history_shape[-1], int) and history_shape[-1] != expected_history:
            raise ValueError(
                f"ONNX '{self.history_name}' input is {history_shape[-1]}-dim but "
                f"{self.spec.history_length} frames x {self.spec.num_obs} values "
                f"= {expected_history}"
            )

    def _policy_order(self, obs: torch.Tensor) -> torch.Tensor:
        if self.joint_map is None:
            return obs
        out = obs.clone()
        n = self.joint_map.numel()
        for start in self.spec.joint_slices:
            out[:, start:start + n] = obs[:, start:start + n][:, self.joint_map]
        return out

    def _flatten_history(self) -> torch.Tensor:
        ranges = []
        offset = 0
        for size in self.spec.term_sizes:
            ranges.append((offset, offset + size))
            offset += size
        return torch.cat(
            [torch.cat([frame[:, a:b] for frame in self._frames], dim=1) for a, b in ranges],
            dim=1,
        )

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        current = self._policy_order(obs).detach()
        if not self._frames:
            self._frames = [current.clone() for _ in range(self.spec.history_length)]
        else:
            self._frames.append(current.clone())
            self._frames.pop(0)
        history = self._flatten_history()
        (action_np,) = self.session.run(
            [self.action_name],
            {
                self.history_name: history.cpu().numpy().astype(np.float32),
                self.obs_name: current.cpu().numpy().astype(np.float32),
            },
        )
        action = torch.from_numpy(action_np).to(obs.device)
        if self.joint_map is not None:
            env_action = torch.empty_like(action)
            env_action[:, self.joint_map] = action
            action = env_action
        return action

    def reset(self) -> None:
        self._frames = []


def load_onnx_backend(path: str, num_envs: int, num_obs: Optional[int] = None) -> PolicyBackend:
    if not HAS_ONNXRUNTIME:
        raise RuntimeError(
            f"Cannot load '{path}': onnx checkpoints require the 'onnxruntime' package "
            "(pip install onnxruntime — see README's community-policy section)."
        )
    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) == 1:
        return OnnxStatelessPolicy(session)
    input_names = [i.name for i in session.get_inputs()]
    if "history" in input_names and "obs" in input_names:
        deploy_yaml = os.path.join(os.path.dirname(os.path.abspath(path)), "deploy.yaml")
        spec = history_obs_spec_from_deploy_yaml(deploy_yaml)
        if spec is None:
            raise RuntimeError(
                f"Cannot load '{path}': its ONNX has a 'history' input, so the wrapper "
                f"needs the observation layout from {deploy_yaml} (term order, "
                f"history_length and joint_ids_map). Ship the policy's deploy.yaml "
                f"next to the checkpoint."
            )
        if num_obs is not None and spec.num_obs != num_obs:
            raise RuntimeError(
                f"Cannot load '{path}': deploy.yaml describes {spec.num_obs}-dim current "
                f"observations but this task produces {num_obs}. Run the task registered "
                f"for this policy's observation contract."
            )
        return OnnxHistoryObsPolicy(session, spec)
    return OnnxExplicitStatePolicy(session, num_envs)


class ZeroActionBackend:
    """The 'damping' fallback: always outputs a zero action, i.e. hold at
    default_dof_pos with whatever Kp/Kd the env config has, no target
    offset. On real hardware this is what SafetyGovernor should switch to
    when something looks wrong (see safety.py) — a real deployment likely
    wants an even more conservative real analogue of this (soft Kd-only
    damping, as deploy_real.py's own zero_torque_state does), but the
    contract (a PolicyBackend that never fights the PD controller) is the
    same idea."""

    def __init__(self, num_envs: int, num_actions: int, device: str = "cpu"):
        self._zeros = torch.zeros(num_envs, num_actions, device=device)

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        return self._zeros

    def reset(self) -> None:
        pass


def damping_policy(num_envs: int, num_actions: int, device: str = "cpu") -> Policy:
    return Policy(
        name="damping",
        backend=ZeroActionBackend(num_envs, num_actions, device=device),
        obs_spec=ObsSpec(num_obs=0, hidden_size=0),
        description="Emergency fallback — holds the default pose, no learned behavior.",
    )


def load_policy_backend(path: str, hidden_size: int, num_envs: int, device: str = "cpu",
                        num_obs: Optional[int] = None) -> PolicyBackend:
    if path.endswith(".onnx"):
        return load_onnx_backend(path, num_envs, num_obs=num_obs)
    module = torch.jit.load(path, map_location=device)
    # forward()'s own declared arg count is what actually distinguishes the
    # three jit shapes — hidden_state/cell_state hasattr() alone isn't
    # enough, since a plain (non-recurrent) PolicyExporter export also has
    # a 2-argument forward(self, obs) but neither buffer at all. 4 args
    # (self, obs, hidden_state, cell_state) is unambiguous either way.
    num_forward_args = len(module.forward.schema.arguments)
    if num_forward_args >= 4:
        return ExplicitStatePolicy(module, hidden_size, num_envs, device=device)
    if hasattr(module, "hidden_state") and hasattr(module, "cell_state"):
        return InternalStatePolicy(module)
    return StatelessPolicy(module)


@dataclasses.dataclass
class ObsSpec:
    """What a policy expects to be fed. PolicySupervisor checks this against
    the obs tensor it's about to pass in and warns (doesn't silently proceed)
    on a mismatch — the single most common way a policy swap goes wrong."""
    num_obs: int
    hidden_size: int


@dataclasses.dataclass
class Policy:
    """One loadable/switchable skill. Bundles the network with everything
    that must travel together with it — if a future skill needs its own PD
    gains or default pose (most won't, on the same robot, but a genuinely
    different gait style might), those overrides go here too rather than
    living implicitly in env config."""
    name: str
    backend: PolicyBackend
    obs_spec: ObsSpec
    description: str = ""


def load_policy(name: str, path: str, num_obs: int, hidden_size: int, num_envs: int,
                 description: str = "", device: str = "cpu") -> Policy:
    backend = load_policy_backend(path, hidden_size, num_envs, device=device, num_obs=num_obs)
    return Policy(name=name, backend=backend, obs_spec=ObsSpec(num_obs, hidden_size), description=description)


def weight_fingerprint(backend: PolicyBackend, max_pixels: int = 480) -> Optional[list]:
    """Real weight tensors from a loaded policy, downsampled for the web
    control panel's "weight fingerprint" strip (one row per 2D weight
    matrix -- Linear/LSTM `weight` params; 1D biases skipped since a
    single-value-per-row strip isn't a useful picture).

    Only works for the three jit backends (ExplicitStatePolicy/
    InternalStatePolicy/StatelessPolicy) -- all three keep the loaded
    module on `.module`, a real torch.jit.ScriptModule, which is a genuine
    nn.Module subclass so `.named_parameters()` works normally. Returns
    None for anything else (OnnxStatelessPolicy/OnnxExplicitStatePolicy --
    only an onnxruntime.InferenceSession, no nn.Module at all --, and
    ZeroActionBackend -- the damping fallback, no weights whatsoever) so
    the caller can render a "not available for this checkpoint format"
    state instead of crashing.

    Row order follows `named_parameters()`'s own order, which for a plain
    Sequential/composed module tracks definition order (input -> output)
    -- true for every checkpoint this fork has produced so far, but not a
    guarantee jit gives you in general, so treat the ordering as "usually
    right," not load-bearing.

    Each row is downsampled to at most `max_pixels` values via evenly
    spaced index sampling (not averaging) -- keeps the WebSocket payload
    small without inventing values that were never actually learned."""
    module = getattr(backend, "module", None)
    if module is None:
        return None
    layers = []
    for pname, p in module.named_parameters():
        if p.dim() != 2:
            continue
        vals = p.detach().flatten().cpu().numpy()
        n = vals.shape[0]
        if n > max_pixels:
            idx = np.linspace(0, n - 1, max_pixels).astype(np.int64)
            vals = vals[idx]
        layers.append({
            "name": pname,
            "shape": list(p.shape),
            "values": [round(float(v), 5) for v in vals],
        })
    return layers
