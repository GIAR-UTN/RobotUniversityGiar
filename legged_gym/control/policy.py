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

load_policy() auto-detects which of these five shapes a given checkpoint file
is and wraps it so the rest of the codebase never needs to know the
difference.
"""
from __future__ import annotations

import dataclasses
from typing import Protocol

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


def _fixed_dim(dim, fallback: int) -> int:
    """onnxruntime reports symbolic batch dims (e.g. 'batch') as non-int —
    fall back to 1 (this whole control stack runs num_envs=1, see
    swap_experiment.py) rather than propagate a string into torch.zeros."""
    return dim if isinstance(dim, int) else fallback


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
        return torch.from_numpy(action_np)

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
        return torch.from_numpy(action_np)

    def reset(self) -> None:
        self.states = [np.zeros(shape, dtype=np.float32) for shape in self.state_shapes]


def load_onnx_backend(path: str, num_envs: int) -> PolicyBackend:
    if not HAS_ONNXRUNTIME:
        raise RuntimeError(
            f"Cannot load '{path}': onnx checkpoints require the 'onnxruntime' package "
            "(pip install onnxruntime — see README's community-policy section)."
        )
    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) == 1:
        return OnnxStatelessPolicy(session)
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


def load_policy_backend(path: str, hidden_size: int, num_envs: int, device: str = "cpu") -> PolicyBackend:
    if path.endswith(".onnx"):
        return load_onnx_backend(path, num_envs)
    module = torch.jit.load(path, map_location=device)
    if hasattr(module, "hidden_state") and hasattr(module, "cell_state"):
        return InternalStatePolicy(module)
    return ExplicitStatePolicy(module, hidden_size, num_envs, device=device)


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
    backend = load_policy_backend(path, hidden_size, num_envs, device=device)
    return Policy(name=name, backend=backend, obs_spec=ObsSpec(num_obs, hidden_size), description=description)
