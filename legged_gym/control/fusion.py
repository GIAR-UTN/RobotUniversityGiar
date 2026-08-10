"""
Policy fusion — combining several already-trained policies' weights into one,
without any further training. This module is the pure algorithm (tensor
math + rsl_rl module (re)construction + reusing legged_gym.scripts.play's
own exporter dispatch) and knows nothing about `policies/<name>/`, meta.json,
or POLICIES_DIR — see TrainingManager.fuse_policies() (training.py) for the
orchestration/disk layer built on top of this, exactly the same split
TrainingManager already keeps against web_train.py for regular training.

Today's only implemented method is "weighted_average" — elementwise
weighted-sum of matching weights across every source (a.k.a. model soup /
SWA-style interpolation). It's cheap and works reasonably well for closely
related checkpoints (e.g. a fine-tune lineage, or same-seed variants), but
has no guarantee for independently-trained policies: two networks trained
from different random inits can converge to functionally-equivalent but
internally *permuted* representations (hidden unit i in one network doesn't
correspond to hidden unit i in the other), and naively averaging permuted
weights usually lands in a bad region between the two minima rather than a
good one. FUSION_METHODS below also lists "git_rebasin" — solving for the
aligning permutation before averaging — as the planned fix for that,
deliberately not implemented yet (see the repo's fusion roadmap in
README.md).
"""
from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

FUSION_METHODS: Dict[str, dict] = {
    "weighted_average": {
        "label": "Weighted average",
        "description": (
            "Elementwise weighted sum of matching weights across every source policy "
            "(a.k.a. model soup / SWA-style interpolation). Cheap and often works well "
            "for closely-related checkpoints (e.g. a fine-tune lineage or same-seed "
            "variants) — no guarantee for independently-trained policies, since their "
            "hidden units aren't necessarily aligned (permutation symmetry)."
        ),
        "available": True,
    },
    "git_rebasin": {
        "label": "Git Re-Basin (permutation alignment)",
        "description": (
            "Solves for the hidden-unit permutation that best aligns two independently-"
            "trained policies before averaging, so the merge isn't hurt by permutation "
            "symmetry — believed to be why naive weighted averaging sometimes collapses "
            "for policies that don't share a training lineage. Planned, not yet "
            "implemented."
        ),
        "available": False,
    },
}

# rsl_rl's ActorCritic builds self.actor/self.critic as nn.Sequential(Linear, activation,
# Linear, activation, ..., Linear, <Hardtanh for actor only>) — only the Linear sublayers
# carry state_dict entries, so their integer indices are never contiguous (activation
# modules with no params don't get a key), but they ARE monotonically increasing in the
# order the layers were built, which is all _mlp_dims() below relies on.
_LAYER_RE = re.compile(r"^(actor|critic)\.(\d+)\.weight$")


def _mlp_dims(state_dict: Dict[str, torch.Tensor], prefix: str):
    idxs = sorted(
        int(m.group(2)) for key in state_dict
        if (m := _LAYER_RE.match(key)) and m.group(1) == prefix
    )
    if not idxs:
        raise ValueError(f"no '{prefix}.<i>.weight' keys found — not an rsl_rl ActorCritic state_dict")
    shapes = [tuple(state_dict[f"{prefix}.{i}.weight"].shape) for i in idxs]
    input_dim = shapes[0][1]
    output_dim = shapes[-1][0]
    hidden_dims = [s[0] for s in shapes[:-1]]
    return input_dim, hidden_dims, output_dim


def _rnn_info(state_dict: Dict[str, torch.Tensor], prefix: str) -> Optional[dict]:
    ih_key = f"{prefix}.rnn.weight_ih_l0"
    if ih_key not in state_dict:
        return None
    w_ih = state_dict[ih_key]
    w_hh = state_dict[f"{prefix}.rnn.weight_hh_l0"]
    hidden_size = w_hh.shape[1]
    gates = w_ih.shape[0] // hidden_size
    rnn_type = {4: "lstm", 3: "gru"}.get(gates)
    if rnn_type is None:
        raise ValueError(
            f"unrecognized RNN gate multiplier ({gates}) in '{prefix}.rnn' — expected 3 (GRU) or 4 (LSTM)")
    num_layers = sum(1 for key in state_dict if re.match(rf"^{re.escape(prefix)}\.rnn\.weight_ih_l\d+$", key))
    return {"type": rnn_type, "hidden_size": hidden_size, "num_layers": num_layers, "input_size": w_ih.shape[1]}


def infer_architecture(state_dict: Dict[str, torch.Tensor]) -> dict:
    """Recovers everything needed to reconstruct the rsl_rl ActorCritic/
    ActorCriticRecurrent that produced `state_dict`, purely from its tensor
    shapes — no live env/task config needed. Raises ValueError on anything
    that doesn't look like one of those two module shapes."""
    actor_in, actor_hidden, num_actions = _mlp_dims(state_dict, "actor")
    critic_in, critic_hidden, critic_out = _mlp_dims(state_dict, "critic")
    if critic_out != 1:
        raise ValueError(f"critic's final layer has {critic_out} outputs, expected 1 (a value function)")

    memory_a = _rnn_info(state_dict, "memory_a")
    memory_c = _rnn_info(state_dict, "memory_c")
    if (memory_a is None) != (memory_c is None):
        raise ValueError("recurrent memory present for only one of actor/critic — inconsistent checkpoint")

    arch = {
        "is_recurrent": memory_a is not None,
        "num_actions": num_actions,
        "actor_hidden_dims": actor_hidden,
        "critic_hidden_dims": critic_hidden,
    }
    if memory_a is not None:
        if (memory_a["type"], memory_a["hidden_size"], memory_a["num_layers"]) != \
           (memory_c["type"], memory_c["hidden_size"], memory_c["num_layers"]):
            raise ValueError("actor/critic RNN configs differ within the same checkpoint")
        if actor_in != memory_a["hidden_size"] or critic_in != memory_c["hidden_size"]:
            raise ValueError("actor/critic MLP input size doesn't match its own RNN's hidden_size")
        arch.update({
            "num_actor_obs": memory_a["input_size"], "num_critic_obs": memory_c["input_size"],
            "rnn_type": memory_a["type"], "rnn_hidden_size": memory_a["hidden_size"],
            "rnn_num_layers": memory_a["num_layers"],
        })
    else:
        arch.update({"num_actor_obs": actor_in, "num_critic_obs": critic_in})
    return arch


_ARCH_FIELDS = (
    "is_recurrent", "num_actions", "actor_hidden_dims", "critic_hidden_dims",
    "num_actor_obs", "num_critic_obs", "rnn_type", "rnn_hidden_size", "rnn_num_layers",
)


def architectures_compatible(a: dict, b: dict) -> Optional[str]:
    """None if `a`/`b` (both from infer_architecture()) describe the same
    network shape (so their state_dicts can be merged key-for-key), else a
    human-readable reason naming the first mismatched field."""
    for field in _ARCH_FIELDS:
        va, vb = a.get(field), b.get(field)
        if va != vb:
            return f"'{field}' differs: {va!r} vs {vb!r}"
    return None


def merge_state_dicts(state_dicts: Sequence[Dict[str, torch.Tensor]],
                       weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Elementwise weighted average of N state_dicts — `weights` is
    normalized to sum to 1 internally, so callers can pass raw un-normalized
    weights (e.g. [1, 1, 1] for a uniform 3-way merge). Every state_dict
    must carry exactly the same keys with exactly the same tensor shapes —
    use infer_architecture()/architectures_compatible() to check that
    *before* calling this, so a mismatch is reported with an architectural
    explanation rather than this function's more generic key/shape error."""
    if len(state_dicts) < 2:
        raise ValueError("need at least 2 state_dicts to merge")
    if len(weights) != len(state_dicts):
        raise ValueError(f"got {len(weights)} weight(s) for {len(state_dicts)} state_dict(s)")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"weights must sum to a positive number, got {total}")
    norm_weights = [w / total for w in weights]

    keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != keys:
            missing = sorted(keys - set(sd.keys()))
            extra = sorted(set(sd.keys()) - keys)
            raise ValueError(f"state_dict #{i} has mismatched keys vs #0 (missing={missing}, extra={extra})")

    merged: Dict[str, torch.Tensor] = {}
    for key in keys:
        shape = state_dicts[0][key].shape
        acc = None
        for i, (sd, w) in enumerate(zip(state_dicts, norm_weights)):
            if sd[key].shape != shape:
                raise ValueError(f"'{key}' has mismatched shape: state_dict #0 is {shape}, #{i} is {sd[key].shape}")
            term = sd[key].float() * w
            acc = term if acc is None else acc + term
        merged[key] = acc.to(state_dicts[0][key].dtype)
    return merged


def build_actor_critic(arch: dict, state_dict: Dict[str, torch.Tensor], activation: str = "elu") -> nn.Module:
    """Instantiates the rsl_rl module `arch` (from infer_architecture())
    describes and loads `state_dict` (e.g. merge_state_dicts()'s output)
    into it. `activation` isn't recoverable from a state_dict (it has no
    learned parameters) — the caller supplies it, normally read straight off
    the source policy's own task config (see TrainingManager.fuse_policies())."""
    if arch["is_recurrent"]:
        from rsl_rl.modules import ActorCriticRecurrent
        actor_critic = ActorCriticRecurrent(
            num_actor_obs=arch["num_actor_obs"], num_critic_obs=arch["num_critic_obs"],
            num_actions=arch["num_actions"], actor_hidden_dims=arch["actor_hidden_dims"],
            critic_hidden_dims=arch["critic_hidden_dims"], activation=activation,
            rnn_type=arch["rnn_type"], rnn_hidden_size=arch["rnn_hidden_size"],
            rnn_num_layers=arch["rnn_num_layers"],
        )
    else:
        from rsl_rl.modules import ActorCritic
        actor_critic = ActorCritic(
            num_actor_obs=arch["num_actor_obs"], num_critic_obs=arch["num_critic_obs"],
            num_actions=arch["num_actions"], actor_hidden_dims=arch["actor_hidden_dims"],
            critic_hidden_dims=arch["critic_hidden_dims"], activation=activation,
        )
    actor_critic.load_state_dict(state_dict, strict=True)
    return actor_critic


def export_actor_critic(actor_critic: nn.Module, out_dir: str, env_cfg, train_cfg, task_type: str,
                         export_onnx: bool = False) -> str:
    """Exports `actor_critic` to a deployable TorchScript checkpoint in
    `out_dir`, by reusing legged_gym.scripts.play.export_policy() verbatim —
    the SAME exporter-dispatch-by-task_type logic and PolicyExporter*
    classes (legged_gym/utils/helpers.py) a real training run's checkpoint
    goes through, so a fused policy is guaranteed loadable by
    control/policy.py's load_policy_backend() exactly like a trained one.
    export_policy() only ever touches `alg_runner.alg.actor_critic` — no
    live env/simulator needed — so a minimal duck-typed namespace stands in
    for the full rsl_rl OnPolicyRunner it normally expects. Returns the path
    to the single .pt file produced (raises if the exporter produced zero or
    more than one — would mean a task_type this function doesn't know how
    to locate output for)."""
    from legged_gym.scripts.play import export_policy

    fake_runner = types.SimpleNamespace(alg=types.SimpleNamespace(actor_critic=actor_critic))
    args = types.SimpleNamespace(export_onnx=export_onnx)
    export_policy(fake_runner, out_dir, args, env_cfg, train_cfg, task_type)

    produced = sorted(Path(out_dir).glob("*.pt"))
    if len(produced) != 1:
        raise RuntimeError(
            f"expected exactly one exported .pt file in {out_dir}, found {[p.name for p in produced]}")
    return str(produced[0])
