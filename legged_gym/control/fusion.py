"""
Policy fusion — combining several already-trained policies' weights into one,
without any further training. This module is the pure algorithm (tensor
math + rsl_rl module (re)construction + reusing legged_gym.scripts.play's
own exporter dispatch) and knows nothing about `policies/<name>/`, meta.json,
or POLICIES_DIR — see TrainingManager.fuse_policies() (training.py) for the
orchestration/disk layer built on top of this, exactly the same split
TrainingManager already keeps against web_train.py for regular training.

Today there are two implemented methods:

- "weighted_average" — elementwise weighted-sum of matching weights across
  every source (a.k.a. model soup / SWA-style interpolation). It's cheap and
  works reasonably well for closely related checkpoints (e.g. a fine-tune
  lineage, or same-seed variants), but has no guarantee for independently-
  trained policies: two networks trained from different random inits can
  converge to functionally-equivalent but internally *permuted*
  representations (hidden unit i in one network doesn't correspond to
  hidden unit i in the other), and naively averaging permuted weights
  usually lands in a bad region between the two minima rather than a good
  one.
- "git_rebasin" — solves for the hidden-unit permutation (Ainsworth et al.,
  2022, "Git Re-Basin") that best aligns every non-reference source to the
  first source *before* averaging, via rebasin_align() below, so the merge
  isn't hurt by the permutation-symmetry problem weighted_average has no
  defense against. Handles both plain actor/critic MLPs and rsl_rl
  ActorCriticRecurrent's LSTM/GRU memory (memory_a/memory_c) — see
  rebasin_align()'s docstring for how the RNN's own per-gate permutation
  symmetry is handled and chained into the downstream MLP's alignment.
"""
from __future__ import annotations

import random
import re
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

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
            "Solves for the hidden-unit permutation that best aligns every source to the "
            "first one before averaging, so the merge isn't hurt by permutation symmetry — "
            "believed to be why naive weighted averaging sometimes collapses for policies "
            "that don't share a training lineage. Works for both plain and recurrent "
            "(LSTM/GRU) actor/critic policies."
        ),
        "available": True,
    },
}

# rsl_rl's ActorCritic builds self.actor/self.critic as nn.Sequential(Linear, activation,
# Linear, activation, ..., Linear, <Hardtanh for actor only>) — only the Linear sublayers
# carry state_dict entries, so their integer indices are never contiguous (activation
# modules with no params don't get a key), but they ARE monotonically increasing in the
# order the layers were built, which is all _mlp_dims() below relies on.
_LAYER_RE = re.compile(r"^(actor|critic)\.(\d+)\.weight$")


def _layer_indices(state_dict: Dict[str, torch.Tensor], prefix: str) -> List[int]:
    return sorted(
        int(m.group(2)) for key in state_dict
        if (m := _LAYER_RE.match(key)) and m.group(1) == prefix
    )


def _mlp_dims(state_dict: Dict[str, torch.Tensor], prefix: str):
    idxs = _layer_indices(state_dict, prefix)
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


def _weight_matching_permutations(ref_sd: Dict[str, torch.Tensor], other_sd: Dict[str, torch.Tensor],
                                   prefix: str, iters: int = 100, seed: int = 0,
                                   input_perm: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    """Solves for the permutation of `other_sd`'s `prefix` (actor/critic)
    hidden units that best aligns it to `ref_sd`'s, via the weight-matching
    algorithm from Ainsworth et al. 2022 ("Git Re-Basin"): coordinate
    descent over each hidden layer's permutation, solving a linear
    assignment problem per layer per sweep, until a full sweep changes
    nothing or `iters` sweeps elapse. Only the LAST Linear layer's output is
    excluded — its width is the network's fixed num_actions/1 output, not a
    free hidden dim any permutation could touch. `input_perm`, when given,
    is the permutation the FIRST layer's input dim must already be read
    through instead of identity — used to chain in an upstream RNN's
    already-solved hidden-unit permutation (see rebasin_align()) for a
    recurrent network, where this MLP's real input is the RNN's hidden
    state, not raw observations. Returns one LongTensor (length = that
    layer's width) per remaining layer, in layer order."""
    idxs = _layer_indices(ref_sd, prefix)
    n_free = len(idxs) - 1  # last layer's output isn't a free hidden dim
    perms = [torch.arange(ref_sd[f"{prefix}.{idxs[i]}.weight"].shape[0]) for i in range(n_free)]
    if n_free == 0:
        return perms

    rng = random.Random(seed)
    for _ in range(iters):
        changed = False
        order = list(range(n_free))
        rng.shuffle(order)
        for i in order:
            w_ref, w_other = ref_sd[f"{prefix}.{idxs[i]}.weight"], other_sd[f"{prefix}.{idxs[i]}.weight"]
            if i > 0:
                in_perm = perms[i - 1]
            elif input_perm is not None:
                in_perm = input_perm
            else:
                in_perm = torch.arange(w_ref.shape[1])
            cost = w_ref @ w_other[:, in_perm].T  # (out_dim, out_dim)

            b_ref, b_other = ref_sd.get(f"{prefix}.{idxs[i]}.bias"), other_sd.get(f"{prefix}.{idxs[i]}.bias")
            if b_ref is not None:
                cost = cost + torch.outer(b_ref, b_other)

            w_ref_next = ref_sd[f"{prefix}.{idxs[i + 1]}.weight"]
            w_other_next = other_sd[f"{prefix}.{idxs[i + 1]}.weight"]
            cost = cost + w_ref_next.T @ w_other_next

            _, col_ind = linear_sum_assignment(-cost.detach().numpy())
            new_perm = torch.as_tensor(col_ind, dtype=torch.long)
            if not torch.equal(new_perm, perms[i]):
                changed = True
            perms[i] = new_perm
        if not changed:
            break
    return perms


def _apply_permutations(state_dict: Dict[str, torch.Tensor], prefix: str,
                         perms: Sequence[torch.Tensor],
                         input_perm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Returns `{key: permuted tensor}` for every `prefix.<i>.{weight,bias}`
    entry in `state_dict`, applying `perms` (one per free hidden layer, from
    _weight_matching_permutations()) to each layer's output dim and the
    following layer's matching input dim. `input_perm` mirrors the same
    parameter on _weight_matching_permutations() — must be passed the exact
    same value used to solve `perms`. Meant to be merged into a copy of
    `state_dict`, not used standalone — it only covers `prefix`'s own keys."""
    idxs = _layer_indices(state_dict, prefix)
    out: Dict[str, torch.Tensor] = {}
    for i, layer_idx in enumerate(idxs):
        w = state_dict[f"{prefix}.{layer_idx}.weight"]
        b = state_dict.get(f"{prefix}.{layer_idx}.bias")
        if i > 0:
            in_perm = perms[i - 1]
        elif input_perm is not None:
            in_perm = input_perm
        else:
            in_perm = torch.arange(w.shape[1])
        out_perm = perms[i] if i < len(perms) else torch.arange(w.shape[0])
        out[f"{prefix}.{layer_idx}.weight"] = w[out_perm][:, in_perm]
        if b is not None:
            out[f"{prefix}.{layer_idx}.bias"] = b[out_perm]
    return out


_RNN_GATES = {"lstm": 4, "gru": 3}


def _rnn_weight_matching_permutations(ref_sd: Dict[str, torch.Tensor], other_sd: Dict[str, torch.Tensor],
                                       rnn_prefix: str, info: dict,
                                       downstream_w_ref: torch.Tensor, downstream_w_other: torch.Tensor,
                                       iters: int = 100, seed: int = 0) -> List[torch.Tensor]:
    """RNN counterpart to _weight_matching_permutations(): solves for the
    permutation of `other_sd`'s `rnn_prefix.rnn` (LSTM/GRU) hidden units
    that best aligns it to `ref_sd`'s, one permutation per stacked layer.
    Each layer's weight_ih/weight_hh row blocks repeat once per gate (4 for
    LSTM — i/f/g/o, 3 for GRU) and ALL gates of a layer share the same
    hidden-unit permutation (it's the same cell state being gated), so the
    per-layer cost sums each gate's contribution before solving one
    assignment per layer. weight_hh's own input columns are the layer's OWN
    previous-timestep hidden state (a self-loop), so — unlike every other
    permutation in this module, which only ever depends on already-solved
    upstream permutations — that column read uses this layer's permutation
    from the START of the current sweep; standard Gauss-Seidel-style lag,
    refined again next sweep, same convergence mechanism the outer sweep
    loop already relies on for ordinary coordinate descent.
    `downstream_w_ref`/`downstream_w_other` are the actor/critic MLP's
    first Linear layer's weight (its input dim IS this RNN's last layer's
    hidden output) — folded into the last layer's cost so the RNN and MLP
    permutations solve toward a mutually consistent choice instead of the
    RNN being aligned in isolation."""
    gates = _RNN_GATES[info["type"]]
    hidden_size, num_layers = info["hidden_size"], info["num_layers"]
    perms = [torch.arange(hidden_size) for _ in range(num_layers)]

    rng = random.Random(seed)
    for _ in range(iters):
        changed = False
        order = list(range(num_layers))
        rng.shuffle(order)
        for layer in order:
            w_ih_ref = ref_sd[f"{rnn_prefix}.rnn.weight_ih_l{layer}"]
            w_ih_other = other_sd[f"{rnn_prefix}.rnn.weight_ih_l{layer}"]
            w_hh_ref = ref_sd[f"{rnn_prefix}.rnn.weight_hh_l{layer}"]
            w_hh_other = other_sd[f"{rnn_prefix}.rnn.weight_hh_l{layer}"]
            b_ih_ref = ref_sd.get(f"{rnn_prefix}.rnn.bias_ih_l{layer}")
            b_ih_other = other_sd.get(f"{rnn_prefix}.rnn.bias_ih_l{layer}")
            b_hh_ref = ref_sd.get(f"{rnn_prefix}.rnn.bias_hh_l{layer}")
            b_hh_other = other_sd.get(f"{rnn_prefix}.rnn.bias_hh_l{layer}")

            in_perm = perms[layer - 1] if layer > 0 else torch.arange(w_ih_ref.shape[1])
            self_perm = perms[layer]

            cost = torch.zeros(hidden_size, hidden_size)
            for g in range(gates):
                sl = slice(g * hidden_size, (g + 1) * hidden_size)
                cost = cost + w_ih_ref[sl] @ w_ih_other[sl][:, in_perm].T
                cost = cost + w_hh_ref[sl] @ w_hh_other[sl][:, self_perm].T
                if b_ih_ref is not None:
                    cost = cost + torch.outer(b_ih_ref[sl], b_ih_other[sl])
                if b_hh_ref is not None:
                    cost = cost + torch.outer(b_hh_ref[sl], b_hh_other[sl])
            if layer == num_layers - 1:
                cost = cost + downstream_w_ref.T @ downstream_w_other

            _, col_ind = linear_sum_assignment(-cost.detach().numpy())
            new_perm = torch.as_tensor(col_ind, dtype=torch.long)
            if not torch.equal(new_perm, perms[layer]):
                changed = True
            perms[layer] = new_perm
        if not changed:
            break
    return perms


def _apply_rnn_permutations(state_dict: Dict[str, torch.Tensor], rnn_prefix: str, info: dict,
                             perms: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """RNN counterpart to _apply_permutations() — applies `perms` (from
    _rnn_weight_matching_permutations()) to every `rnn_prefix.rnn.*_l<i>`
    entry, permuting each gate's row block identically (same hidden-unit
    permutation across all gates of a layer) and, for weight_hh, its
    self-recurrent input columns too."""
    gates = _RNN_GATES[info["type"]]
    hidden_size, num_layers = info["hidden_size"], info["num_layers"]
    out: Dict[str, torch.Tensor] = {}
    for layer in range(num_layers):
        in_perm = perms[layer - 1] if layer > 0 else \
            torch.arange(state_dict[f"{rnn_prefix}.rnn.weight_ih_l{layer}"].shape[1])
        self_perm = perms[layer]
        gate_perm = torch.cat([self_perm + g * hidden_size for g in range(gates)])

        w_ih = state_dict[f"{rnn_prefix}.rnn.weight_ih_l{layer}"]
        out[f"{rnn_prefix}.rnn.weight_ih_l{layer}"] = w_ih[gate_perm][:, in_perm]

        w_hh = state_dict[f"{rnn_prefix}.rnn.weight_hh_l{layer}"]
        out[f"{rnn_prefix}.rnn.weight_hh_l{layer}"] = w_hh[gate_perm][:, self_perm]

        b_ih = state_dict.get(f"{rnn_prefix}.rnn.bias_ih_l{layer}")
        if b_ih is not None:
            out[f"{rnn_prefix}.rnn.bias_ih_l{layer}"] = b_ih[gate_perm]
        b_hh = state_dict.get(f"{rnn_prefix}.rnn.bias_hh_l{layer}")
        if b_hh is not None:
            out[f"{rnn_prefix}.rnn.bias_hh_l{layer}"] = b_hh[gate_perm]
    return out


def rebasin_align(ref_sd: Dict[str, torch.Tensor], other_sd: Dict[str, torch.Tensor],
                   iters: int = 100, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Returns a copy of `other_sd` with its hidden units permuted to best
    align with `ref_sd` (same architecture required — call this only after
    architectures_compatible() confirms it), via the Git Re-Basin
    weight-matching algorithm (Ainsworth et al., 2022) — the fix for the
    permutation-symmetry problem plain merge_state_dicts() has no defense
    against. Permuting hidden units this way is a pure relabeling — for any
    single network it doesn't change what function it computes — so
    `other_sd`'s behavior is unaffected by this call on its own; the payoff
    only shows up once the RESULT is averaged against `ref_sd` in
    merge_state_dicts(), landing the merge inside (rather than between) the
    two policies' loss basins.

    Handles both plain (non-recurrent) actor/critic MLPs and rsl_rl
    ActorCriticRecurrent's memory_a/memory_c LSTM/GRU — a recurrent
    network's actor/critic MLP doesn't consume raw observations, it
    consumes the RNN's hidden state, so its first Linear layer's input
    permutation is chained from the RNN's own last-layer permutation
    (`rnn_out_perm` below) rather than left at identity."""
    aligned = dict(other_sd)
    rnn_out_perm: Dict[str, Optional[torch.Tensor]] = {"actor": None, "critic": None}

    for prefix, rnn_prefix in (("actor", "memory_a"), ("critic", "memory_c")):
        info = _rnn_info(ref_sd, rnn_prefix)
        if info is None:
            continue
        first_idx = _layer_indices(ref_sd, prefix)[0]
        downstream_w_ref = ref_sd[f"{prefix}.{first_idx}.weight"]
        downstream_w_other = other_sd[f"{prefix}.{first_idx}.weight"]
        perms = _rnn_weight_matching_permutations(
            ref_sd, other_sd, rnn_prefix, info, downstream_w_ref, downstream_w_other, iters=iters, seed=seed)
        aligned.update(_apply_rnn_permutations(other_sd, rnn_prefix, info, perms))
        rnn_out_perm[prefix] = perms[-1]

    for prefix in ("actor", "critic"):
        if len(_layer_indices(ref_sd, prefix)) < 2:
            continue  # single-layer stack — no interior hidden dim to permute
        perms = _weight_matching_permutations(
            ref_sd, other_sd, prefix, iters=iters, seed=seed, input_perm=rnn_out_perm[prefix])
        aligned.update(_apply_permutations(other_sd, prefix, perms, input_perm=rnn_out_perm[prefix]))
    return aligned


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
