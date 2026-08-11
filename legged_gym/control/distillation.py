"""
Policy distillation — behavior-cloning an already-trained TEACHER policy
(loaded via legged_gym.control.policy.load_policy_backend(), so ANY of that
module's five checkpoint shapes work: TorchScript/onnx, stateful/stateless)
into a freshly-initialized rsl_rl ActorCritic/ActorCriticRecurrent matching a
target task's own architecture.

Unlike legged_gym/control/fusion.py, the source policy doesn't need to be an
rsl_rl state_dict at all — fusion merges WEIGHTS, this clones BEHAVIOR (obs ->
action). That's exactly what lets an externally-sourced policy with no
train_checkpoint.pt (e.g. policies/stable.pt — see
TrainingManager._train_checkpoint_from_export()'s docstring in training.py
for why that one can never get one the normal way) become fine-tunable: a BC-
trained student IS a normal rsl_rl checkpoint, random-init critic and all,
exactly like the start of any other training run.

See TrainingManager.start_distillation() (training.py) for the
orchestration/disk layer built on top of this — same pure-algorithm-vs-
orchestration split fusion.py already keeps against TrainingManager.
fuse_policies().

Today there is one implemented method:

- "behavior_cloning" — roll the teacher for `rollout_steps` env ticks in the
  target task's own simulator, collect (obs, teacher_action) pairs, then
  supervise-train a fresh student with MSE against the teacher's action
  (nn.functional.mse_loss on actor_critic.act_inference(obs) — the same loss
  convention rsl_rl/algorithms/ppo_ts.py's _compute_encoder_loss() already
  uses in this codebase for its own distillation-style loss, just between
  two full policies here instead of two encoders). The critic is left at
  random init — it isn't distilled, since the teacher exposes no compatible
  value function — PPO fine-tuning warms it up on its own once training
  resumes, same as the start of any other run.

"dagger" (interleaving student rollouts with teacher relabeling, to correct
for BC's covariate-shift problem — the student never visits its OWN
mistakes during one-shot BC, only the teacher's trajectory) is listed below
for UI/CLI parity with fusion.FUSION_METHODS's own "planned but not
implemented" entries, but not implemented in this pass.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

DISTILL_METHODS: Dict[str, dict] = {
    "behavior_cloning": {
        "label": "Behavior cloning",
        "description": (
            "Rolls the teacher policy through the target task's own simulator, collecting "
            "(observation, action) pairs, then supervise-trains a fresh network to imitate "
            "them via MSE. One-shot — the student never acts during data collection, so "
            "distribution shift between the states the student would visit on its own and "
            "the teacher's own trajectory isn't corrected for (that's what DAgger is for)."
        ),
        "available": True,
    },
    "dagger": {
        "label": "DAgger (dataset aggregation)",
        "description": (
            "Interleaves student rollouts with teacher relabeling to correct behavior "
            "cloning's covariate-shift problem — the student visits its OWN mistakes and "
            "gets corrected on them, not just the teacher's trajectory."
        ),
        "available": False,
    },
}


def check_dimensions_compatible(teacher_backend, num_obs: int, num_actions: int,
                                 num_envs: int, device: str = "cpu") -> None:
    """Fail-fast dummy forward pass through the teacher — same "reject up
    front, don't silently produce garbage" precedent training.py's
    fuse_policies() already establishes for its export_task obs/action
    hard-check (training.py:739-745). A shape mismatch here means the
    teacher wasn't trained on this task's observation/action convention — a
    real risk since, unlike fuse_policies' sources, there is zero
    architectural metadata to compare against up front here; the teacher
    might be a completely different network format (see
    legged_gym/control/policy.py's module docstring for the five shapes it
    could be)."""
    probe_obs = torch.zeros(num_envs, num_obs, device=device)
    try:
        probe_actions = teacher_backend.step(probe_obs)
    except Exception as e:
        raise ValueError(
            f"teacher policy rejected a ({num_envs}, {num_obs})-shaped observation (the "
            f"target task's own obs size) — it likely wasn't trained on this task's "
            f"observation convention: {e}") from None
    if tuple(probe_actions.shape) != (num_envs, num_actions):
        raise ValueError(
            f"teacher policy produced actions shaped {tuple(probe_actions.shape)}, expected "
            f"({num_envs}, {num_actions}) for this target task — architecture mismatch")
    teacher_backend.reset()


def build_student(num_obs: int, num_critic_obs: int, num_actions: int, policy_cfg,
                   is_recurrent: bool) -> nn.Module:
    """Instantiates a FRESH (randomly-initialized) rsl_rl ActorCritic/
    ActorCriticRecurrent matching the target task's architecture. Unlike
    fusion.build_actor_critic(), which reconstructs a network AROUND an
    existing state_dict (infer_architecture()'s whole point), there is no
    state_dict to infer from here — BC trains this network from scratch, so
    it's built directly off the task's own train_cfg.policy, exactly like
    rsl_rl.runners.OnPolicyRunner itself does (on_policy_runner.py) when it
    starts a policy from random init."""
    kwargs = dict(
        num_actor_obs=num_obs, num_critic_obs=num_critic_obs, num_actions=num_actions,
        actor_hidden_dims=list(policy_cfg.actor_hidden_dims),
        critic_hidden_dims=list(policy_cfg.critic_hidden_dims),
        activation=policy_cfg.activation, init_noise_std=getattr(policy_cfg, "init_noise_std", 1.0),
    )
    if is_recurrent:
        from rsl_rl.modules import ActorCriticRecurrent
        return ActorCriticRecurrent(
            rnn_type=policy_cfg.rnn_type, rnn_hidden_size=policy_cfg.rnn_hidden_size,
            rnn_num_layers=policy_cfg.rnn_num_layers, **kwargs)
    from rsl_rl.modules import ActorCritic
    return ActorCritic(**kwargs)


@torch.no_grad()
def collect_rollout(env, teacher_backend, num_steps: int,
                     callback: Optional[Callable[[int, int], None]] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drives `env.step()` with the teacher's own actions for `num_steps`
    ticks — the exact vanilla-task loop legged_gym/scripts/play.py's
    interaction_loop() already uses (`actions = policy(obs_buf.detach());
    obs_buf, _, rews, dones, infos = env.step(actions.detach())`) — and
    stacks every (obs, action) pair seen. `env.step()` auto-resets any env
    whose episode ended (standard vectorized-env convention this whole
    codebase relies on elsewhere), but a stateful teacher (LSTM-backed) must
    ALSO have its own hidden state reset at that exact boundary — otherwise
    it keeps conditioning on stale memory from the terminated episode while
    `obs` already reflects the freshly-reset robot, corrupting every
    (obs, action) pair collected until the next reset. `teacher_backend`
    exposes a whole-batch `reset()` (see policy.py), which is the correct
    per-env reset as long as `num_envs == 1` — the convention this whole
    distillation feature requires teachers to run under.

    Returns `(obs_buf, action_buf, dones_buf)`, each shaped
    (num_steps, num_envs, dim) for the first two and (num_steps, num_envs)
    for `dones_buf` — kept as separate per-step tensors (not flattened) so
    bc_train() can choose to flatten (non-recurrent student) or walk them in
    temporal order, resetting hidden state at the real episode boundaries
    `dones_buf` records (recurrent student)."""
    obs = env.get_observations()
    obs_list, action_list, done_list = [], [], []
    for step in range(num_steps):
        actions = teacher_backend.step(obs.detach())
        obs_list.append(obs.detach().clone())
        action_list.append(actions.detach().clone())
        obs, _, _, dones, _ = env.step(actions.detach())
        done_mask = dones.detach().clone().bool() if torch.is_tensor(dones) else \
            torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        done_list.append(done_mask)
        if torch.any(done_mask):
            teacher_backend.reset()
        if callback is not None:
            callback(step, num_steps)
    return torch.stack(obs_list), torch.stack(action_list), torch.stack(done_list)


def bc_train(student: nn.Module, obs_buf: torch.Tensor, action_buf: torch.Tensor, epochs: int,
             lr: float = 1e-3, num_mini_batches: int = 4, chunk_len: int = 25,
             dones_buf: Optional[torch.Tensor] = None,
             callback: Optional[Callable[[int, float], None]] = None) -> float:
    """Supervise-trains `student` (from build_student(), or any rsl_rl
    ActorCritic/ActorCriticRecurrent) to imitate `(obs_buf, action_buf)`
    (from collect_rollout()) via MSE on the actor's deterministic output
    (`act_inference()` — no sampling, matching how ppo_ts.py's own
    `_compute_encoder_loss()` computes an MSE distillation loss between two
    of THIS codebase's networks already). `callback(epoch, mean_loss)` fires
    once per epoch, e.g. for a caller to write a progress file. Returns the
    final epoch's mean loss.

    Non-recurrent students: obs_buf/action_buf are flattened across (steps,
    envs) and shuffled into `num_mini_batches` mini-batches per epoch —
    ordinary supervised BC, no temporal structure to preserve, same
    `num_mini_batches` convention rsl_rl's own PPO already uses elsewhere in
    this codebase.

    Recurrent students: shuffling would break the whole point of an RNN
    (each step's hidden state depends on having actually seen the preceding
    steps in order), so each epoch instead walks obs_buf/action_buf in their
    original temporal order, split into `chunk_len`-step chunks (a pure
    truncated-BPTT window, DETACHED — not reset — between chunks so
    temporal continuity carries across the boundary), with the hidden state
    reset to zero at the start of every epoch AND, if `dones_buf` (from
    collect_rollout()) is given, again — per-env, via the student's own
    masked `reset(dones)` — at every real episode boundary it records. This
    matters: without it, the hidden state silently keeps flowing across a
    point where the observed trajectory actually restarted from scratch,
    training the student to condition its action on a memory of an episode
    that (per the actual data) no longer exists."""
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    num_steps, num_envs = obs_buf.shape[0], obs_buf.shape[1]
    final_loss = 0.0

    if not student.is_recurrent:
        flat_obs = obs_buf.reshape(num_steps * num_envs, -1)
        flat_actions = action_buf.reshape(num_steps * num_envs, -1)
        num_samples = flat_obs.shape[0]
        batch_size = max(1, num_samples // num_mini_batches)
        for epoch in range(epochs):
            perm = torch.randperm(num_samples)
            batch_losses = []
            for start in range(0, num_samples, batch_size):
                idx = perm[start:start + batch_size]
                pred = student.act_inference(flat_obs[idx])
                loss = nn.functional.mse_loss(pred, flat_actions[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            final_loss = sum(batch_losses) / len(batch_losses)
            if callback is not None:
                callback(epoch, final_loss)
        return final_loss

    for epoch in range(epochs):
        student.reset(dones=torch.ones(num_envs, dtype=torch.bool, device=obs_buf.device))
        chunk_losses = []
        for start in range(0, num_steps, chunk_len):
            end = min(start + chunk_len, num_steps)
            chunk_loss = torch.zeros((), device=obs_buf.device)
            for t in range(start, end):
                pred = student.act_inference(obs_buf[t])
                chunk_loss = chunk_loss + nn.functional.mse_loss(pred, action_buf[t])
                if dones_buf is not None and torch.any(dones_buf[t]):
                    student.reset(dones=dones_buf[t])
            chunk_loss = chunk_loss / (end - start)
            optimizer.zero_grad()
            chunk_loss.backward()
            optimizer.step()
            chunk_losses.append(chunk_loss.item())
            # Truncated BPTT: cut the graph so the NEXT chunk's backward()
            # doesn't walk back through this chunk's already-consumed graph,
            # while keeping the actual hidden-state VALUES (a plain detach,
            # not a reset) so temporal continuity across the chunk boundary
            # is preserved.
            for memory in (student.memory_a, student.memory_c):
                if memory.hidden_states is None:
                    continue
                memory.hidden_states = (
                    tuple(h.detach() for h in memory.hidden_states)
                    if isinstance(memory.hidden_states, tuple) else memory.hidden_states.detach()
                )
        final_loss = sum(chunk_losses) / len(chunk_losses)
        if callback is not None:
            callback(epoch, final_loss)
    return final_loss
