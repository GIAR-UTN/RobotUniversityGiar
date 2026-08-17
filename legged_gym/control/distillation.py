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

- "dagger" (Dataset Aggregation) — one-shot "behavior_cloning" has a hard,
  well-documented ceiling: covariate shift. The student is only ever trained
  on states the TEACHER visited, so the moment its own imperfect actions
  drift it even slightly off that trajectory, it's in a state distribution
  with no correction signal, and error compounds roughly O(T^2) instead of
  O(T) over an episode. dagger_train() below fixes this the standard way:
  roll the env out under a student/teacher action MIX (a decaying `beta`
  fraction is the teacher, so round 0 isn't a totally untrained student
  immediately face-planting), record the actual obs sequence visited, and
  aggregate it — together with every PRIOR round's data, not just the
  latest — into bc_train()'s growing dataset each round. See HANDOFF_dagger_
  distillation.md (repo root, at implementation time) for the full external
  research trail motivating this.
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
        "available": True,
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

    Returns `(obs_buf, action_buf, dones_buf, ground_truth)`. The first three are shaped
    (num_steps, num_envs, dim) / (num_steps, num_envs) as before. `ground_truth` is a dict
    of RAW simulator-state tensors — {"commands", "base_lin_vel", "base_ang_vel"}, each
    (num_steps, num_envs, 3) — recorded straight off `env.commands`/`env.simulator.*`
    every step, for summarize_rollout() below. Deliberately NOT decoded from obs_buf: each
    task subclass (e.g. G1Robot in envs/g1/g1.py) can freely override compute_observations()
    with its own column order/contents (g1's own layout has no base_lin_vel at all, and puts
    commands at columns [6:9] instead of the base LeggedRobot's [9:12]) — decoding obs_buf by
    a fixed slice silently reads the wrong columns on any task that doesn't match the base
    layout exactly, producing plausible-looking but wrong numbers (confirmed the hard way:
    an early version of this function read `dof_pos` deltas as if they were commands on g1).
    Reading simulator state directly, same as legged_robot.py's own actual_lin_vel_x-style
    diagnostics, is layout-independent by construction."""
    obs = env.get_observations()
    obs_list, action_list, done_list = [], [], []
    cmd_list, lin_vel_list, ang_vel_list = [], [], []
    for step in range(num_steps):
        actions = teacher_backend.step(obs.detach())
        obs_list.append(obs.detach().clone())
        action_list.append(actions.detach().clone())
        cmd_list.append(env.commands[:, :3].detach().clone())
        lin_vel_list.append(env.simulator.base_lin_vel.detach().clone())
        ang_vel_list.append(env.simulator.base_ang_vel.detach().clone())
        obs, _, _, dones, _ = env.step(actions.detach())
        done_mask = dones.detach().clone().bool() if torch.is_tensor(dones) else \
            torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        done_list.append(done_mask)
        if torch.any(done_mask):
            teacher_backend.reset()
        if callback is not None:
            callback(step, num_steps)
    ground_truth = {
        "commands": torch.stack(cmd_list),
        "base_lin_vel": torch.stack(lin_vel_list),
        "base_ang_vel": torch.stack(ang_vel_list),
    }
    return torch.stack(obs_list), torch.stack(action_list), torch.stack(done_list), ground_truth


def summarize_rollout(ground_truth: Dict[str, torch.Tensor], action_buf: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Diagnostic-only summary of a collect_rollout() dataset — never fed into bc_train(),
    just printed/logged so a stubbornly-high final_bc_loss (or a clone that visibly doesn't
    move like its teacher) can be told apart from a plain DATA-COVERAGE gap: with
    `--num_envs 1` (the required default for a batch-locked teacher — see this module's
    docstring), the whole BC dataset is a SINGLE rollout_steps-long trajectory under
    whatever commands the env's own resampler happened to draw during that one run —
    nothing guarantees it ever visited the teacher's full command envelope (e.g. turning,
    or a full-speed command) even once. A rollout whose commanded_ang_vel_yaw range never
    leaves ~0 means the student was never shown what the teacher does under a turn command,
    regardless of how low final_bc_loss gets on the (narrow) data it did see.

    `ground_truth` is collect_rollout()'s own return (raw simulator state, not decoded from
    obs — see that function's docstring for why decoding obs_buf by column slice is fragile
    across task subclasses). `action_buf` is collect_rollout()'s own (steps, envs, dim)."""
    commands = ground_truth["commands"].reshape(-1, 3)
    lin_vel = ground_truth["base_lin_vel"].reshape(-1, 3)
    ang_vel = ground_truth["base_ang_vel"].reshape(-1, 3)
    flat_actions = action_buf.reshape(-1, action_buf.shape[-1])

    def stats(t: torch.Tensor) -> Dict[str, float]:
        return {"mean": t.mean().item(), "std": t.std().item(),
                "min": t.min().item(), "max": t.max().item()}

    return {
        "actual_lin_vel_x": stats(lin_vel[:, 0]),
        "actual_lin_vel_y": stats(lin_vel[:, 1]),
        "actual_ang_vel_yaw": stats(ang_vel[:, 2]),
        "commanded_lin_vel_x": stats(commands[:, 0]),
        "commanded_lin_vel_y": stats(commands[:, 1]),
        "commanded_ang_vel_yaw": stats(commands[:, 2]),
        "action_abs_mean": flat_actions.abs().mean().item(),
    }


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


def dagger_train(env, teacher_backend, student: nn.Module, num_rounds: int, round_steps: int,
                  bc_epochs: int, lr: float = 1e-3, beta0: float = 1.0, beta_decay: float = 0.5,
                  num_mini_batches: int = 4, chunk_len: int = 25,
                  callback: Optional[Callable[[str, int, int, int, Optional[float]], None]] = None
                  ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """DAgger: fixes one-shot `bc_train()`'s covariate-shift ceiling (see this
    module's docstring) by training on states the STUDENT actually visits,
    correctly relabeled with what the teacher would have done there.

    For `num_rounds` rounds: roll `env` out for `round_steps` ticks under a
    per-step, per-env MIX of student and teacher actions (`beta = beta0 *
    beta_decay**round` fraction teacher — round 0 defaults to all-teacher so
    the very first rollout isn't an untrained student immediately
    face-planting and only ever visiting "on the ground" states, decaying
    toward mostly-student in later rounds); aggregate this round's (obs,
    label) pairs onto every PRIOR round's (vanilla DAgger — the whole point
    is a training set spanning both old and newly-discovered states); retrain
    `student` in place via `bc_train()` (continued from its current weights,
    not from scratch — cheaper, and standard practice) on the full aggregate.

    Labeling: the teacher backend is run *inline*, once, over the exact same
    obs sequence the mixed policy is producing in the exact order it's
    produced — not a separate replay pass. Since `teacher_backend.step()` is
    a pure function of the input history in order, stepping it forward live
    during the mixed rollout to decide the mix IS already "replaying the obs
    sequence through the teacher in order" (the operation a recurrent
    teacher's hidden-state continuity requires, see collect_rollout()'s and
    HANDOFF_distillation_hidden_state_bug.md's docstrings for why that
    matters) — a second, separate replay pass would recompute the identical
    values.

    A round boundary resets both the teacher's and (if recurrent) the
    student's hidden state, but does NOT reset `env` itself (rollout simply
    continues from wherever the previous round's episode was) — so that
    reset is invisible to the environment's own `dones` signal. To keep
    `bc_train()`'s chunked-BPTT hidden-state resets aligned with what
    actually happened during collection, each round's first recorded step is
    OR'd into `True` in the returned/aggregated dones buffer, same as a real
    episode boundary.

    Returns `(final_loss, obs_buf, action_buf, dones_buf, ground_truth)` —
    same shapes as `collect_rollout()` plus the final round's BC loss, so
    callers (web_distill.py) can feed `(ground_truth, action_buf)` into
    `summarize_rollout()` exactly as they already do for `behavior_cloning`.
    `callback(phase, round_idx, i, total, loss)` fires with
    phase="rollout" (i=step, total=round_steps, loss=None) during collection
    and phase="bc_train" (i=epoch, total=bc_epochs, loss=mean_loss) during
    each round's retrain."""
    device = next(student.parameters()).device
    all_obs, all_actions, all_dones = [], [], []
    all_cmd, all_lin_vel, all_ang_vel = [], [], []
    final_loss = 0.0

    for round_idx in range(num_rounds):
        beta = beta0 * (beta_decay ** round_idx)
        obs = env.get_observations()
        teacher_backend.reset()
        student.reset(dones=torch.ones(env.num_envs, dtype=torch.bool, device=obs.device))

        obs_list, label_list, done_list = [], [], []
        cmd_list, lin_vel_list, ang_vel_list = [], [], []
        with torch.no_grad():
            for step in range(round_steps):
                teacher_action = teacher_backend.step(obs.detach())
                student_action = student.act_inference(obs.detach())
                use_teacher = (torch.rand(env.num_envs, device=obs.device) < beta).unsqueeze(-1)
                mixed_action = torch.where(use_teacher, teacher_action, student_action)

                obs_list.append(obs.detach().clone())
                label_list.append(teacher_action.detach().clone())
                cmd_list.append(env.commands[:, :3].detach().clone())
                lin_vel_list.append(env.simulator.base_lin_vel.detach().clone())
                ang_vel_list.append(env.simulator.base_ang_vel.detach().clone())

                obs, _, _, dones, _ = env.step(mixed_action.detach())
                done_mask = dones.detach().clone().bool() if torch.is_tensor(dones) else \
                    torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
                done_list.append(done_mask)
                if torch.any(done_mask):
                    teacher_backend.reset()
                    student.reset(dones=done_mask)
                if callback is not None:
                    callback("rollout", round_idx, step, round_steps, None)

        done_list[0] = torch.ones_like(done_list[0])  # round boundary == forced hidden-state reset, see docstring
        all_obs.append(torch.stack(obs_list))
        all_actions.append(torch.stack(label_list))
        all_dones.append(torch.stack(done_list))
        all_cmd.append(torch.stack(cmd_list))
        all_lin_vel.append(torch.stack(lin_vel_list))
        all_ang_vel.append(torch.stack(ang_vel_list))

        agg_obs = torch.cat(all_obs, dim=0)
        agg_actions = torch.cat(all_actions, dim=0)
        agg_dones = torch.cat(all_dones, dim=0)

        final_loss = bc_train(
            student, agg_obs, agg_actions, epochs=bc_epochs, lr=lr,
            num_mini_batches=num_mini_batches, chunk_len=chunk_len, dones_buf=agg_dones,
            callback=(lambda epoch, loss: callback("bc_train", round_idx, epoch, bc_epochs, loss))
            if callback is not None else None)

    ground_truth = {
        "commands": torch.cat(all_cmd, dim=0),
        "base_lin_vel": torch.cat(all_lin_vel, dim=0),
        "base_ang_vel": torch.cat(all_ang_vel, dim=0),
    }
    return final_loss, torch.cat(all_obs, dim=0), torch.cat(all_actions, dim=0), torch.cat(all_dones, dim=0), ground_truth
