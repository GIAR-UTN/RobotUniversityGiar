"""
Time-boxed training "packages" — the reusable answer to requests shaped like
"give me N minutes of training improving X" or "N minutes improving Y, sacrificing
at most P% of X", instead of one-off scripts per request.

Two objectives today:
  --objective stability
      Train with ONLY the task's stability-related rewards active (falling/tilting/
      command-tracking/etc.) and the open-ended crouch_depth reward OFF (scale 0).
      After training, evaluate the final checkpoint's stability_score (see below) and
      write it to checkpoints/<task>/stability_baseline.json — this becomes the
      reference every later "bounded sacrifice" package is measured against.

  --objective stability_bounded
      Same stability rewards PLUS the open-ended crouch_depth reward (see
      legged_gym/envs/g1/g1.py:_reward_crouch_depth) turned on: reward is directly
      proportional to how far below crouch_depth_reference the pelvis currently is,
      no fixed target height — "get as low as you can" rather than "reach height H".
      After training, several checkpoints spaced through the run are each evaluated;
      whichever has the LOWEST mean height among those whose stability_score is within
      --max_sacrifice_pct of the baseline is the one kept. If none qualify, the closest
      one is reported and the report says so explicitly — never silently pick a
      checkpoint that broke the bound.

stability_score definition (see _eval_checkpoint): fraction of env-steps, across ALL
parallel envs over --eval_steps ticks post-training, that did NOT end in an
environment reset (i.e. 1 - falls/total env-steps). Falls are the training env's own
termination condition (contact-force-based), the same one that ends an episode during
training — not the demo's SafetyGovernor tilt threshold, which is a different, looser
detector (see README §5).

Usage:
    python legged_gym/scripts/train_package.py --task g1_crouch --minutes 20 \\
        --objective stability

    python legged_gym/scripts/train_package.py --task g1_crouch --minutes 20 \\
        --objective stability_bounded --max_sacrifice_pct 5 \\
        --from_checkpoint checkpoints/g1_crouch/<name>.pt
"""
import argparse
import json
import os
import time

import torch

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import PolicyExporterLSTM

REPO_ROOT = LEGGED_GYM_ROOT_DIR
ITER_SECONDS_DEFAULT = 0.95  # measured throughput @ num_envs=64, CPU, this Mac — see HANDOFF/README


def eval_checkpoint(ppo_runner, env, checkpoint_path, eval_steps):
    """Load a raw checkpoint's weights (no optimizer state) into ppo_runner's actor_critic
    and roll it forward eval_steps ticks across every parallel env, counting real resets
    (falls) and average pelvis height. Returns (stability_score, mean_height)."""
    ppo_runner.load(checkpoint_path, load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=env.device)
    ppo_runner.alg.actor_critic.memory_a.hidden_states = None  # drop training's leftover hidden state

    obs = env.get_observations() if hasattr(env, "get_observations") else env.obs_buf
    total_resets = 0
    height_sum = 0.0
    height_count = 0
    with torch.inference_mode():  # matches how on_policy_runner itself calls the policy during rollout
        for _ in range(eval_steps):
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())
            total_resets += dones.sum().item()
            height_sum += env.simulator.base_pos[:, 2].sum().item()
            height_count += env.simulator.base_pos.shape[0]

    total_env_steps = env.num_envs * eval_steps
    stability_score = 1.0 - (total_resets / total_env_steps)
    mean_height = height_sum / height_count
    return stability_score, mean_height


def main():
    parser = argparse.ArgumentParser(description="Time-boxed training package")
    parser.add_argument('--task', type=str, default='g1_crouch')
    parser.add_argument('--minutes', type=float, required=True, help="wall-clock budget for this package")
    parser.add_argument('--objective', type=str, required=True, choices=['stability', 'stability_bounded'])
    parser.add_argument('--max_sacrifice_pct', type=float, default=5.0,
                         help="stability_bounded only: max allowed drop vs. the stability baseline")
    parser.add_argument('--crouch_depth_scale', type=float, default=2.0,
                         help="stability_bounded only: weight on the open-ended crouch_depth reward")
    parser.add_argument('--from_checkpoint', type=str, default=None,
                         help="raw checkpoint (with optimizer/critic state) to resume from")
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--iter_seconds', type=float, default=ITER_SECONDS_DEFAULT)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--num_candidates', type=int, default=3,
                         help="stability_bounded only: checkpoints spaced through the run to evaluate")
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--cpu', action='store_true', default=True)
    cli = parser.parse_args()

    num_candidates = 1 if cli.objective == 'stability' else cli.num_candidates
    reserved_eval_s = num_candidates * cli.eval_steps * cli.iter_seconds
    train_seconds = max(cli.minutes * 60 - reserved_eval_s, cli.iter_seconds * 50)
    max_iterations = int(train_seconds / cli.iter_seconds)
    print(f"Budget: {cli.minutes} min total, ~{reserved_eval_s:.0f}s reserved for eval "
          f"({num_candidates} candidate(s) x {cli.eval_steps} steps) -> {max_iterations} training iterations")

    args = argparse.Namespace(
        task=cli.task, headless=cli.headless, cpu=cli.cpu, num_envs=cli.num_envs,
        max_iterations=max_iterations, resume=False, sync_wandb=False, export_onnx=False,
        debug=False, load_run=None, ckpt=-1, use_joystick=False, joystick_type='xbox',
        follow_robot=False, viewer='native', viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )

    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level='warning')

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.rewards.scales.base_height = 0.0  # replaced by the open-ended crouch_depth reward
    env_cfg.rewards.scales.crouch_depth = cli.crouch_depth_scale if cli.objective == 'stability_bounded' else 0.0

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    base_iteration = 0
    if cli.from_checkpoint:
        print(f"Loading base weights from: {cli.from_checkpoint} (optimizer state NOT carried over)")
        ppo_runner.load(cli.from_checkpoint, load_optimizer=False)
        base_iteration = ppo_runner.current_learning_iteration

    log_dir = ppo_runner.log_dir
    os.makedirs(log_dir, exist_ok=True)

    t0 = time.time()
    ppo_runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    train_wall_s = time.time() - t0
    final_iteration = base_iteration + max_iterations  # on_policy_runner saves/reports ABSOLUTE iteration
    print(f"Training done: {max_iterations} iterations (base {base_iteration} -> {final_iteration}) in {train_wall_s:.0f}s")

    # ---- candidate checkpoints ----
    # Checkpoint filenames are the ABSOLUTE iteration count (on_policy_runner's own counter, which
    # continues from base_iteration on a resume), not the relative count just trained this package —
    # candidate_iters must be computed in that same absolute space or model_{it}.pt won't exist.
    save_interval = 50  # LeggedRobotCfgPPO.runner.save_interval — checkpoints only exist at multiples of this,
                         # EXCEPT the final iteration, which on_policy_runner.py always saves unconditionally.

    if cli.objective == 'stability':
        candidate_iters = [final_iteration]
    else:
        # Round each ABSOLUTE target iteration to the nearest saved multiple of save_interval —
        # rounding a relative offset and then adding base_iteration (the previous bug) only lands
        # on a multiple of save_interval when base_iteration itself already is one, which isn't
        # true after a resume (e.g. base_iteration=1163).
        aligned = {
            min(final_iteration, max(save_interval, round((base_iteration + max_iterations * f / num_candidates) / save_interval) * save_interval))
            for f in range(1, num_candidates)
        }
        aligned.add(final_iteration)
        candidate_iters = sorted(aligned)

    results = []
    for it in candidate_iters:
        path = os.path.join(log_dir, f"model_{it}.pt")
        if not os.path.exists(path):
            print(f"  skip iteration {it}: no checkpoint at {path}")
            continue
        score, height = eval_checkpoint(ppo_runner, env, path, cli.eval_steps)
        print(f"  iteration {it}: stability_score={score:.4f} mean_height={height:.3f}")
        results.append({"iteration": it, "path": path, "stability_score": score, "mean_height": height})

    if not results:
        raise RuntimeError(
            f"No candidate checkpoints found among {candidate_iters} in {log_dir} — "
            f"training itself may have produced no saves, or save_interval changed mid-run."
        )

    task_ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", cli.task)
    os.makedirs(task_ckpt_dir, exist_ok=True)
    baseline_path = os.path.join(task_ckpt_dir, "stability_baseline.json")

    report = {
        "task": cli.task, "objective": cli.objective, "minutes_requested": cli.minutes,
        "max_iterations": max_iterations, "train_wall_seconds": train_wall_s,
        "crouch_depth_scale": cli.crouch_depth_scale, "candidates": results,
    }

    if cli.objective == 'stability':
        winner = results[0]
        with open(baseline_path, "w") as f:
            json.dump({"stability_score": winner["stability_score"],
                       "mean_height": winner["mean_height"],
                       "run": log_dir, "iteration": winner["iteration"]}, f, indent=2)
        print(f"Baseline written: {baseline_path} (stability_score={winner['stability_score']:.4f})")
    else:
        if not os.path.exists(baseline_path):
            raise RuntimeError(f"No baseline at {baseline_path} — run --objective stability first")
        baseline = json.load(open(baseline_path))
        floor = baseline["stability_score"] * (1 - cli.max_sacrifice_pct / 100)
        report["baseline"] = baseline
        report["stability_floor"] = floor
        qualifying = [r for r in results if r["stability_score"] >= floor]
        if qualifying:
            winner = min(qualifying, key=lambda r: r["mean_height"])
            report["bound_satisfied"] = True
        else:
            winner = max(results, key=lambda r: r["stability_score"])
            report["bound_satisfied"] = False
            print(f"WARNING: no candidate met the {cli.max_sacrifice_pct}% sacrifice bound "
                  f"(floor={floor:.4f}) — reporting closest instead: {winner}")

    report["winner"] = winner

    # ---- export the winning checkpoint ----
    ppo_runner.load(winner["path"], load_optimizer=False)
    exporter = PolicyExporterLSTM(ppo_runner.alg.actor_critic)
    exported_dir = os.path.join(log_dir, "exported")
    exporter.export(exported_dir, env_cfg, export_onnx=True)

    name = f"{cli.task}_{cli.objective}"
    ckpt_out = os.path.join(task_ckpt_dir, f"{name}.pt")
    policies_dir = os.path.join(REPO_ROOT, "policies")
    os.makedirs(policies_dir, exist_ok=True)
    import shutil
    shutil.copy(winner["path"], ckpt_out)
    shutil.copy(os.path.join(log_dir, "exported", "policy_lstm_1.pt"),
                os.path.join(policies_dir, f"{name}.pt"))
    shutil.copy(os.path.join(log_dir, "exported", "policy_lstm_1.onnx"),
                os.path.join(policies_dir, f"{name}.onnx"))
    report["exported_policy"] = f"policies/{name}.pt"
    report["archived_checkpoint"] = f"checkpoints/{cli.task}/{name}.pt"

    report_path = os.path.join(task_ckpt_dir, f"package_report_{cli.objective}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
