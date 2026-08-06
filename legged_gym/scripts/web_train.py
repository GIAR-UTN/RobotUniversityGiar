"""
Training worker launched by legged_gym/control/training.py (TrainingManager)
as a plain subprocess, one per "Create Policy" job started from the control
web's sidebar — not meant to be hand-invoked normally, though every flag
below is exactly what that UI composes and shows in its command preview (the
whole point of that panel is to expose the real command, not hide it behind
a form — see HANDOFF_control_web.md's Stage E notes).

Trains one policy end-to-end and exports it as a .pt file loadable by
legged_gym.control.policy.load_policy(), then writes its path to
--result_path as JSON so the (non-blocking) parent process knows where to
find it once this exits.

Three "target" concepts the UI exposes map directly onto flags here:
  - relative target: --from_checkpoint fine-tunes an existing policy's
    weights (same pattern as scripts/finetune_from_checkpoint.py) instead of
    training from random init — optimizer state is intentionally NOT
    carried over (see finetune_from_checkpoint.py's docstring for why). The
    new policy is defined relative to an old one.
  - measurement target: --cmd_*_range overrides the velocity command
    envelope (env_cfg.commands.ranges) the policy is trained across — a
    measured quantity (m/s, rad/s), not a reward-shaping knob.
  - stability target: --base_height_target/--push_robots/--max_push_vel_xy/
    --push_interval_s/--push_dir override env_cfg.rewards.base_height_target
    and env_cfg.domain_rand.* directly — for training a policy to hold a
    SPECIFIC pelvis height (e.g. matching a crouch checkpoint's height,
    rather than the task's own default) under push disturbances, optionally
    biased from one direction (--push_dir) instead of every direction at
    once — see Simulator.sample_push_vel_xy() for the angle convention.

Time budget is EITHER or BOTH of --max_iterations and --max_minutes —
whichever is hit first stops training (see the chunked learn() loop below).
At least one must be given.

Usage (as composed by the web UI; also runnable by hand):
    python legged_gym/scripts/web_train.py --task g1 --name my_policy \\
        --max_iterations 500 --max_minutes 30 --num_envs 64 --headless --cpu \\
        --from_checkpoint logs/g1/<run>/model_1800.pt \\
        --cmd_vx_range -0.5 0.5 --base_height_target 0.5 --push_robots on \\
        --reward_scale tracking_lin_vel 0.5 --reward_scale action_rate -0.1 \\
        --result_path /tmp/result.json
"""
import argparse
import json
import os
import time

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.scripts.play import export_policy

# Iterations run per learn() call before re-checking the wall-clock deadline.
# Small enough for a --max_minutes budget to be honored within roughly this
# many iterations' worth of overshoot; large enough that per-chunk overhead
# (each call re-enters rsl_rl's logging/eval bookkeeping) stays negligible.
TIME_BUDGET_CHUNK_ITERS = 10


def main():
    parser = argparse.ArgumentParser(
        description="Train (or fine-tune) and export one policy — worker process for the control web's Create Policy panel")
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--name', type=str, required=True,
                         help="the policy's display name once loaded into the control web (not the exported filename — see export_policy()'s own naming convention)")
    parser.add_argument('--max_iterations', type=int, default=None,
                         help="stop after this many learning iterations (from wherever --from_checkpoint resumed, if given)")
    parser.add_argument('--max_minutes', type=float, default=None,
                         help="stop after this many minutes of wall-clock training time, regardless of iteration count")
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--cpu', action='store_true', default=True)
    parser.add_argument('--gpu', action='store_true', default=False,
                         help="run Genesis (and rsl_rl's own tensors — see task_registry.py's "
                              "sim_device) on GPU instead of CPU. Overrides --cpu. A separate flag "
                              "rather than making --cpu take on/off, because --cpu is store_true "
                              "with default=True — simply NOT passing it can never mean 'off', so "
                              "there was previously no CLI way at all to request GPU here. Only "
                              "makes sense on a machine that actually has a CUDA GPU (e.g. a Kaggle "
                              "kernel — see kaggle_backend.py's _build_kernel_script).")
    parser.add_argument('--from_checkpoint', type=str, default=None,
                         help="fine-tune from this checkpoint's weights (optimizer state NOT carried over) instead of random init")
    parser.add_argument('--cmd_vx_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--cmd_vy_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--cmd_yaw_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--base_height_target', type=float, default=None,
                         help="pelvis height (m) the height-tracking reward targets — override to match a fine-tuning base's height (e.g. a crouch) instead of the task's own default")
    parser.add_argument('--push_robots', type=str, default=None, choices=['on', 'off'],
                         help="force random push disturbances on/off during training (default: leave the task's own setting)")
    parser.add_argument('--max_push_vel_xy', type=float, default=None,
                         help="peak push velocity (m/s) when --push_robots on")
    parser.add_argument('--push_interval_s', type=float, default=None,
                         help="seconds between pushes when --push_robots on")
    parser.add_argument('--push_dir', type=str, default=None, choices=['behind', 'front', 'left', 'right'],
                         help="bias push direction relative to the robot's heading instead of sampling it "
                              "isotropically (default) — named for where the shove comes from, e.g. 'behind' "
                              "shoves the robot forward")
    parser.add_argument('--reward_scale', type=str, nargs=2, action='append', default=None,
                         metavar=('NAME', 'VALUE'),
                         help="override one reward-term weight, e.g. --reward_scale alive 0.3 — repeatable, "
                              "one flag per term. NAME is any attribute of the task's own "
                              "<Cfg>.rewards.scales (e.g. alive, base_height, tracking_lin_vel, action_rate, "
                              "dof_pos_limits — see the task's *_config.py for the full list and what each "
                              "one actually penalizes/rewards). A positive scale rewards more of that term, "
                              "negative penalizes it; magnitude is relative to the other terms, not absolute — "
                              "there's no fixed 'good' value, only relative-to-what-it-was-before.")
    parser.add_argument('--entropy_coef', type=float, default=None,
                         help="PPO's exploration-noise bonus weight (default: the task's own train_cfg.algorithm.entropy_coef). "
                              "Too high relative to a weak task reward and the action noise std can grow "
                              "instead of shrink over training — an unstable run gets MORE erratic the longer "
                              "it goes, not less (watch 'Mean action noise std' in the log; it should trend "
                              "down, never up). Lower this if that happens.")
    parser.add_argument('--result_path', type=str, required=True,
                         help="JSON {policy_path, task, name, iterations_done, stopped_reason} written here on success — the parent process polls for this file rather than parsing stdout")
    parser.add_argument('--progress_path', type=str, default=None,
                         help="JSON {iterations_done, elapsed_s, updated_at} overwritten after every "
                              "TIME_BUDGET_CHUNK_ITERS-iteration chunk, for the parent process to show live "
                              "progress on a still-running job — same 'poll a file instead of parsing stdout' "
                              "pattern as --result_path, just written mid-run instead of once at the end")
    cli = parser.parse_args()
    if cli.gpu:
        cli.cpu = False

    if cli.max_iterations is None and cli.max_minutes is None:
        parser.error("give at least one of --max_iterations / --max_minutes")

    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level='warning')

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)

    if cli.cmd_vx_range:
        env_cfg.commands.ranges.lin_vel_x = list(cli.cmd_vx_range)
    if cli.cmd_vy_range:
        env_cfg.commands.ranges.lin_vel_y = list(cli.cmd_vy_range)
    if cli.cmd_yaw_range:
        env_cfg.commands.ranges.ang_vel_yaw = list(cli.cmd_yaw_range)
    if cli.base_height_target is not None:
        env_cfg.rewards.base_height_target = cli.base_height_target
    if cli.push_robots is not None:
        env_cfg.domain_rand.push_robots = (cli.push_robots == 'on')
    if cli.max_push_vel_xy is not None:
        env_cfg.domain_rand.max_push_vel_xy = cli.max_push_vel_xy
    if cli.push_interval_s is not None:
        env_cfg.domain_rand.push_interval_s = cli.push_interval_s
    if cli.push_dir is not None:
        env_cfg.domain_rand.push_dir = cli.push_dir
    if cli.reward_scale:
        for name, value in cli.reward_scale:
            if not hasattr(env_cfg.rewards.scales, name):
                parser.error(f"unknown reward scale '{name}' for task '{cli.task}' — "
                              f"see {type(env_cfg.rewards.scales).__name__} for the valid names")
            setattr(env_cfg.rewards.scales, name, float(value))
    if cli.entropy_coef is not None:
        train_cfg.algorithm.entropy_coef = cli.entropy_coef

    args = argparse.Namespace(
        task=cli.task, headless=cli.headless, cpu=cli.cpu, num_envs=cli.num_envs,
        max_iterations=cli.max_iterations, resume=False, sync_wandb=False, export_onnx=False,
        debug=False, load_run=None, ckpt=-1, use_joystick=False, joystick_type='xbox',
        follow_robot=False, viewer='native', viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )

    env, env_cfg = task_registry.make_env(name=cli.task, args=args, env_cfg=env_cfg)
    # train_cfg passed explicitly — without it make_alg_runner() re-fetches
    # a fresh, un-overridden copy by task name, silently dropping
    # --entropy_coef (same mistake env_cfg avoids by being passed to
    # make_env() above).
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cli.task, args=args, train_cfg=train_cfg)

    if cli.from_checkpoint:
        print(f"Fine-tuning from: {cli.from_checkpoint} (optimizer state NOT carried over)")
        ppo_runner.load(cli.from_checkpoint, load_optimizer=False)
        print(f"Resumed at iteration {ppo_runner.current_learning_iteration}")

    log_dir = ppo_runner.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # rsl_rl's OnPolicyRunner.learn() is safe to call repeatedly — it resumes
    # from current_learning_iteration each time (see on_policy_runner.py) and
    # saves a checkpoint at the end of every call — so a --max_minutes budget
    # is enforced by running it in small chunks and checking the wall clock
    # between them, rather than needing a callback rsl_rl doesn't expose.
    start_iteration = ppo_runner.current_learning_iteration
    start_time = time.time()
    deadline = start_time + cli.max_minutes * 60 if cli.max_minutes is not None else None
    iteration_cap = start_iteration + cli.max_iterations if cli.max_iterations is not None else None
    stopped_reason = "max_iterations" if cli.max_minutes is None else "max_minutes"

    def write_progress():
        if not cli.progress_path:
            return
        # Best-effort — a failed write must not crash a training run that's
        # otherwise fine; the parent just won't see this update and tries
        # again next chunk. Same "poll a file, don't lean on it" spirit as
        # TrainingManager.poll()'s own except-and-continue on the read side.
        try:
            with open(cli.progress_path, 'w') as f:
                json.dump({
                    "iterations_done": ppo_runner.current_learning_iteration - start_iteration,
                    "elapsed_s": round(time.time() - start_time, 1),
                    "updated_at": time.time(),
                }, f)
        except OSError:
            pass

    while True:
        if deadline is not None and time.time() >= deadline:
            stopped_reason = "max_minutes"
            break
        if iteration_cap is not None and ppo_runner.current_learning_iteration >= iteration_cap:
            stopped_reason = "max_iterations"
            break
        chunk = TIME_BUDGET_CHUNK_ITERS
        if iteration_cap is not None:
            chunk = min(chunk, iteration_cap - ppo_runner.current_learning_iteration)
        ppo_runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=True)
        write_progress()

    iterations_done = ppo_runner.current_learning_iteration - start_iteration
    print(f"Stopped after {iterations_done} iterations ({stopped_reason}).")

    # Mirrors play.py's own task-type parsing (first underscore-part is the
    # robot name, the rest selects the exporter — see play.py:export_policy).
    task_type = "_".join(cli.task.split("_")[1:])
    export_dir = os.path.join(log_dir, 'exported')
    export_policy(ppo_runner, export_dir, argparse.Namespace(export_onnx=False), env_cfg, train_cfg, task_type)

    # export_policy() names the file after its own convention (e.g. the LSTM
    # exporter always writes 'policy_lstm_1.pt'), not --name — --name is only
    # the display name the web UI will show once this is loaded. export_dir
    # is a freshly created directory, so "most recently written .pt" is
    # unambiguous.
    exported = sorted(
        (os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.pt')),
        key=os.path.getmtime,
    )
    if not exported:
        raise RuntimeError(f"export_policy() reported success but no .pt file was found in {export_dir}")
    policy_path = exported[-1]

    # The raw rsl_rl checkpoint for THIS run — OnPolicyRunner.learn() always
    # ends its final chunk by saving to exactly this path (see
    # on_policy_runner.py's learn(), last line), so unlike
    # TrainingManager._train_checkpoint_from_export()'s after-the-fact
    # directory-walk (for checkpoints this script didn't produce), there's
    # nothing to guess here — we just wrote it. Lets the caller
    # (TrainingManager.finalize_policy()) hand this straight to Clone-from
    # instead of leaving every fresh policy in the same "no raw checkpoint
    # found" limbo external/legacy ones can fall into.
    train_checkpoint_path = os.path.join(log_dir, f'model_{ppo_runner.current_learning_iteration}.pt')
    if not os.path.isfile(train_checkpoint_path):
        train_checkpoint_path = None

    with open(cli.result_path, 'w') as f:
        json.dump({
            "policy_path": policy_path, "task": cli.task, "name": cli.name,
            "iterations_done": iterations_done, "stopped_reason": stopped_reason,
            "train_checkpoint_path": train_checkpoint_path,
        }, f)
    print(f"Done. Exported to {policy_path}")


if __name__ == '__main__':
    main()
