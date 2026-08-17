"""
One-variant rollout collector for `legged_gym/scripts/distill_multi_variant.py`'s
parallel-variant distillation — NOT a general-purpose CLI (not wired into
`rugiar`/the control web), a worker this orchestrator launches once per variant
as its own subprocess and waits on.

Why this exists: `rugiar distill`'s single rollout (`legged_gym/control/
distillation.collect_rollout()`) is one `--num_envs 1` trajectory under whatever
commands the env's own resampler happens to draw — with `commands.resampling_time
= 10s` (see legged_robot_config.py), a 4000-step/80s rollout only draws ~8
independent commands total, so the BC dataset can easily miss whole regions of
the teacher's command envelope (confirmed here: a real run's commanded_lin_vel_x
never exceeded +0.41 m/s). The community reference (unitree_rl_gym / ETH
legged_gym) gets coverage for free from num_envs=4096 parallel envs each
resampling independently; `stable`-style externally-sourced teachers can't use
that (see policy.py's InternalStatePolicy — hidden-state buffers are batch-locked
to whatever the export used, normally 1). This script's answer: force ONE
biased corner of command space per process (via --cmd_*_range, the exact same
env_cfg.commands.ranges override `rugiar train` already exposes), run N of them
as separate OS processes in parallel (cheap — CPU-only, single-env each), and
let the orchestrator concatenate the results into one bigger, deliberately
diverse dataset before a single bc_train() pass. A structured stand-in for
mass parallel randomized coverage, not a replacement for it.

Dumps {obs, action, dones, label, cmd_ranges} to --out_path as a torch .pt file.
"""
import argparse
import json

import torch

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.control import distillation
from legged_gym.control.policy import load_policy_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--rollout_steps', type=int, required=True)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--seed', type=int, required=True,
                         help="distinct per variant so each process's random draws "
                              "(initial state, in-range command sampling) don't repeat")
    parser.add_argument('--label', type=str, required=True, help="variant name, for logging")
    parser.add_argument('--cmd_vx_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--cmd_vy_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--cmd_yaw_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--zero_cmd_prob', type=float, default=None,
                         help="override cfg.commands.zero_cmd_prob (default 0.4 — a 40%% chance ANY "
                              "resample gets forced to zero regardless of --cmd_*_range, 'to encourage "
                              "standing still'; set 0 for a variant meant to deliberately stay biased)")
    parser.add_argument('--heading_command', type=str, default=None, choices=['on', 'off'],
                         help="override cfg.commands.heading_command (default True — when on, the yaw "
                              "command is NOT held at its resampled value: it's recomputed every step as "
                              "proportional control toward a target heading, self-correcting toward 0 once "
                              "reached, so --cmd_yaw_range only bounds the clip, it doesn't force a "
                              "sustained turn. Set 'off' for a variant meant to hold a real turn command.)")
    parser.add_argument('--cpu', action='store_true', default=True)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--out_path', type=str, required=True,
                         help="where to torch.save({obs, action, dones, label, coverage}, ...)")
    cli = parser.parse_args()
    if cli.gpu:
        cli.cpu = False

    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level='warning')

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)
    env_cfg.seed = cli.seed
    if cli.cmd_vx_range:
        env_cfg.commands.ranges.lin_vel_x = list(cli.cmd_vx_range)
    if cli.cmd_vy_range:
        env_cfg.commands.ranges.lin_vel_y = list(cli.cmd_vy_range)
    if cli.cmd_yaw_range:
        env_cfg.commands.ranges.ang_vel_yaw = list(cli.cmd_yaw_range)
    if cli.zero_cmd_prob is not None:
        env_cfg.commands.zero_cmd_prob = cli.zero_cmd_prob
    if cli.heading_command is not None:
        env_cfg.commands.heading_command = (cli.heading_command == 'on')

    args = argparse.Namespace(
        task=cli.task, headless=True, cpu=cli.cpu, num_envs=cli.num_envs,
        max_iterations=None, resume=False, sync_wandb=False, export_onnx=False,
        debug=False, load_run=None, ckpt=-1, use_joystick=False, joystick_type='xbox',
        follow_robot=False, viewer='native', viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )
    env, env_cfg = task_registry.make_env(name=cli.task, args=args, env_cfg=env_cfg)

    # Force an immediate command resample: cfg.commands.resampling_time=10s (500 control
    # steps) means the env's own periodic resample never fires within a shorter rollout
    # unless the robot happens to fall and get reset first — leaving `commands` stuck at
    # its post-construction zero state the whole time (confirmed the hard way: a 200-step
    # smoke test reported EXACTLY zero commanded_lin_vel_x/_y for every variant). This makes
    # the --cmd_*_range bias for this variant take effect from step 0 instead of maybe never.
    env._resample_commands(torch.arange(env.num_envs, device=env.device))

    device = "cuda" if cli.gpu else "cpu"
    hidden_size = getattr(train_cfg.policy, "rnn_hidden_size", 64)
    teacher_backend = load_policy_backend(
        cli.teacher_checkpoint, hidden_size=hidden_size, num_envs=cli.num_envs, device=device)
    distillation.check_dimensions_compatible(
        teacher_backend, num_obs=env.num_obs, num_actions=env.num_actions, num_envs=cli.num_envs, device=device)

    print(f"[{cli.label}] rolling out {cli.rollout_steps} steps "
          f"(cmd_vx={cli.cmd_vx_range}, cmd_vy={cli.cmd_vy_range}, cmd_yaw={cli.cmd_yaw_range})...")
    obs_buf, action_buf, dones_buf, ground_truth = distillation.collect_rollout(env, teacher_backend, cli.rollout_steps)
    coverage = distillation.summarize_rollout(ground_truth, action_buf)
    print(f"[{cli.label}] coverage: " + json.dumps(coverage))

    torch.save({
        "obs": obs_buf, "action": action_buf, "dones": dones_buf, "ground_truth": ground_truth,
        "label": cli.label, "coverage": coverage,
        "cmd_ranges": {"vx": cli.cmd_vx_range, "vy": cli.cmd_vy_range, "yaw": cli.cmd_yaw_range},
    }, cli.out_path)
    print(f"[{cli.label}] done -> {cli.out_path}")


if __name__ == '__main__':
    main()
