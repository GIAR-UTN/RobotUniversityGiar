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

Two "target" concepts the UI exposes map directly onto flags here:
  - relative target: --from_checkpoint fine-tunes an existing policy's
    weights (same pattern as scripts/finetune_cautious.py) instead of
    training from random init — optimizer state is intentionally NOT
    carried over (see finetune_cautious.py's docstring for why). The new
    policy is defined relative to an old one, and --max_iterations is the
    time budget spent moving from A to B.
  - measurement target: --cmd_*_range overrides the velocity command
    envelope (env_cfg.commands.ranges) the policy is trained across — a
    measured quantity (m/s, rad/s), not a reward-shaping knob.

Usage (as composed by the web UI; also runnable by hand):
    python legged_gym/scripts/web_train.py --task g1_cautious --name my_policy \\
        --max_iterations 500 --num_envs 64 --headless --cpu \\
        --from_checkpoint logs/g1/<run>/model_1800.pt \\
        --cmd_vx_range -0.5 0.5 --result_path /tmp/result.json
"""
import argparse
import json
import os

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.scripts.play import export_policy


def main():
    parser = argparse.ArgumentParser(
        description="Train (or fine-tune) and export one policy — worker process for the control web's Create Policy panel")
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--name', type=str, required=True,
                         help="the policy's display name once loaded into the control web (not the exported filename — see export_policy()'s own naming convention)")
    parser.add_argument('--max_iterations', type=int, required=True)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--cpu', action='store_true', default=True)
    parser.add_argument('--from_checkpoint', type=str, default=None,
                         help="fine-tune from this checkpoint's weights (optimizer state NOT carried over) instead of random init")
    parser.add_argument('--cmd_vx_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--cmd_vy_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--cmd_yaw_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--result_path', type=str, required=True,
                         help="JSON {policy_path, task, name} written here on success — the parent process polls for this file rather than parsing stdout")
    cli = parser.parse_args()

    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level='warning')

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)

    if cli.cmd_vx_range:
        env_cfg.commands.ranges.lin_vel_x = list(cli.cmd_vx_range)
    if cli.cmd_vy_range:
        env_cfg.commands.ranges.lin_vel_y = list(cli.cmd_vy_range)
    if cli.cmd_yaw_range:
        env_cfg.commands.ranges.ang_vel_yaw = list(cli.cmd_yaw_range)

    args = argparse.Namespace(
        task=cli.task, headless=cli.headless, cpu=cli.cpu, num_envs=cli.num_envs,
        max_iterations=cli.max_iterations, resume=False, sync_wandb=False, export_onnx=False,
        debug=False, load_run=None, ckpt=-1, use_joystick=False, joystick_type='xbox',
        follow_robot=False, viewer='native', viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )

    env, env_cfg = task_registry.make_env(name=cli.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cli.task, args=args)

    if cli.from_checkpoint:
        print(f"Fine-tuning from: {cli.from_checkpoint} (optimizer state NOT carried over)")
        ppo_runner.load(cli.from_checkpoint, load_optimizer=False)
        print(f"Resumed at iteration {ppo_runner.current_learning_iteration}, "
              f"training +{cli.max_iterations} more -> target iteration "
              f"{ppo_runner.current_learning_iteration + cli.max_iterations}")

    log_dir = ppo_runner.log_dir
    os.makedirs(log_dir, exist_ok=True)

    ppo_runner.learn(num_learning_iterations=cli.max_iterations, init_at_random_ep_len=True)

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

    with open(cli.result_path, 'w') as f:
        json.dump({"policy_path": policy_path, "task": cli.task, "name": cli.name}, f)
    print(f"Done. Exported to {policy_path}")


if __name__ == '__main__':
    main()
