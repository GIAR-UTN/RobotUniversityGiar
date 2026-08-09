"""
Fine-tune a registered task FROM an already-trained checkpoint, instead of from random init.
Unlike plain train.py --resume (which only resumes within the SAME task's log root and
config), this loads a checkpoint from any path — same task (continuing under a revised
reward) or a different one (e.g. a go2 checkpoint used to seed a related go2 variant task)
— by calling ppo_runner.load() directly with an explicit path.

The optimizer state is intentionally NOT loaded (load_optimizer=False): the source
checkpoint's Adam momentum was accumulated under a different reward function (or a
revised one), and carrying it over would bias early fine-tuning steps toward the old
objective instead of adapting cleanly to the new one.

A pure reward-WEIGHT variant (same reward terms, different numbers — no new task needed)
doesn't need this script at all: use the Create Policy web panel's clone-from +
reward-scale overrides instead (see HANDOFF_task_reward_harmony.md). This script is for
resuming training under a task that's structurally different from the source checkpoint's.

Usage:
    python legged_gym/scripts/finetune_from_checkpoint.py --task g1 \
        --from_checkpoint logs/g1/<run>/model_1000.pt \
        --max_iterations 1500 --headless --cpu --num_envs=64
"""
import argparse
import os

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a registered task from an existing checkpoint")
    parser.add_argument('--task', type=str, required=True,
                         help="registered task to train under (e.g. g1, go2)")
    parser.add_argument('--from_checkpoint', type=str, required=True,
                         help="path to the checkpoint to start from")
    parser.add_argument('--max_iterations', type=int, default=1000,
                         help="additional iterations to train under this task's reward")
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--cpu', action='store_true', default=True)
    parser.add_argument('--num_envs', type=int, default=64)
    cli = parser.parse_args()

    args = argparse.Namespace(
        task=cli.task, headless=cli.headless, cpu=cli.cpu, num_envs=cli.num_envs,
        max_iterations=cli.max_iterations, resume=False, sync_wandb=False, export_onnx=False,
        debug=False, load_run=None, ckpt=-1, use_joystick=False, joystick_type='xbox',
        follow_robot=False, viewer='native', viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )

    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level='warning')

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    print(f"Loading base weights from: {cli.from_checkpoint} (optimizer state NOT carried over)")
    ppo_runner.load(cli.from_checkpoint, load_optimizer=False)
    print(f"Resumed at iteration {ppo_runner.current_learning_iteration}, "
          f"fine-tuning +{cli.max_iterations} more under {cli.task}'s reward "
          f"-> target iteration {ppo_runner.current_learning_iteration + cli.max_iterations}")

    log_dir = ppo_runner.log_dir
    os.makedirs(log_dir, exist_ok=True)

    ppo_runner.learn(num_learning_iterations=cli.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()
