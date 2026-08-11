"""
Distillation worker launched by legged_gym/control/training.py
(TrainingManager.start_distillation()) as a plain subprocess, one per
"Distill policy" job started from the control web's sidebar — same
subprocess-isolation reasoning as web_train.py's own module docstring (a
live Genesis/task_registry env can't share this process with
rugiar_driver.py's own sim), and the same "poll a result/progress JSON file
instead of parsing stdout" contract.

Behavior-clones a TEACHER checkpoint (any shape legged_gym.control.policy.
load_policy_backend() understands — TorchScript/onnx, stateful/stateless;
crucially, one with NO train_checkpoint.pt at all, e.g. policies/stable.pt)
into a freshly-initialized rsl_rl ActorCritic/ActorCriticRecurrent matching
--task's own architecture, via legged_gym/control/distillation.py's pure BC
algorithm, then exports both a deployable checkpoint (same
legged_gym.scripts.play.export_policy() dispatch web_train.py/fusion.py both
already reuse) and a raw train_checkpoint.pt an rsl_rl OnPolicyRunner can
resume PPO from — exactly the thing an externally-sourced teacher like
`stable` could never produce on its own (see TrainingManager.
_train_checkpoint_from_export()'s docstring in training.py for why).

Usage (as composed by TrainingManager.start_distillation(); also runnable by
hand):
    python legged_gym/scripts/web_distill.py --task g1 --name stable_distilled \\
        --teacher_checkpoint policies/stable/checkpoint.pt \\
        --rollout_steps 4000 --bc_epochs 20 --lr 1e-3 --num_envs 64 --headless --cpu \\
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
from legged_gym.control import distillation
from legged_gym.control.fusion import export_actor_critic
from legged_gym.control.policy import load_policy_backend


def main():
    parser = argparse.ArgumentParser(
        description="Behavior-clone a teacher policy into a fresh, fine-tunable checkpoint "
                    "— worker process for the control web's Distill policy panel")
    parser.add_argument('--task', type=str, required=True,
                         help="target task — determines the student's own obs/action/architecture")
    parser.add_argument('--name', type=str, required=True,
                         help="the resulting policy's display name once loaded into the control web")
    parser.add_argument('--teacher_checkpoint', type=str, required=True,
                         help="path to the teacher's deployable checkpoint (.pt/.onnx) — "
                              "loaded via legged_gym.control.policy.load_policy_backend(), "
                              "so a train_checkpoint.pt is NOT required for the teacher")
    parser.add_argument('--method', type=str, default='behavior_cloning',
                         choices=list(distillation.DISTILL_METHODS.keys()),
                         help="distillation method (see distillation.DISTILL_METHODS)")
    parser.add_argument('--rollout_steps', type=int, default=4000,
                         help="env ticks to roll the teacher for while collecting (obs, action) pairs")
    parser.add_argument('--bc_epochs', type=int, default=20,
                         help="supervised training epochs over the collected rollout")
    parser.add_argument('--lr', type=float, default=1e-3, help="Adam learning rate for BC training")
    parser.add_argument('--num_envs', type=int, default=1,
                         help="parallel envs for rollout collection. Defaults to 1 (not 64, unlike "
                              "training) because some externally-sourced teachers (e.g. unitree_rl_gym's "
                              "own exports, which bake a FIXED batch=1 into the exported TorchScript "
                              "module's internal hidden-state buffers — see legged_gym.control.policy."
                              "InternalStatePolicy) can only ever be stepped one env at a time; a "
                              "locally-trained teacher (this repo's own ExplicitStatePolicy export "
                              "convention) has no such limit and can safely use a higher value for a "
                              "faster rollout.")
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--cpu', action='store_true', default=True)
    parser.add_argument('--gpu', action='store_true', default=False,
                         help="run Genesis/rsl_rl on GPU instead of CPU — same flag semantics as web_train.py's own --gpu")
    parser.add_argument('--result_path', type=str, required=True,
                         help="JSON {policy_path, train_checkpoint_path, task, name, iterations_done, "
                              "final_bc_loss} written here on success")
    parser.add_argument('--progress_path', type=str, default=None,
                         help="JSON {phase, iterations_done, elapsed_s, updated_at} overwritten "
                              "periodically — same 'poll a file' contract as web_train.py's own --progress_path")
    cli = parser.parse_args()
    if cli.gpu:
        cli.cpu = False

    if distillation.DISTILL_METHODS[cli.method]["available"] is False:
        parser.error(f"distillation method '{cli.method}' is planned but not yet implemented")

    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level='warning')

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)

    args = argparse.Namespace(
        task=cli.task, headless=cli.headless, cpu=cli.cpu, num_envs=cli.num_envs,
        max_iterations=None, resume=False, sync_wandb=False, export_onnx=False,
        debug=False, load_run=None, ckpt=-1, use_joystick=False, joystick_type='xbox',
        follow_robot=False, viewer='native', viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )
    env, env_cfg = task_registry.make_env(name=cli.task, args=args, env_cfg=env_cfg)

    device = "cuda" if cli.gpu else "cpu"
    hidden_size = getattr(train_cfg.policy, "rnn_hidden_size", 64)
    teacher_backend = load_policy_backend(
        cli.teacher_checkpoint, hidden_size=hidden_size, num_envs=cli.num_envs, device=device)

    # Fail fast with a clear message rather than let a shape mismatch produce
    # silently-garbage rollout data — same "reject up front" precedent
    # fuse_policies' export_task hard-check sets (training.py:739-745).
    distillation.check_dimensions_compatible(
        teacher_backend, num_obs=env.num_obs, num_actions=env.num_actions, num_envs=cli.num_envs, device=device)

    start_time = time.time()

    def write_progress(phase: str, done: int) -> None:
        if not cli.progress_path:
            return
        try:
            with open(cli.progress_path, 'w') as f:
                json.dump({
                    # iterations_done/elapsed_s are read by TrainingManager._refresh_progress()
                    # into job.iterations_done, the SAME field renderTrainingJobs()'s jobProgress()
                    # already uses against job.max_iterations (== cli.bc_epochs, see
                    # TrainingManager.start_distillation()) to compute a % and ETA — kept in
                    # [0, bc_epochs] throughout (0 during the rollout phase, which has no
                    # bc_epochs-scale equivalent of its own) so that shared math stays correct
                    # rather than briefly overshooting 100% while collecting a long rollout.
                    "phase": phase, "iterations_done": done,
                    "elapsed_s": round(time.time() - start_time, 1), "updated_at": time.time(),
                }, f)
        except OSError:
            pass  # best-effort — see web_train.py's identical write_progress() docstring

    print(f"Rolling out teacher for {cli.rollout_steps} steps ({cli.num_envs} envs)...")
    obs_buf, action_buf, dones_buf = distillation.collect_rollout(
        env, teacher_backend, cli.rollout_steps,
        callback=lambda step, total: write_progress("rollout", 0))

    is_recurrent = train_cfg.runner.policy_class_name == "ActorCriticRecurrent"
    num_critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
    student = distillation.build_student(
        env.num_obs, num_critic_obs, env.num_actions, train_cfg.policy, is_recurrent)

    print(f"Behavior-cloning student for {cli.bc_epochs} epochs...")
    final_loss = distillation.bc_train(
        student, obs_buf, action_buf, epochs=cli.bc_epochs, lr=cli.lr, dones_buf=dones_buf,
        callback=lambda epoch, loss: (
            write_progress("bc_train", epoch + 1),
            print(f"epoch {epoch + 1}/{cli.bc_epochs}: mse_loss={loss:.6f}"),
        ))

    # Mirrors web_train.py's own task_type parsing (first underscore-part is
    # the robot name, the rest selects the exporter — see play.py:export_policy).
    task_type = "_".join(cli.task.split("_")[1:])
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', cli.task, f'{time.strftime("%b%d_%H-%M-%S")}_distill_{cli.name}')
    export_dir = os.path.join(log_dir, 'exported')
    os.makedirs(export_dir, exist_ok=True)
    export_actor_critic(student, export_dir, env_cfg, train_cfg, task_type)

    exported = sorted(
        (os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.pt')),
        key=os.path.getmtime,
    )
    if not exported:
        raise RuntimeError(f"export_policy() reported success but no .pt file was found in {export_dir}")
    policy_path = exported[-1]

    # Raw rsl_rl checkpoint — same shape TrainingManager.fuse_policies() writes
    # (training.py:772-774) and OnPolicyRunner.load() reads (model_state_dict/
    # optimizer_state_dict/iter/infos), so `rugiar train --from_policy <name>`
    # can resume PPO from this student exactly like any other checkpoint.
    import torch
    train_checkpoint_path = os.path.join(log_dir, 'model_bc.pt')
    torch.save(
        {"model_state_dict": student.state_dict(), "optimizer_state_dict": {}, "iter": 0, "infos": {}},
        train_checkpoint_path)

    with open(cli.result_path, 'w') as f:
        json.dump({
            "policy_path": policy_path, "task": cli.task, "name": cli.name,
            "iterations_done": cli.bc_epochs, "train_checkpoint_path": train_checkpoint_path,
            "final_bc_loss": final_loss, "method": cli.method,
            "rollout_steps": cli.rollout_steps, "bc_epochs": cli.bc_epochs,
            "teacher_checkpoint": cli.teacher_checkpoint,
        }, f)
    print(f"Done. final_bc_loss={final_loss:.6f}. Exported to {policy_path}")


if __name__ == '__main__':
    main()
