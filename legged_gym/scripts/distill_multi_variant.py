"""
Parallel-variant distillation — behavior-clones a teacher (e.g. `stable`)
against SEVERAL structurally-biased command-range rollouts collected in
parallel (one OS process per variant, via `collect_rollout_variant.py`),
concatenates every variant's (obs, action, dones) into one combined dataset,
then runs a single bc_train() over the union.

Why this exists (see collect_rollout_variant.py's docstring for the full
reasoning): `rugiar distill`'s single `--num_envs 1` rollout only draws a
handful of independent commands over its whole duration (resampling_time=10s
means a 4000-step/80s rollout gets ~8 draws), so the BC dataset can miss
whole regions of the teacher's command envelope. The community reference
(unitree_rl_gym/legged_gym) gets broad coverage for free from num_envs=4096
parallel envs; a batch-locked externally-sourced teacher like `stable` can't
use that. This script's stand-in: force each of N parallel single-env
rollouts into a different deliberate corner of command space (forward-only,
backward-only, turn-left-only, turn-right-only, strafe-left, strafe-right,
plus one unbiased full-range control), instead of hoping ~8 random draws
happen to cover them all.

Usage:
    export SIMULATOR=genesis
    .venv/bin/python legged_gym/scripts/distill_multi_variant.py \\
        --teacher stable --task g1 --name stable_distilled_multivariant \\
        --rollout_steps 4000 --bc_epochs 20

Registers the result as a normal ./policies/<name>/ folder via the same
TrainingManager.finalize_policy() every other distillation path uses —
fine-tunable via `rugiar train --from_policy` exactly like any other clone.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from legged_gym import *          # noqa: F401,F403 — registers tasks into task_registry
from legged_gym.envs import *     # noqa: F401,F403 — this process only reads cfgs (no gs.init here)
from legged_gym.utils import task_registry
from legged_gym.control import distillation

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = REPO_ROOT / "legged_gym" / "scripts" / "collect_rollout_variant.py"

# (label, cmd_vx_range, cmd_vy_range, cmd_yaw_range) — None means "leave the
# task's own default range" (used by full_range, the unbiased control group).
DEFAULT_VARIANTS = [
    ("forward",       (0.7, 1.0),   (-0.2, 0.2),  (-0.3, 0.3)),
    ("backward",      (-1.0, -0.7), (-0.2, 0.2),  (-0.3, 0.3)),
    ("turn_left",     (-0.3, 0.3),  (-0.2, 0.2),  (0.7, 1.0)),
    ("turn_right",    (-0.3, 0.3),  (-0.2, 0.2),  (-1.0, -0.7)),
    ("strafe_left",   (-0.3, 0.3),  (0.6, 1.0),   (-0.3, 0.3)),
    ("strafe_right",  (-0.3, 0.3),  (-1.0, -0.6), (-0.3, 0.3)),
    ("full_range",    None,         None,         None),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--teacher', type=str, required=True, help="local policy name to distill from")
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--name', type=str, required=True, help="output policy name")
    parser.add_argument('--rollout_steps', type=int, default=4000,
                         help="steps PER VARIANT (each is its own single-env rollout, not shared)")
    parser.add_argument('--bc_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--variants', type=str, default=None,
                         help="comma-separated subset of variant labels to run "
                              "(default: all of " + ",".join(v[0] for v in DEFAULT_VARIANTS) + ")")
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser


def main():
    cli = build_parser().parse_args()

    from legged_gym.control.training import POLICIES_DIR, TrainingManager, TrainingJob

    teacher_dir = POLICIES_DIR / cli.teacher
    teacher_checkpoint = teacher_dir / "checkpoint.pt"
    if not teacher_checkpoint.is_file():
        sys.exit(f"'{cli.teacher}' has no checkpoint.pt at {teacher_checkpoint}")
    if (POLICIES_DIR / cli.name).exists():
        sys.exit(f"'{cli.name}' already exists — pick a different --name")

    variants = DEFAULT_VARIANTS
    if cli.variants:
        wanted = set(cli.variants.split(","))
        variants = [v for v in DEFAULT_VARIANTS if v[0] in wanted]
        missing = wanted - {v[0] for v in variants}
        if missing:
            sys.exit(f"unknown variant label(s): {missing}")

    started_at = time.time()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"distill_multivariant_{cli.name}_"))
    print(f"[distill_multi_variant] {len(variants)} variants, {cli.rollout_steps} steps each, "
          f"scratch dir {tmp_dir}")

    # ---- launch every variant's rollout collection as its own parallel OS process ----
    procs = []
    for i, (label, vx, vy, yaw) in enumerate(variants):
        out_path = tmp_dir / f"{label}.pt"
        log_path = tmp_dir / f"{label}.log"
        argv = [
            sys.executable, "-u", str(WORKER_SCRIPT),
            "--task", cli.task, "--teacher_checkpoint", str(teacher_checkpoint),
            "--rollout_steps", str(cli.rollout_steps), "--num_envs", "1",
            "--seed", str(100 + i), "--label", label, "--out_path", str(out_path),
        ]
        if vx: argv += ["--cmd_vx_range", str(vx[0]), str(vx[1])]
        if vy: argv += ["--cmd_vy_range", str(vy[0]), str(vy[1])]
        if yaw: argv += ["--cmd_yaw_range", str(yaw[0]), str(yaw[1])]
        if label != "full_range":
            # biased variants: hold the deliberate corner for real — otherwise zero_cmd_prob's
            # default 40% zeroes the command regardless of our override, and heading_command's
            # default self-correcting yaw controller undoes a forced turn (see
            # collect_rollout_variant.py's --zero_cmd_prob/--heading_command docstrings).
            # full_range is the unbiased control group, left at the task's own defaults on
            # purpose — it's the one variant meant to look like an ordinary training rollout.
            argv += ["--zero_cmd_prob", "0", "--heading_command", "off"]
        if cli.gpu: argv += ["--gpu"]
        log_f = open(log_path, "w")
        proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((label, proc, log_f, out_path, log_path))
        print(f"[distill_multi_variant] launched '{label}' (pid {proc.pid})")

    failed = []
    for label, proc, log_f, out_path, log_path in procs:
        rc = proc.wait()
        log_f.close()
        if rc != 0:
            failed.append((label, log_path))
        else:
            print(f"[distill_multi_variant] '{label}' done (rc=0)")
    if failed:
        for label, log_path in failed:
            print(f"[distill_multi_variant] '{label}' FAILED — see {log_path}", file=sys.stderr)
        sys.exit(f"{len(failed)}/{len(variants)} variant(s) failed, aborting before bc_train")

    # ---- merge every variant's rollout into one dataset ----
    dumps = [torch.load(tmp_dir / f"{label}.pt", weights_only=False) for label, *_ in variants]
    obs_buf = torch.cat([d["obs"] for d in dumps], dim=1)      # (steps, sum(envs)=len(variants), dim)
    action_buf = torch.cat([d["action"] for d in dumps], dim=1)
    dones_buf = torch.cat([d["dones"] for d in dumps], dim=1)
    per_variant_coverage = {d["label"]: d["coverage"] for d in dumps}
    merged_ground_truth = {
        key: torch.cat([d["ground_truth"][key] for d in dumps], dim=1)
        for key in ("commands", "base_lin_vel", "base_ang_vel")
    }
    print(f"[distill_multi_variant] merged dataset shape: obs={tuple(obs_buf.shape)}, "
          f"action={tuple(action_buf.shape)}")

    # ---- diagnostic: merged coverage, straight off raw simulator state each variant
    # recorded (see distillation.collect_rollout()'s docstring for why this reads
    # ground_truth instead of decoding obs_buf by column slice — task-specific
    # compute_observations() overrides make a fixed slice unreliable across tasks).
    merged_coverage = distillation.summarize_rollout(merged_ground_truth, action_buf)
    print("[distill_multi_variant] merged coverage (physical units):")
    for key, s in merged_coverage.items():
        if isinstance(s, dict):
            print(f"  {key}: mean={s['mean']:.4f} std={s['std']:.4f} range=[{s['min']:.4f}, {s['max']:.4f}]")
        else:
            print(f"  {key}: {s:.4f}")

    # ---- one bc_train() pass over the combined, deliberately-diverse dataset ----
    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)
    is_recurrent = train_cfg.runner.policy_class_name == "ActorCriticRecurrent"
    num_obs = obs_buf.shape[-1]
    num_actions = action_buf.shape[-1]
    student = distillation.build_student(num_obs, num_obs, num_actions, train_cfg.policy, is_recurrent)

    print(f"[distill_multi_variant] behavior-cloning student for {cli.bc_epochs} epochs "
          f"over {obs_buf.shape[1]} merged rollouts...")
    final_loss = distillation.bc_train(
        student, obs_buf, action_buf, epochs=cli.bc_epochs, lr=cli.lr, dones_buf=dones_buf,
        callback=lambda epoch, loss: print(f"epoch {epoch + 1}/{cli.bc_epochs}: mse_loss={loss:.6f}"))
    print(f"[distill_multi_variant] final_bc_loss={final_loss:.6f}")

    # ---- export + register exactly like web_distill.py's own tail end ----
    from legged_gym.scripts.play import export_policy  # noqa: F401 (dispatch parity, see web_distill.py)
    from legged_gym.control.fusion import export_actor_critic

    task_type = "_".join(cli.task.split("_")[1:])
    log_dir = REPO_ROOT / "logs" / cli.task / f'{time.strftime("%b%d_%H-%M-%S")}_distill_multivariant_{cli.name}'
    export_dir = log_dir / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_actor_critic(student, str(export_dir), env_cfg, train_cfg, task_type)
    exported = sorted(export_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not exported:
        sys.exit(f"export_policy() reported success but no .pt file was found in {export_dir}")
    policy_path = str(exported[-1])

    train_checkpoint_path = str(log_dir / "model_bc.pt")
    torch.save(
        {"model_state_dict": student.state_dict(), "optimizer_state_dict": {}, "iter": 0, "infos": {}},
        train_checkpoint_path)

    finished_at = time.time()
    combined_log_path = log_dir / "train.log"
    with open(combined_log_path, "w") as f:
        f.write(f"distill_multi_variant — teacher={cli.teacher} task={cli.task} name={cli.name}\n")
        f.write(f"variants: {[v[0] for v in variants]}\n")
        f.write(f"rollout_steps per variant: {cli.rollout_steps}, bc_epochs: {cli.bc_epochs}, lr: {cli.lr}\n")
        f.write(f"final_bc_loss: {final_loss:.6f}\n\n")
        f.write("per-variant coverage:\n" + json.dumps(per_variant_coverage, indent=2) + "\n\n")
        f.write("merged coverage:\n" + json.dumps(merged_coverage, indent=2) + "\n")

    job = TrainingJob(
        id="multivariant", policy_name=cli.name, task=cli.task,
        command=(f"distill_multi_variant --teacher {cli.teacher} --task {cli.task} --name {cli.name} "
                 f"--rollout_steps {cli.rollout_steps} --bc_epochs {cli.bc_epochs} --lr {cli.lr} "
                 f"--variants {','.join(v[0] for v in variants)}"),
        log_path=str(combined_log_path), result_path="", progress_path="",
        started_at=started_at, finished_at=finished_at,
        max_iterations=cli.bc_epochs, max_minutes=None, num_envs=len(variants),
        job_type="distill", teacher_policy=cli.teacher, simulator="genesis",
        final_bc_loss=final_loss, rollout_diagnostics=merged_coverage,
    )
    mgr = TrainingManager()
    final_checkpoint = mgr.finalize_policy(
        cli.name, task=cli.task, checkpoint=policy_path,
        train_checkpoint=train_checkpoint_path, job=job)
    print(f"\n[distill_multi_variant] '{cli.name}' ready — distilled from '{cli.teacher}' across "
          f"{len(variants)} variants, final_bc_loss={final_loss:.6f}, checkpoint at {final_checkpoint}")


if __name__ == '__main__':
    main()
