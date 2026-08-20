"""
Training worker for **mjlab** tasks — the mjlab sibling of
legged_gym/scripts/web_train.py, launched by legged_gym/control/training.py
(TrainingManager.start()) as a subprocess under `.venv-mjlab`, one per
"Create Policy" job whose task lives in mjlab's registry (e.g.
Rugiar-G1-Mimic). Item 4 of HANDOFF_mimic_motion_library_ux.md; the frozen
design it implements is docs/mjlab_training_contract.md.

This is a real PORT of mjlab's own `mjlab/scripts/train.py::run_train()` —
the same relationship legged_gym/scripts/process_reference_motion_mjlab.py
has to its upstream converter — not a subprocess wrapper around mjlab's
tyro CLI. What's dropped from upstream: tyro (this repo's jobs are composed
by TrainingManager as plain argv, and the shared flag vocabulary with
web_train.py is the point), wandb (see --result_path/--progress_path below:
the parent process polls files, and forcing the tensorboard logger is what
keeps a job from needing a wandb login — see agent_cfg.logger below),
torchrunx multi-GPU, and video recording. What's added on top of upstream:
the same budget/chunking, progress-file and result-file contract
web_train.py already has with TrainingManager, plus a stateless ONNX export.

Three things here are deliberately NOT copied from web_train.py, because
mjlab's config model differs (see docs/mjlab_training_contract.md §6, §7, §4):

  - reward overrides go through `env_cfg.rewards[<term>].weight` (mjlab's
    rewards are a plain dict of RewardTermCfg) rather than Genesis's
    `env_cfg.rewards.scales.<term>` class attributes;
  - the iteration budget is counted by OUR OWN counter, never by deltas of
    `runner.current_learning_iteration` — rsl-rl 5.x sets that to the LAST
    iteration index, not last+1, so it advances by chunk-1 per resumed
    chunk (web_train.py's delta arithmetic would silently under-count here);
  - the exported policy is written by MjlabOnPolicyRunner's OWN
    export_policy_to_onnx, called explicitly on the parent class, to get a
    1-input/1-output (obs -> actions) graph. The tracking runner's
    auto-export is a 2-input/7-output graph that load_onnx_backend() would
    route to OnnxExplicitStatePolicy and drive with the wrong tensor fed
    back as `time_step` — silently wrong, no crash. See §4.

Usage (as composed by TrainingManager; also runnable by hand):
    CUDA_VISIBLE_DEVICES="" SIMULATOR=mjlab .venv-mjlab/bin/python \\
        legged_gym/scripts/mjlab_train.py --task Rugiar-G1-Mimic \\
        --name my_mimic_policy --num_envs 64 --max_iterations 500 \\
        --motion_file resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz \\
        --reward_scale motion_body_pos 2.0 \\
        --result_path /tmp/result.json --progress_path /tmp/progress.json
"""
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])

# R1 (docs/mjlab_migration.md): the repo root has to be ON sys.path (that's
# the only place `mjlab_tasks` and `legged_gym` live) but LAST, so PyPI
# rsl-rl-lib in .venv-mjlab/site-packages wins over this repo's vendored
# top-level rsl_rl/ package. Same reorder rugiar_driver_mjlab.py and
# tests/conftest.py apply. Must run before any mjlab/rsl_rl/legged_gym
# import — and it is also what makes an inherited PYTHONPATH=<repo root>
# harmless, since that entry gets filtered and re-appended last.
sys.path = [p for p in sys.path if p not in ("", ".", REPO_ROOT)] + [REPO_ROOT]

# legged_gym/__init__.py refuses to import without this, and 'mjlab' is the
# value that makes it skip the Genesis/Isaac import entirely (there is no
# Genesis in this venv, by design). setdefault, not assignment, so an
# explicitly-set value still surfaces as the error it should be.
os.environ.setdefault("SIMULATOR", "mjlab")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402

import mjlab_tasks  # noqa: F401,E402 - import side effect: registers Rugiar-G1-Mimic
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks import registry  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

# Iterations run per learn() call before re-checking the wall-clock deadline —
# same constant and same rationale as web_train.py:59 (small enough that a
# --max_minutes budget is honored within roughly this much overshoot, large
# enough that per-chunk overhead stays negligible).
TIME_BUDGET_CHUNK_ITERS = 10

# web_train.py flags with no mjlab analogue at all. Declared on the parser
# (so a stale caller gets a real explanation instead of argparse's bare
# "unrecognized arguments") and then rejected in one message listing
# everything that was supplied.
INAPPLICABLE = (
    "cmd_vx_range", "cmd_vy_range", "cmd_yaw_range", "base_height_target",
    "lin_vel_z_target", "ang_vel_xy_target", "orientation_tilt_target",
    "push_robots", "max_push_vel_xy", "push_interval_s", "push_dir",
)


def build_parser() -> argparse.ArgumentParser:
    """Split out from main() so the CLI surface itself (which flags exist,
    which are rejected) is unit-testable without constructing an mjlab env."""
    parser = argparse.ArgumentParser(
        description="Train (or fine-tune) and export one mjlab policy — worker process "
                    "for the control web's Create Policy panel (mjlab tasks only)")
    parser.add_argument('--task', type=str, required=True,
                        help="mjlab registry task id (e.g. Rugiar-G1-Mimic)")
    parser.add_argument('--name', type=str, required=True,
                        help="the policy's display name once loaded into the control web; "
                             "also agent_cfg.run_name and the log-dir suffix")
    parser.add_argument('--max_iterations', type=int, default=None,
                        help="stop after this many learning iterations RUN BY THIS JOB "
                             "(not an absolute rsl-rl iteration index — see the budget loop)")
    parser.add_argument('--max_minutes', type=float, default=None,
                        help="stop after this many minutes of wall-clock training time")
    parser.add_argument('--num_envs', type=int, default=64,
                        help="env_cfg.scene.num_envs (the registered default is 1)")
    parser.add_argument('--motion_file', type=str, default=None,
                        help="reference-motion .npz to train against — repo-root-relative or "
                             "absolute. REQUIRED for a tracking task (its registered command "
                             "term has no default clip)")
    parser.add_argument('--device', type=str, default='cpu',
                        help="torch/mujoco device for both the env and the runner ('cpu', 'cuda:0')")
    parser.add_argument('--from_checkpoint', type=str, default=None,
                        help="fine-tune from this checkpoint's weights (optimizer state NOT "
                             "carried over, and the iteration counter is NOT resumed — this "
                             "job's budget always starts at zero)")
    parser.add_argument('--reward_scale', type=str, nargs=2, action='append', default=None,
                        metavar=('NAME', 'VALUE'),
                        help="override one reward-term weight, e.g. --reward_scale motion_body_pos 2.0 "
                             "— repeatable. NAME is any key of this task's env_cfg.rewards dict "
                             "(for Rugiar-G1-Mimic: motion_global_root_pos, motion_global_root_ori, "
                             "motion_body_pos, motion_body_ori, motion_body_lin_vel, "
                             "motion_body_ang_vel, action_rate_l2, joint_limit, self_collisions). "
                             "Positive rewards, negative penalizes; setting exactly 0.0 makes "
                             "RewardManager skip the term entirely (its chart series stays, at 0)")
    parser.add_argument('--entropy_coef', type=float, default=None,
                        help="PPO's exploration-noise bonus weight (default: the task's own "
                             "agent_cfg.algorithm.entropy_coef)")
    parser.add_argument('--seed', type=int, default=None,
                        help="RNG seed for both env and agent (default: the task's own agent_cfg.seed)")
    parser.add_argument('--headless', action='store_true', default=True,
                        help="accepted and IGNORED — training never opens a viewer. Exists only so "
                             "the shared flag list on the TrainingManager side needn't special-case it")
    parser.add_argument('--result_path', type=str, required=True,
                        help="JSON {policy_path, task, name, iterations_done, stopped_reason, "
                             "train_checkpoint_path, ...} written here on success — the parent "
                             "process polls for this file rather than parsing stdout")
    parser.add_argument('--progress_path', type=str, default=None,
                        help="JSON {iterations_done, elapsed_s, updated_at, ...} overwritten once "
                             "per learning iteration, for the parent to show live progress")

    # --- accepted by argparse purely so they can be rejected with a real message ---
    for flag in ('--cmd_vx_range', '--cmd_vy_range', '--cmd_yaw_range'):
        parser.add_argument(flag, type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                            help=argparse.SUPPRESS)
    for flag in ('--base_height_target', '--lin_vel_z_target', '--ang_vel_xy_target',
                 '--orientation_tilt_target', '--max_push_vel_xy', '--push_interval_s'):
        parser.add_argument(flag, type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--push_robots', type=str, default=None, choices=['on', 'off'],
                        help=argparse.SUPPRESS)
    parser.add_argument('--push_dir', type=str, default=None,
                        choices=['behind', 'front', 'left', 'right'], help=argparse.SUPPRESS)
    parser.add_argument('--cpu', action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('--gpu', action='store_true', default=False, help=argparse.SUPPRESS)
    return parser


def validate_cli(parser: argparse.ArgumentParser, cli: argparse.Namespace) -> None:
    """Every check that doesn't need mjlab's registry — see main() for the
    two that do (unknown reward term, missing clip for a tracking task).
    Exits with argparse's code 2; TrainingManager surfaces that as
    "mjlab_train.py exited with code <rc> — see <log>"."""
    if cli.max_iterations is None and cli.max_minutes is None:
        parser.error("give at least one of --max_iterations / --max_minutes")

    supplied = [f"--{n}" for n in INAPPLICABLE if getattr(cli, n) not in (None, False)]
    if supplied:
        parser.error(f"{', '.join(supplied)} do not apply to mjlab task '{cli.task}' — "
                     f"a motion-tracking task has no velocity-command envelope, no "
                     f"base-height/tilt reward targets and no push domain randomization "
                     f"(see legged_gym/control/mjlab_adapter.py). Use --motion_file and "
                     f"--reward_scale instead.")

    # --cpu is a silent no-op: --device already defaults to cpu, so a caller
    # that still passes web_train.py's --cpu means exactly what we do anyway.
    if cli.gpu:
        parser.error("--gpu is not supported by mjlab_train.py; pass --device cuda:0")


def main():
    parser = build_parser()
    cli = parser.parse_args()
    validate_cli(parser, cli)

    env_cfg = registry.load_env_cfg(cli.task)
    agent_cfg = registry.load_rl_cfg(cli.task)
    runner_cls = registry.load_runner_cls(cli.task) or MjlabOnPolicyRunner

    env_cfg.scene.num_envs = cli.num_envs

    # A tracking task's registered motion_file is "" — mjlab raises at env
    # construction without one, so this is required rather than defaulted.
    motion_path = None
    if "motion" in env_cfg.commands:
        if not cli.motion_file:
            parser.error(f"--motion_file is required for tracking task '{cli.task}' "
                         f"(its command term has no default clip)")
        motion_path = cli.motion_file
        if not os.path.isabs(motion_path):
            motion_path = os.path.join(REPO_ROOT, motion_path)
        motion_path = os.path.abspath(motion_path)
        if not os.path.isfile(motion_path):
            parser.error(f"--motion_file not found: {motion_path}")
        env_cfg.commands["motion"].motion_file = motion_path

    # mjlab's rewards are a plain dict of name -> RewardTermCfg(weight=...),
    # NOT Genesis's env_cfg.rewards.scales class attributes. Applied before
    # ManagerBasedRlEnv(), since RewardManager reads weights at construction.
    for name, value in (cli.reward_scale or []):
        if name not in env_cfg.rewards:
            parser.error(f"unknown reward term '{name}' for mjlab task '{cli.task}' — "
                         f"valid terms: {', '.join(sorted(env_cfg.rewards))}")
        env_cfg.rewards[name].weight = float(value)

    seed = cli.seed if cli.seed is not None else agent_cfg.seed
    agent_cfg.seed = seed
    env_cfg.seed = seed
    # The registered default is "wandb", which would require a login on a
    # machine that has none. Forced to tensorboard — and this matters beyond
    # cosmetics: rsl-rl gates BOTH per-iteration printing and checkpoint
    # saving on `logger.writer is not None`.
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False  # MjlabOnPolicyRunner.save() would call logger.save_model()
    agent_cfg.run_name = cli.name
    if cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = cli.entropy_coef
    if cli.max_iterations is not None:
        # Cosmetic only (rsl-rl's own ETA line) — the real budget is the loop below.
        agent_cfg.max_iterations = cli.max_iterations

    log_dir = os.path.join(
        REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{cli.name}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"[mjlab_train] task={cli.task!r} num_envs={cli.num_envs} device={cli.device} "
          f"motion={motion_path} log_dir={log_dir}")

    configure_torch_backends()
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cli.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # registry_name deliberately not passed — MotionTrackingOnPolicyRunner
    # defaults it to None, which is what we want (no wandb registry).
    runner = runner_cls(env, asdict(agent_cfg), log_dir, cli.device)

    if cli.from_checkpoint:
        print(f"Fine-tuning from: {cli.from_checkpoint} (optimizer state NOT carried over)")
        runner.load(cli.from_checkpoint, load_cfg={
            "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": True})

    start_time = time.time()
    deadline = start_time + cli.max_minutes * 60 if cli.max_minutes is not None else None
    # Our OWN counter, never a current_learning_iteration delta — rsl-rl 5.x
    # sets that attribute to the LAST iteration index it ran (not last+1), so
    # it advances by chunk-1 on every resumed chunk. See the module docstring.
    state = {"iterations_done": 0}
    stopped_reason = "max_iterations" if cli.max_minutes is None else "max_minutes"

    def write_progress():
        if not cli.progress_path:
            return
        # Best-effort — a failed write must not kill a training run that's
        # otherwise fine; the parent just misses this update and reads the
        # next one. Same spirit as web_train.py's write_progress().
        payload = {
            "iterations_done": state["iterations_done"],
            "elapsed_s": round(time.time() - start_time, 1),
            "updated_at": time.time(),
            "backend": "mjlab",
        }
        for key in ("iteration", "reward", "episode_length", "noise_std", "reward_terms"):
            if key in state:
                payload[key] = state[key]
        try:
            with open(cli.progress_path, 'w') as f:
                json.dump(payload, f)
        except OSError:
            pass

    # rsl-rl clears logger.ep_extras at the END of every log() call, so the
    # per-term episode rewards simply don't exist between learn() chunks —
    # they have to be snapshotted from inside the call. Wrapping log() also
    # gets us one progress write per ITERATION instead of per chunk, with
    # values that match the printed block exactly (both read the same window).
    _orig_log = runner.logger.log

    def _log_and_report(**kwargs):
        import statistics
        terms = {}
        for entry in runner.logger.ep_extras:  # snapshot BEFORE log() clears it
            for key, value in entry.items():
                if key.startswith("Episode_Reward/"):
                    terms.setdefault(key[len("Episode_Reward/"):], []).append(float(value.mean()))
        _orig_log(**kwargs)  # prints + tensorboard + clears ep_extras
        state["iteration"] = kwargs["it"]
        state["noise_std"] = float(kwargs["action_std"].mean().item())
        state["reward_terms"] = {k: sum(v) / len(v) for k, v in terms.items()}
        if runner.logger.rewbuffer:
            state["reward"] = statistics.mean(runner.logger.rewbuffer)
        if runner.logger.lenbuffer:
            state["episode_length"] = statistics.mean(runner.logger.lenbuffer)
        write_progress()

    runner.logger.log = _log_and_report

    while True:
        if deadline is not None and time.time() >= deadline:
            stopped_reason = "max_minutes"
            break
        if cli.max_iterations is not None and state["iterations_done"] >= cli.max_iterations:
            stopped_reason = "max_iterations"
            break
        chunk = TIME_BUDGET_CHUNK_ITERS
        if cli.max_iterations is not None:
            chunk = min(chunk, cli.max_iterations - state["iterations_done"])
        # init_at_random_ep_len only on the FIRST chunk — re-randomizing
        # episode lengths mid-run would corrupt the tracking phase of
        # in-flight episodes. (The Genesis path passes True every chunk;
        # deliberately not copied.)
        runner.learn(num_learning_iterations=chunk,
                     init_at_random_ep_len=(state["iterations_done"] == 0))
        state["iterations_done"] += chunk
        write_progress()

    iterations_done = state["iterations_done"]
    print(f"Stopped after {iterations_done} iterations ({stopped_reason}).")

    # The tracking runner's save() auto-exports a 2-input/7-output ONNX next
    # to the checkpoints; that file is NOT usable as a policy_path (see the
    # module docstring). Calling the PARENT class's exporter explicitly is
    # what bypasses that override and yields the stateless obs->actions graph
    # load_onnx_backend() routes to OnnxStatelessPolicy — the same path
    # Javier's imported checkpoints already use.
    export_dir = os.path.join(log_dir, 'exported')
    MjlabOnPolicyRunner.export_policy_to_onnx(runner, export_dir, "policy.onnx")
    policy_path = os.path.join(export_dir, "policy.onnx")

    # learn() ends every call by saving model_{current_learning_iteration}.pt,
    # so unlike TrainingManager._train_checkpoint_from_export()'s after-the-
    # fact directory walk there's nothing to guess here — we just wrote it.
    train_checkpoint_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
    if not os.path.isfile(train_checkpoint_path):
        train_checkpoint_path = None

    with open(cli.result_path, 'w') as f:
        json.dump({
            "policy_path": policy_path, "task": cli.task, "name": cli.name,
            "iterations_done": iterations_done, "stopped_reason": stopped_reason,
            "train_checkpoint_path": train_checkpoint_path,
            "motion_file": cli.motion_file,
            "simulator": "mjlab",
        }, f)
    print(f"Done. Exported to {policy_path}")


if __name__ == '__main__':
    main()
