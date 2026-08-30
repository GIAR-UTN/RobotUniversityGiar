"""
rugiar — command-line front end for creating (training/fine-tuning) a
policy, without hand-composing a `python legged_gym/scripts/web_train.py
--task ... --name ... --...` invocation yourself.

This is deliberately a thin wrapper around
legged_gym.control.training.TrainingManager — the SAME engine the control
web's "Create Policy" panel drives (see TrainingManager.start()/poll()/
finalize_policy() and rugiar_driver.py's drain_finished_training(), which
this module's run_train() mirrors step for step: start() launches
web_train.py as a subprocess, poll() checks on it non-blockingly, and
finalize_policy() copies the result into its own self-contained
policies/<name>/ folder with a meta.json). Nothing about how a policy gets
built lives here a second time — rugiar only turns argv into the same
keyword arguments TrainingManager.start() already accepts, then prints what
it's doing. This is what keeps rugiar's flags and TrainingManager's params
impossible to let drift apart.

This one command line covers BOTH backends — a Genesis locomotion task
(e.g. 'g1') and an mjlab motion-tracking task (e.g. 'Rugiar-G1-Mimic') take
the exact same `rugiar train --task ... --name ...` shape, dispatched to
whichever interpreter/entrypoint that task actually needs (see
TrainingManager's TrainingBackend registry) — you never have to know or
care which venv you happen to be running `rugiar` from; the discovery
flags below (--list_tasks/--list_reward_scales/--list_motions) all work
correctly for an mjlab task even invoked from this (Genesis) venv, via a
one-shot probe into .venv-mjlab when this process can't import mjlab
itself.

Usage:
    rugiar train --task g1 --name crouch --max_minutes 15 \\
        --base_height_target 0.45 --push_robots off

    rugiar train --task g1 --name crouch_v2 --from_policy crouch \\
        --max_iterations 500 --reward_scale action_rate -0.1

    rugiar train --list_tasks
    rugiar train --task g1 --list_reward_scales

    # mjlab motion-tracking task: same shape, plus a required --motion_file
    rugiar train --list_motions --task Rugiar-G1-Mimic
    rugiar train --task Rugiar-G1-Mimic --name mimic_dance --max_iterations 3000 \\
        --motion_file resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz

    rugiar order --show
    rugiar order --set stable_home_made_4 crouch_walk
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# The registry itself, NOT TrainingManager: importing this costs nothing (no
# simulator, no torch) and it's what makes --backend's choices derived rather
# than hand-listed — a new backend descriptor becomes a valid --backend value
# with no edit here. See legged_gym/control/backends/__init__.py.
from legged_gym.control.backends import REQUESTABLE_BACKENDS, requestable_backend_options
# Same reasoning as the backends import above -- this is just the SCENARIOS dict
# (legged_gym/utils/scenarios.py), needed so 'drive's --scenario choices are derived,
# not hand-listed a second time.
from legged_gym.utils.scenarios import SCENARIOS

try:
    from rich_argparse import RawDescriptionRichHelpFormatter as _HelpFormatter
    _HelpFormatter.group_name_formatter = staticmethod(str)  # keep our own Title Case, don't re-.title() it
except ImportError:  # pragma: no cover - rich_argparse is a declared dependency, but don't hard-crash --help
    _HelpFormatter = argparse.RawDescriptionHelpFormatter

PROG = "rugiar"


def _build_train_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "train",
        formatter_class=_HelpFormatter,
        help="Train (or fine-tune) and register a new policy",
        description=(
            "Train a policy from scratch, or fine-tune an existing one, and register the "
            "result as ./policies/<name>/ — exactly what the control web's 'Create Policy' "
            "panel does, driven from the command line instead of a browser. Works for both "
            "Genesis locomotion tasks (e.g. 'g1') and mjlab motion-tracking tasks (e.g. "
            "'Rugiar-G1-Mimic', which additionally needs --motion_file) — same flags, "
            "auto-dispatched to whichever backend the task actually needs."
        ),
        epilog=(
            "examples:\n"
            "  # plain training run, stop after 15 minutes\n"
            "  rugiar train --task g1 --name crouch --max_minutes 15 \\\n"
            "      --base_height_target 0.45 --push_robots off\n\n"
            "  # fine-tune an existing local policy for another 500 iterations,\n"
            "  # penalizing jerky actions harder\n"
            "  rugiar train --task g1 --name crouch_v2 --from_policy crouch \\\n"
            "      --max_iterations 500 --reward_scale action_rate -0.1\n\n"
            "  # discover what's available before committing to a run\n"
            "  rugiar train --list_tasks\n"
            "  rugiar train --task g1 --list_reward_scales\n\n"
            "  # mjlab motion-tracking task: check which clips still need a policy, then train one\n"
            "  rugiar train --list_motions --task Rugiar-G1-Mimic\n"
            "  rugiar train --task Rugiar-G1-Mimic --list_reward_scales\n"
            "  rugiar train --task Rugiar-G1-Mimic --name mimic_dance --max_iterations 3000 \\\n"
            "      --motion_file resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz\n"
        ),
    )

    required = p.add_argument_group("Required")
    required.add_argument("--task", type=str, help="task/robot config to train against, e.g. g1 (see --list_tasks)")
    required.add_argument("--name", type=str,
                           help="policy name — becomes the ./policies/<name>/ folder and its display name "
                                "in the control web once trained")

    budget = p.add_argument_group("Compute budget (give at least one of the two stopping conditions)")
    budget.add_argument("--num_envs", type=int, default=64, help="parallel simulated environments")
    budget.add_argument("--max_iterations", type=int, default=None,
                         help="stop after this many learning iterations (counted from wherever "
                              "--from_policy resumed, if given)")
    budget.add_argument("--max_minutes", type=float, default=None,
                         help="stop after this many minutes of wall-clock training time, regardless "
                              "of iteration count")

    finetune = p.add_argument_group("Fine-tuning (start from an existing local policy instead of random init)")
    finetune.add_argument("--from_policy", type=str, default=None, metavar="NAME",
                           help="name of an existing ./policies/<NAME>/ to fine-tune from (must have a "
                                "train_checkpoint.pt — see --list_policies). Optimizer state is NOT "
                                "carried over, only weights.")

    envelope = p.add_argument_group("Command envelope (the velocity commands the policy is trained across)")
    envelope.add_argument("--cmd_vx_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                           help="forward/backward velocity range in m/s")
    envelope.add_argument("--cmd_vy_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                           help="lateral velocity range in m/s")
    envelope.add_argument("--cmd_yaw_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                           help="yaw angular velocity range in rad/s")

    stability = p.add_argument_group("Stability targets (override what the task's own reward considers 'good')")
    stability.add_argument("--base_height_target", type=float, default=None, metavar="M",
                            help="base height (m) the height-tracking reward targets — e.g. lower this "
                                 "to train a crouch instead of the task's default standing height")
    stability.add_argument("--lin_vel_z_target", type=float, default=None, metavar="M_S",
                            help="vertical base velocity (m/s) targeted — default 0 (no bobbing)")
    stability.add_argument("--ang_vel_xy_target", type=float, default=None, metavar="RAD_S",
                            help="roll/pitch angular velocity magnitude (rad/s) targeted — default 0 (no wobble)")
    stability.add_argument("--orientation_tilt_target", type=float, default=None, metavar="TILT",
                            help="body tilt off upright targeted — default 0 (upright)")

    push = p.add_argument_group("Push perturbation (domain randomization)")
    push.add_argument("--push_robots", type=str, default=None, choices=["on", "off"],
                       help="force random push disturbances on/off (default: leave the task's own setting)")
    push.add_argument("--max_push_vel_xy", type=float, default=None, metavar="M_S",
                       help="peak push velocity (m/s) when --push_robots on")
    push.add_argument("--push_interval_s", type=float, default=None,
                       help="seconds between pushes when --push_robots on")
    push.add_argument("--push_dir", type=str, default=None, choices=["behind", "front", "left", "right"],
                       help="bias push direction relative to the robot's heading instead of sampling it "
                            "isotropically — e.g. 'behind' shoves the robot forward")

    reward = p.add_argument_group("Reward shaping")
    reward.add_argument("--reward_scale", type=str, nargs=2, action="append", default=None,
                         metavar=("NAME", "VALUE"),
                         help="override one reward-term weight, e.g. --reward_scale alive 0.3 — repeatable, "
                              "one flag per term (see --list_reward_scales for valid NAMEs and current "
                              "defaults for --task). Positive rewards more of that term, negative penalizes "
                              "it; magnitude is relative to the other terms, not absolute.")
    reward.add_argument("--motion_file", type=str, default=None, metavar="PATH",
                         help="reference-motion clip to train against, relative to "
                              "resources/reference_motion/ — a .pkl for a Genesis motion-imitation task "
                              "whose env cfg has its own 'motion_file' default (e.g. g1_deepmimic), or a "
                              ".npz under unitree_g1/mjlab_run/ for an mjlab tracking task (e.g. "
                              "Rugiar-G1-Mimic — see --list_motions). REQUIRED for an mjlab tracking task; "
                              "errors on any task that doesn't use one.")
    reward.add_argument("--entropy_coef", type=float, default=None,
                         help="PPO's exploration-noise bonus weight (default: the task's own). Lower this "
                              "if 'Mean action noise std' trends up instead of down over training.")

    backend = p.add_argument_group("Backend")
    backend.add_argument("--backend", type=str, default="local", choices=list(REQUESTABLE_BACKENDS),
                          help="where training runs: " + ", ".join(
                              f"'{o['id']}' ({o['label']})" for o in requestable_backend_options())
                              + ". 'local' runs on this machine (CPU); 'kaggle' uploads and runs on "
                                "a Kaggle GPU kernel (requires ~/.kaggle/kaggle.json)")

    discover = p.add_argument_group("Discovery (print information and exit — no training)")
    discover.add_argument("--list_tasks", action="store_true",
                           help="list every registered task name, both Genesis (this venv's own "
                                "task_registry) and mjlab (probed via .venv-mjlab if this process can't "
                                "import mjlab itself), then exit")
    discover.add_argument("--list_reward_scales", action="store_true",
                           help="list every overridable --reward_scale NAME and its current default for "
                                "--task, then exit — works for an mjlab --task too (e.g. Rugiar-G1-Mimic), "
                                "even run from this (Genesis) venv")
    discover.add_argument("--list_policies", action="store_true",
                           help="list every local ./policies/<name>/ available as a --from_policy base, "
                                "then exit")
    discover.add_argument("--list_motions", action="store_true",
                           help="list every reference-motion .npz clip under "
                                "resources/reference_motion/unitree_g1/mjlab_run/, each flagged whether a "
                                "local policy already targets it (see --motion_file) — pass --task to scope "
                                "the has_policy flag to just that task, then exit")

    execu = p.add_argument_group("Execution")
    execu.add_argument("--poll_interval", type=float, default=2.0, metavar="SECONDS",
                        help="how often to check on the running job and flush its log to this terminal")

    return p


DRIVE_PRESETS = {
    "genesis": {
        "script": "legged_gym/scripts/rugiar_driver.py",
        "python": ".venv/bin/python",
        "default_task": "g1",
        "env": {"SIMULATOR": "genesis"},
        "extra_args": [],
    },
    "mjlab": {
        "script": "legged_gym/scripts/rugiar_driver_mjlab.py",
        "python": ".venv-mjlab/bin/python",
        "default_task": "Rugiar-G1-Mimic",
        # CUDA_VISIBLE_DEVICES="" matches this driver's own documented launch line
        # (see rugiar_driver_mjlab.py's module docstring) -- this repo's mjlab setup
        # is CPU-only on Mac, same reasoning as genesis's own cpu-backend default.
        "env": {"SIMULATOR": "mjlab", "CUDA_VISIBLE_DEVICES": ""},
        "extra_args": ["--ramp_ticks", "15"],
    },
}


def _build_drive_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "drive",
        formatter_class=_HelpFormatter,
        help="Launch the control web (Genesis or mjlab) without hand-composing rugiar_driver.py flags",
        description=(
            "Thin launcher around 'python legged_gym/scripts/rugiar_driver.py' (Genesis) / "
            "'rugiar_driver_mjlab.py' (mjlab) -- picks the right venv's interpreter, the right "
            "SIMULATOR/env vars, and a sensible default --task per backend, so switching which "
            "system is running doesn't require remembering both scripts' full flag set. Local "
            "policies under ./policies/ are auto-discovered at startup either way -- no need to "
            "list them with --policy. If something is already listening on --control_port, it's "
            "stopped first (same 'one session per port' rule the Family panel's own switch uses) "
            "so this never leaves two drivers fighting over the same port."
        ),
        epilog=(
            "examples:\n"
            "  # Genesis backend, default task 'g1', default ports\n"
            "  rugiar drive genesis\n\n"
            "  # mjlab backend, a specific task\n"
            "  rugiar drive mjlab --task g1_target\n\n"
            "  # preview a different reference-motion clip before training against it\n"
            "  rugiar drive mjlab --motion_file resources/reference_motion/unitree_g1/mjlab_run/g1moves_B_DadDance.npz\n\n"
            "  # custom ports (e.g. running a second, unrelated session)\n"
            "  rugiar drive genesis --task g1_deepmimic --control_port 9018 --viser_port 9007\n"
        ),
    )
    p.add_argument("system", choices=sorted(DRIVE_PRESETS), help="which backend/driver to launch")
    p.add_argument("--task", type=str, default=None,
                    help="registered task to drive (default: 'g1' for genesis, 'Rugiar-G1-Mimic' for mjlab)")
    p.add_argument("--control_port", type=int, default=9017,
                    help="control web + WebSocket port (default: 9017, the LAS port-registry value)")
    p.add_argument("--viser_port", type=int, default=9006, help="raw 3D viewer port (default: 9006)")
    p.add_argument("--headless", action="store_true", default=False,
                    help="no viewer/control web -- runs a scripted smoke test instead")
    p.add_argument("--motion_file", type=str, default=None,
                    help="mjlab only -- reference-motion .npz to preview/track instead of the task's "
                         "default (see resources/reference_motion/unitree_g1/mjlab_run/*.npz). Lets you "
                         "sanity-check a new clip retargets/plays sensibly on the robot before spending "
                         "a training run on it. Errors if passed with --system genesis.")
    p.add_argument("--ramp_ticks", type=int, default=None,
                    help="control ticks to cross-fade over on a policy switch -- forwarded to either "
                         "backend as-is. mjlab's own preset already passes 15 by default; this overrides "
                         "that (or genesis's own script default) when given.")
    p.add_argument("--scenario", type=str, default=None, choices=sorted(SCENARIOS),
                    help="genesis only -- which named scenario's props/scenery/web-UI config to use "
                         "(see legged_gym/utils/scenarios.py::SCENARIOS). Forwarded as-is; errors if "
                         "passed with --system mjlab, which has no --scenario of its own.")
    p.add_argument("--scenario-option", action="append", default=[], metavar="KEY=VALUE",
                    dest="scenario_option",
                    help="genesis only -- override one of --scenario's default options, e.g. "
                         "track_length=10 (repeatable). Errors if passed with --system mjlab.")
    p.add_argument("--camera", action="store_true", default=False,
                    help="genesis only -- stream a robot-POV RGB camera feed to the control web. "
                         "Errors if passed with --system mjlab.")
    p.add_argument("--speed", type=float, default=None,
                    help="genesis only -- sim playback speed multiplier (1.0 = real-time 50Hz). "
                         "Errors if passed with --system mjlab (mjlab paces off its own control_dt).")
    p.add_argument("--cruise_limit", type=float, default=None,
                    help="genesis only -- startup default for ControlService.set_operator_speed_limit() "
                         "(fraction of the trained command envelope, 1.0 = full range). Errors if passed "
                         "with --system mjlab.")
    p.add_argument("--no_replace", action="store_true", default=False,
                    help="don't stop an existing process on --control_port first -- fail instead "
                         "if the port's already taken")
    return p


def _pids_listening_on(port: int) -> list:
    import subprocess
    try:
        out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                              capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    return [int(pid) for pid in out.stdout.split() if pid.strip()]


def _stop_listening_on(port: int, label: str = "process") -> list:
    """SIGTERMs whatever's listening on `port`, waits up to 8s, then SIGKILLs any
    holdout -- the same "stop whatever's already here" logic 'drive' uses before
    launching, factored out so 'stop' (bring the environment down without starting
    anything new) can reuse it verbatim. Returns the pid(s) it found (empty if the
    port was already free)."""
    import os
    import signal
    import time

    existing = _pids_listening_on(port)
    if not existing:
        return []
    print(f"[rugiar] stopping {label} on port {port}: pid(s) {existing}")
    for pid in existing:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 8
    while time.time() < deadline and _pids_listening_on(port):
        time.sleep(0.3)
    for pid in _pids_listening_on(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return existing


def run_drive(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import os
    import shlex

    preset = DRIVE_PRESETS[args.system]
    repo_root = Path.cwd()
    python_bin = repo_root / preset["python"]
    script = repo_root / preset["script"]
    if not python_bin.is_file():
        parser.error(
            f"{preset['python']} not found under {repo_root} -- run this from the repo root, "
            f"with the '{args.system}' venv already set up (see README)."
        )
    if not script.is_file():
        parser.error(f"{preset['script']} not found under {repo_root} -- run this from the repo root.")
    if args.motion_file and args.system != "mjlab":
        parser.error("--motion_file only applies to --system mjlab")
    # --scenario/--scenario-option/--camera/--speed/--cruise_limit are genesis-only --
    # rugiar_driver_mjlab.py has no equivalent flags at all (see its own --help).
    genesis_only = {
        "--scenario": args.scenario, "--scenario-option": args.scenario_option,
        "--camera": args.camera, "--speed": args.speed, "--cruise_limit": args.cruise_limit,
    }
    if args.system != "genesis":
        bad = [flag for flag, val in genesis_only.items() if val]
        if bad:
            parser.error(f"{', '.join(bad)} only apply(ies) to --system genesis")

    if args.no_replace:
        if _pids_listening_on(args.control_port):
            parser.error(f"port {args.control_port} is already in use "
                         f"(pass without --no_replace to stop it automatically)")
    else:
        _stop_listening_on(args.control_port, label="existing process")

    task = args.task or preset["default_task"]
    argv = [str(python_bin), str(script), "--task", task,
            "--viser_port", str(args.viser_port), "--control_port", str(args.control_port)]
    argv += preset["extra_args"]
    if args.headless:
        argv.append("--headless")
    if args.motion_file:
        argv += ["--motion_file", args.motion_file]
    # Appended AFTER preset["extra_args"] deliberately -- for a non-append flag like
    # --ramp_ticks, argparse (and every underlying script here) takes the LAST value
    # given, so an explicit --ramp_ticks on this command always wins over mjlab's own
    # preset default of 15, without needing to special-case that preset here.
    if args.ramp_ticks is not None:
        argv += ["--ramp_ticks", str(args.ramp_ticks)]
    if args.scenario:
        argv += ["--scenario", args.scenario]
    for opt in args.scenario_option:
        argv += ["--scenario-option", opt]
    if args.camera:
        argv.append("--camera")
    if args.speed is not None:
        argv += ["--speed", str(args.speed)]
    if args.cruise_limit is not None:
        argv += ["--cruise_limit", str(args.cruise_limit)]

    env = os.environ.copy()
    env.update(preset["env"])

    print(f"[rugiar] launching {args.system} — {' '.join(shlex.quote(a) for a in argv)}")
    if not args.headless:
        print(f"[rugiar] control web: http://localhost:{args.control_port}/")
    os.chdir(repo_root)
    os.execvpe(argv[0], argv, env)  # replaces this process -- Ctrl-C behaves exactly like a direct launch
    return 0  # unreachable, execvpe never returns on success


def _build_stop_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "stop",
        formatter_class=_HelpFormatter,
        help="Stop whatever 'rugiar drive' session (genesis or mjlab) is listening on a port",
        description=(
            "Brings a running 'rugiar drive' session down without starting a replacement -- the "
            "same SIGTERM-then-SIGKILL-on-timeout logic 'drive' already runs before launching, "
            "exposed on its own for 'just stop it' / 'stop the one on port X, leave the other "
            "running' without needing a second 'drive' invocation to trigger it."
        ),
        epilog=(
            "examples:\n"
            "  # stop whatever's on the default control port\n"
            "  rugiar stop\n\n"
            "  # stop a session running on a non-default port (e.g. a second, parallel one)\n"
            "  rugiar stop --control_port 9018\n"
        ),
    )
    p.add_argument("--control_port", type=int, default=9017,
                    help="port to stop whatever's listening on (default: 9017, same default 'drive' "
                         "uses)")
    return p


def run_stop(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    stopped = _stop_listening_on(args.control_port, label="session")
    if not stopped:
        print(f"[rugiar] nothing was listening on port {args.control_port}")
    else:
        print(f"[rugiar] stopped pid(s) {stopped} on port {args.control_port}")
    return 0


def _build_order_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "order",
        formatter_class=_HelpFormatter,
        help="View or set the display order of local policies",
        description=(
            "Configure the order local policies are listed in — not what's used to TRAIN or "
            "fine-tune a policy, but the catalog order a separate downstream layer (e.g. a "
            "control app that only selects among already-trained policies) can read via "
            "TrainingManager.get_policy_order() to decide what to offer first."
        ),
        epilog=(
            "examples:\n"
            "  # see the current order (unpinned policies float to the end, alphabetically)\n"
            "  rugiar order --show\n\n"
            "  # pin these two first; every other local policy still appears, after them\n"
            "  rugiar order --set stable_home_made_4 crouch_walk\n"
        ),
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--show", action="store_true",
                        help="print the current policy order, then exit")
    group.add_argument("--set", type=str, nargs="+", default=None, metavar="NAME",
                        help="set the display order to this sequence of local policy names "
                             "(see --list_policies under 'train' for valid names). Names left "
                             "out keep appearing, alphabetically, after the ones given here.")
    return p


def _build_fuse_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "fuse",
        formatter_class=_HelpFormatter,
        description=(
            "Merge 2+ already-trained local policies' weights into a new one — no further "
            "training involved. Exactly what the control web's 'Fuse policies…' panel does, "
            "driven from the command line instead of a browser. The result is registered as a "
            "normal ./policies/<name>/ — fine-tunable via `train --from_policy` and fusable "
            "again, same as anything trained through this UI."
        ),
        help="Merge 2+ existing policies' weights into a new policy",
        epilog=(
            "examples:\n"
            "  # uniform 2-way weighted average\n"
            "  rugiar fuse --policies stable_home_made_3 stable_home_made_4 --name blended\n\n"
            "  # weighted 3-way merge, favoring the first source\n"
            "  rugiar fuse --policies base_a base_b base_c --weights 2 1 1 --name blended_v2\n\n"
            "  # see what fusion methods exist (implemented and planned)\n"
            "  rugiar fuse --list_fusion_methods\n"
        ),
    )
    required = p.add_argument_group("Required")
    required.add_argument("--policies", type=str, nargs="+", metavar="NAME",
                           help="2+ local policy names to merge (see `train --list_policies`); "
                                "each must have a train_checkpoint.pt, same restriction "
                                "`train --from_policy` has")
    required.add_argument("--name", type=str, help="output policy name — becomes ./policies/<name>/, "
                                                     "same as a trained policy")

    weighting = p.add_argument_group("Weighting")
    weighting.add_argument("--weights", type=float, nargs="+", default=None, metavar="W",
                            help="one weight per --policies entry, same order (default: uniform). "
                                 "Normalized to sum to 1 internally — raw ratios are fine, e.g. "
                                 "'2 1 1' weighs the first source twice as heavily as the others")
    weighting.add_argument("--method", type=str, default="weighted_average",
                            help="fusion method (see --list_fusion_methods for the full list) — "
                                 "'weighted_average' (default) or 'git_rebasin' (permutation "
                                 "alignment, works for both plain and recurrent policies)")

    disambig = p.add_argument_group(
        "Task disambiguation (only matters if --policies spans more than one task)")
    disambig.add_argument("--export_task", type=str, default=None, metavar="TASK",
                           help="which task to register the fused policy under (default: the "
                                "first --policies entry's task). Must describe the same obs/action "
                                "space as the sources being merged — a mismatch is rejected outright, "
                                "not just warned about")

    discover = p.add_argument_group("Discovery (print information and exit — no fusing)")
    discover.add_argument("--list_fusion_methods", action="store_true",
                           help="list every fusion method this build knows about, including ones "
                                "not implemented yet, then exit")
    discover.add_argument("--list_policies", action="store_true",
                           help="list every local ./policies/<name>/ available to fuse — same as "
                                "`train --list_policies`; 'fine-tunable=yes' means 'fusable=yes' too")

    return p


def _build_distill_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "distill",
        formatter_class=_HelpFormatter,
        help="Behavior-clone a teacher policy into a fresh, fine-tunable policy",
        description=(
            "Rolls a TEACHER policy (any local policy — crucially including ones with no "
            "train_checkpoint.pt at all, e.g. an externally-sourced policy like 'stable') "
            "through --task's own simulator, collects (observation, action) pairs, and "
            "supervise-trains a fresh network to imitate it. Unlike `rugiar fuse` (which "
            "merges WEIGHTS and requires matching architectures), this clones BEHAVIOR — "
            "the result is an ordinary rsl_rl checkpoint, registered as ./policies/<name>/ "
            "exactly like a trained-from-scratch run, so it's immediately usable with "
            "`rugiar train --from_policy <name>` to keep training it with PPO."
        ),
        epilog=(
            "examples:\n"
            "  # clone 'stable' (no training history of its own) into a fine-tunable policy\n"
            "  rugiar distill --teacher stable --task g1 --name stable_distilled\n\n"
            "  # then keep training it normally\n"
            "  rugiar train --task g1 --name stable_distilled_ft --from_policy stable_distilled \\\n"
            "      --max_iterations 500\n\n"
            "  # see what distillation methods exist (implemented and planned)\n"
            "  rugiar distill --list_distill_methods\n"
        ),
    )
    required = p.add_argument_group("Required")
    required.add_argument("--teacher", type=str, metavar="NAME",
                           help="local policy to clone (see `train --list_policies`) — a "
                                "train_checkpoint.pt is NOT required, unlike `fuse`'s sources")
    required.add_argument("--task", type=str, help="target task — determines the student's own "
                                                     "obs/action space and network architecture")
    required.add_argument("--name", type=str, help="output policy name — becomes ./policies/<name>/, "
                                                     "same as a trained policy")

    hyper = p.add_argument_group("Hyperparameters (optional — sane defaults if omitted)")
    hyper.add_argument("--rollout_steps", type=int, default=None,
                        help="env ticks to roll the teacher for while collecting data (default: 4000)")
    hyper.add_argument("--bc_epochs", type=int, default=None,
                        help="supervised training epochs over the collected rollout (default: 20)")
    hyper.add_argument("--lr", type=float, default=None, help="Adam learning rate for BC training (default: 1e-3)")
    hyper.add_argument("--num_envs", type=int, default=None,
                        help="parallel envs for rollout collection (default: 1 — some externally-sourced "
                             "teachers can only be stepped one at a time, see start_distillation()'s "
                             "docstring; a locally-trained teacher can safely go higher for a faster rollout)")
    hyper.add_argument("--method", type=str, default="behavior_cloning",
                        help="distillation method (see --list_distill_methods) — 'behavior_cloning' (default)")
    hyper.add_argument("--dagger_rounds", type=int, default=None,
                        help="'dagger' method only — rollout+relabel+retrain rounds (default: 5). "
                             "--rollout_steps/--bc_epochs become PER-ROUND values under dagger.")
    hyper.add_argument("--dagger_beta0", type=float, default=None,
                        help="'dagger' method only — teacher-action fraction in round 0 (default: 1.0)")
    hyper.add_argument("--dagger_beta_decay", type=float, default=None,
                        help="'dagger' method only — per-round multiplicative decay on that fraction (default: 0.5)")

    discover = p.add_argument_group("Discovery (print information and exit — no distilling)")
    discover.add_argument("--list_distill_methods", action="store_true",
                           help="list every distillation method this build knows about, including "
                                "ones not implemented yet, then exit")
    discover.add_argument("--list_policies", action="store_true",
                           help="list every local ./policies/<name>/ available to distill from — "
                                "same as `train --list_policies`, but here fine-tunable=no is fine too")

    poll = p.add_argument_group("Polling")
    poll.add_argument("--poll_interval", type=float, default=2.0,
                       help="seconds between progress checks while the job runs (default: 2.0)")
    return p


def _list_distill_methods() -> None:
    from legged_gym.control import distillation
    rows = []
    for key, info in distillation.DISTILL_METHODS.items():
        status = "available" if info["available"] else "planned, not yet implemented"
        rows.append(f"{key} ({status}): {info['label']} — {info['description']}")
    _print_lines("Distillation methods:", rows)


def run_distill(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import legged_gym.envs  # noqa: F401 — see run_train()'s identical import for why
    from legged_gym.control.training import TrainingManager

    mgr = TrainingManager()
    for name, info in mgr.discover_local_policies().items():
        mgr.register_source(name, **info)

    if args.list_distill_methods:
        _list_distill_methods()
        return 0
    if args.list_policies:
        _list_policies(mgr)
        return 0

    if not args.teacher or not args.task or not args.name:
        parser.error("--teacher, --task, and --name are required to distill a policy")

    try:
        job_id = mgr.start_distillation(
            args.teacher, args.task, args.name, rollout_steps=args.rollout_steps,
            bc_epochs=args.bc_epochs, lr=args.lr, num_envs=args.num_envs, method=args.method,
            dagger_rounds=args.dagger_rounds, dagger_beta0=args.dagger_beta0,
            dagger_beta_decay=args.dagger_beta_decay,
        )
    except ValueError as e:
        parser.error(str(e))
        return 2  # unreachable, parser.error() exits — keeps type-checkers happy

    job = mgr.jobs[job_id]
    print(f"[rugiar] started job {job_id} — {job.command}")

    log_path = Path(job.log_path)
    read_so_far = 0

    def flush_log() -> None:
        nonlocal read_so_far
        if not log_path.is_file():
            return
        with open(log_path) as f:
            f.seek(read_so_far)
            chunk = f.read()
            read_so_far = f.tell()
        if chunk:
            print(chunk, end="")

    try:
        while True:
            mgr.poll()
            flush_log()
            if job.status != "running":
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        proc = mgr._procs.get(job_id)  # local backend only — see TrainingManager.start()
        if proc is not None:
            proc.terminate()
        print("\n[rugiar] interrupted — distillation process terminated, policy NOT registered.", file=sys.stderr)
        return 130

    flush_log()

    if job.status == "failed":
        print(f"[rugiar] job {job_id} failed: {job.error}", file=sys.stderr)
        return 1

    final_checkpoint = mgr.finalize_policy(
        job.policy_name, task=job.task, checkpoint=job.policy_path,
        train_checkpoint=job.train_checkpoint_path, job=job,
    )
    print(f"\n[rugiar] '{job.policy_name}' ready — distilled from '{args.teacher}', "
          f"final_bc_loss={job.final_bc_loss}, checkpoint at {final_checkpoint}")
    diag = job.rollout_diagnostics or {}
    yaw = diag.get("commanded_ang_vel_yaw")
    vx = diag.get("commanded_lin_vel_x")
    if yaw and vx:
        yaw_span = yaw["max"] - yaw["min"]
        vx_span = vx["max"] - vx["min"]
        print(f"[rugiar] rollout command coverage — vx range={vx_span:.3f} m/s, "
              f"yaw range={yaw_span:.3f} rad/s (single {args.num_envs}-env trajectory; "
              "a narrow range here means the student never saw the teacher do that, "
              "regardless of final_bc_loss)")
    return 0


def _list_fusion_methods() -> None:
    from legged_gym.control import fusion
    rows = []
    for key, info in fusion.FUSION_METHODS.items():
        status = "available" if info["available"] else "planned, not yet implemented"
        rows.append(f"{key} ({status}): {info['label']} — {info['description']}")
    _print_lines("Fusion methods:", rows)


def run_fuse(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import legged_gym.envs  # noqa: F401 — see run_train()'s identical import for why
    from legged_gym.control.training import TrainingManager

    mgr = TrainingManager()
    for name, info in mgr.discover_local_policies().items():
        mgr.register_source(name, **info)

    if args.list_fusion_methods:
        _list_fusion_methods()
        return 0
    if args.list_policies:
        _list_policies(mgr)
        return 0

    if not args.policies or len(args.policies) < 2:
        parser.error("--policies needs at least 2 names")
    if not args.name:
        parser.error("--name is required")

    try:
        result = mgr.fuse_policies(
            args.policies, args.name, weights=args.weights,
            method=args.method, export_task=args.export_task,
        )
    except ValueError as e:
        parser.error(str(e))
        return 2  # unreachable, parser.error() exits — keeps type-checkers happy

    print(f"[rugiar] '{result['name']}' ready — {result['checkpoint_path']}")
    for warning in result["warnings"]:
        print(f"[rugiar] warning: {warning}", file=sys.stderr)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROG,
        formatter_class=_HelpFormatter,
        description="rugiar — command-line interface for RobotUniversityGiar policy creation and driving.",
        epilog=(
            "Scope: rugiar trains, fine-tunes, fuses, and distills policies, and launches the\n"
            "control web/driver ('rugiar drive') -- for anything beyond that (raw --policy-by-\n"
            "--policy control, --real hardware flags), see\n"
            "'python legged_gym/scripts/rugiar_driver.py --help' directly (--control_port serves\n"
            "both a browser control web and a WebSocket protocol any custom client, e.g. a\n"
            "home-made joystick, can speak — docs/index.html's 'Talking to the robot' section has\n"
            "the full wire format). For the system-wide area map, see legged_gym/control/ARCHITECTURE.md."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = _build_train_parser(subparsers)
    order_parser = _build_order_parser(subparsers)
    fuse_parser = _build_fuse_parser(subparsers)
    distill_parser = _build_distill_parser(subparsers)
    drive_parser = _build_drive_parser(subparsers)
    stop_parser = _build_stop_parser(subparsers)
    return parser, {"train": train_parser, "order": order_parser, "fuse": fuse_parser,
                     "distill": distill_parser, "drive": drive_parser, "stop": stop_parser}


def _print_lines(title: str, rows) -> None:
    print(title)
    for row in rows:
        print(f"  {row}")


def _list_tasks(mgr) -> None:
    # _legged_gym_registered_tasks() (not a bare `from legged_gym.utils import
    # task_registry`) on purpose: `rugiar` is only ever pip-installed under
    # `.venv` today, so this practically always succeeds there — but the CLI
    # module is still directly importable/callable from `.venv-mjlab` (e.g.
    # by tests, or a future `.venv-mjlab/bin/rugiar` install), where
    # legged_gym.utils doesn't import cleanly (docs/mjlab_migration.md
    # R1-adjacent), AND `from legged_gym.utils import task_registry` can
    # resolve to the bare SUBMODULE instead of the re-exported TaskRegistry
    # instance depending on import order — this shared helper already
    # handles both failure modes instead of re-deriving them here.
    from legged_gym.control.training import _legged_gym_registered_tasks, _mjlab_registry_snapshot

    genesis_tasks_set = _legged_gym_registered_tasks()
    if genesis_tasks_set is not None:
        genesis_tasks = sorted(genesis_tasks_set)
        _print_lines("Registered tasks (genesis):", genesis_tasks)
    else:
        _print_lines("Registered tasks (genesis — not importable from this process):", [])
        genesis_tasks = []

    mjlab_tasks_ = _mjlab_registry_snapshot()
    if mjlab_tasks_ is not None:
        _print_lines("Registered tasks (mjlab):", sorted(mjlab_tasks_))
        return
    # Probe failed outright (no .venv-mjlab on this machine, or it errored) —
    # fall back to whatever mjlab tasks local policies already declare, so
    # e.g. Rugiar-G1-Mimic still shows up via Javier's imported checkpoints
    # even when the mjlab venv/registry itself can't be reached at all.
    fallback = sorted({info["task"] for info in mgr.discover_local_policies().values()
                        if info.get("task") not in genesis_tasks})
    _print_lines(
        "Registered tasks (mjlab — probe failed, showing tasks known only via local policies):",
        fallback or ["(none — no .venv-mjlab found and no local mjlab policies either)"],
    )


def _list_motions(mgr, task: Optional[str]) -> None:
    from legged_gym.control.service import MOTION_DIR, motion_clip_rows

    if not MOTION_DIR.is_dir():
        _print_lines(f"Reference-motion clips ({MOTION_DIR}):", ["(directory not found)"])
        return
    rows = motion_clip_rows(mgr.discover_local_policies(), task)
    lines = [f"{r['name']}  (has_policy={'yes' if r['has_policy'] else 'no'})  — {r['path']}" for r in rows]
    title = f"Reference-motion clips under {MOTION_DIR}" + (f" (task={task})" if task else "") + ":"
    _print_lines(title, lines or ["(none found)"])


def _list_reward_scales(mgr, task: str) -> None:
    defaults = mgr.task_defaults(task)
    rows = []
    for name, value in sorted(defaults["reward_scales"].items()):
        note = defaults["reward_scale_notes"].get(name)
        rows.append(f"{name} = {value}" + (f"  # {note}" if note else ""))
    _print_lines(f"Reward scales for task '{task}' (current defaults):", rows)


def _list_policies(mgr) -> None:
    discovered = mgr.discover_local_policies()
    order = mgr.get_policy_order()
    rows = [
        f"{name}  (task={discovered[name]['task']}, "
        f"fine-tunable={'yes' if discovered[name]['train_checkpoint'] else 'no'})"
        for name in order
    ]
    _print_lines("Local policies (./policies/<name>/), in configured order:", rows or ["(none found)"])


def run_order(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import legged_gym.envs  # noqa: F401 — see run_train()'s identical import for why
    from legged_gym.control.training import TrainingManager

    mgr = TrainingManager()
    for name, info in mgr.discover_local_policies().items():
        mgr.register_source(name, **info)

    if args.set is not None:
        try:
            mgr.set_policy_order(args.set)
        except ValueError as e:
            parser.error(str(e))
            return 2  # unreachable, parser.error() exits — keeps type-checkers happy
        print(f"[rugiar] order set — pinned {len(args.set)} name(s), others float to the end alphabetically")

    _print_lines("Policy order (./policies/.policy_order.json):", mgr.get_policy_order() or ["(none found)"])
    return 0


def run_train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    # Populates task_registry.task_classes as a side effect of import — see
    # web_train.py's own `from legged_gym.envs import *`. Needed even for
    # --list_tasks/--list_reward_scales, not just an actual training run.
    import legged_gym.envs  # noqa: F401
    from legged_gym.control.training import TrainingManager

    mgr = TrainingManager()
    for name, info in mgr.discover_local_policies().items():
        mgr.register_source(name, **info)

    if args.list_tasks:
        _list_tasks(mgr)
        return 0
    if args.list_policies:
        _list_policies(mgr)
        return 0
    if args.list_motions:
        _list_motions(mgr, args.task)
        return 0
    if args.list_reward_scales:
        if not args.task:
            parser.error("--list_reward_scales needs --task")
        _list_reward_scales(mgr, args.task)
        return 0

    if not args.task or not args.name:
        parser.error("--task and --name are required to train a policy")
    if args.max_iterations is None and args.max_minutes is None:
        parser.error("give at least one of --max_iterations / --max_minutes")

    reward_scale_overrides = None
    if args.reward_scale:
        reward_scale_overrides = {name: float(value) for name, value in args.reward_scale}
    push_robots = {"on": True, "off": False}.get(args.push_robots) if args.push_robots else None

    try:
        job_id = mgr.start(
            policy_name=args.name, task=args.task, num_envs=args.num_envs,
            max_iterations=args.max_iterations, max_minutes=args.max_minutes,
            base_policy=args.from_policy,
            cmd_vx=args.cmd_vx_range, cmd_vy=args.cmd_vy_range, cmd_yaw=args.cmd_yaw_range,
            base_height_target=args.base_height_target, lin_vel_z_target=args.lin_vel_z_target,
            ang_vel_xy_target=args.ang_vel_xy_target, orientation_tilt_target=args.orientation_tilt_target,
            push_robots=push_robots, max_push_vel_xy=args.max_push_vel_xy,
            push_interval_s=args.push_interval_s, push_dir=args.push_dir,
            entropy_coef=args.entropy_coef, reward_scale_overrides=reward_scale_overrides,
            motion_file=args.motion_file,
            backend=args.backend,
        )
    except ValueError as e:
        parser.error(str(e))
        return 2  # unreachable, parser.error() exits — keeps type-checkers happy

    job = mgr.jobs[job_id]
    print(f"[rugiar] started job {job_id} — {job.command}")

    log_path = Path(job.log_path)
    read_so_far = 0

    def flush_log() -> None:
        nonlocal read_so_far
        if not log_path.is_file():
            return
        with open(log_path) as f:
            f.seek(read_so_far)
            chunk = f.read()
            read_so_far = f.tell()
        if chunk:
            print(chunk, end="")

    try:
        while True:
            mgr.poll()
            flush_log()
            if job.status != "running":
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        proc = mgr._procs.get(job_id)  # local backend only — see TrainingManager.start()
        if proc is not None:
            proc.terminate()
        print("\n[rugiar] interrupted — training process terminated, policy NOT registered.", file=sys.stderr)
        return 130

    flush_log()

    if job.status == "failed":
        print(f"[rugiar] job {job_id} failed: {job.error}", file=sys.stderr)
        return 1

    final_checkpoint = mgr.finalize_policy(
        job.policy_name, task=job.task, checkpoint=job.policy_path,
        train_checkpoint=job.train_checkpoint_path, job=job,
    )
    info = mgr.policy_info(job.policy_name)
    print(f"\n[rugiar] '{job.policy_name}' ready — {info['iterations_done']} iterations, "
          f"{info['elapsed_s']}s, checkpoint at {final_checkpoint}")
    final_metrics = info.get("metrics", {}).get("final")
    if final_metrics:
        print(f"[rugiar] final: {final_metrics}")
    return 0


def main(argv: Optional[list] = None) -> None:
    parser, subparsers = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        sys.exit(run_train(args, subparsers["train"]))
    elif args.command == "order":
        sys.exit(run_order(args, subparsers["order"]))
    elif args.command == "fuse":
        sys.exit(run_fuse(args, subparsers["fuse"]))
    elif args.command == "distill":
        sys.exit(run_distill(args, subparsers["distill"]))
    elif args.command == "drive":
        sys.exit(run_drive(args, subparsers["drive"]))
    elif args.command == "stop":
        sys.exit(run_stop(args, subparsers["stop"]))
    else:  # pragma: no cover - argparse's required=True already prevents this
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
