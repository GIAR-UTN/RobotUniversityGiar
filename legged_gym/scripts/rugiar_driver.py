"""
Policy-switching demo, built on legged_gym/control/ (SimAdapter, PolicySupervisor,
SafetyGovernor, ControlService — see that package's README for the full
design writeup).

This script is the "supervised from a web UI, in simulation" corner of the
control architecture. All robot control (policy switching, pause/restart,
E-STOP, velocity commands) lives in the unified control web (web/index.html,
served at --control_port) via ControlService/ControlServer — the same
methods an autonomous Selector loop or, eventually, a networked bridge to a
real robot would call. viser here is ONLY the 3D scene renderer plus its own
native camera controls — it has no robot-control GUI of its own; that would
just be a second, unsynchronized copy of what the unified web already does.

This is the driver for the "g1" (walking) family of tasks — a registered task
here is treated as a separate EXPERIMENT, not a live mode to hot-swap within
one process (see the "Family selector" plan and its follow-up discussion for
why: Genesis can't rebuild its scene in-process, and more importantly, the
user explicitly wants experiments kept architecturally separate, not unified
into one policy). `legged_gym/scripts/rugiar_driver_gaze.py` is the sibling
driver for the "target-aware" family (g1_gaze and future siblings) — same
plumbing, plus the per-tick target-bearing injection that family's tasks
expect. The control web's Family panel switches between them by relaunching
the correct one for the chosen task — see _relaunch_for_family()/
_script_for_task() below.

Usage:
    python legged_gym/scripts/rugiar_driver.py \
        --policy stable:/path/to/unitree_rl_gym/deploy/pre_train/g1/motion.pt \
        --policy crouch:logs/g1/<run>/exported/policy_lstm_1.pt \
        --active stable
"""
import argparse
import glob
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi.staticfiles import StaticFiles
from PIL import Image

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.viser_viewer import create_viser_viewer
from legged_gym.utils.props import default_ball_prop

from legged_gym.control import (
    SimAdapter, PolicySupervisor, SafetyGovernor, ControlService,
    load_policy, damping_policy, TrainingManager,
)
from legged_gym.control.transport import ControlServer


def _encode_camera_frame_jpeg(frame) -> bytes:
    """frame: (H, W, 3) uint8 RGB, as returned by RobotAdapter.get_camera_frame().
    JPEG (not PNG) — this is a live preview streamed at ~12Hz (see
    ControlServer._mjpeg_stream), not an archival asset."""
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def parse_policy_args(policy_args):
    """--policy name:/path/to/file.pt, repeatable."""
    policies = {}
    for spec in policy_args:
        name, path = spec.split(":", 1)
        policies[name] = path
    return policies


def _sibling_meta_simulator(checkpoint_path: str) -> str:
    """A --policy name:path spec is only ever cross-referenced against
    discover_local_policies() when `name` wasn't ALSO given on the command
    line (see main()'s `discovered = ...(exclude=policy_paths.keys())`) —
    right for train_checkpoint (a raw file the caller might not want auto-
    linked), wrong for `simulator`: when `checkpoint_path` sits inside a
    self-contained policies/<name>/ folder (finalize_policy()'s own
    convention), its meta.json already has the real answer sitting right
    there. Defaulting to "genesis" instead — the previous behavior — was
    simply wrong for anything actually trained on Kaggle (isaacgym), and fed
    a false "sources were trained on different simulators" warning into the
    Fuse policies panel for any such policy. Returns "genesis" (the
    long-standing default) if there's no sibling meta.json to read, or it
    doesn't parse — this must never crash startup over a missing/malformed
    file."""
    meta_path = os.path.join(os.path.dirname(checkpoint_path), "meta.json")
    try:
        with open(meta_path) as f:
            return json.load(f).get("simulator", "genesis")
    except (OSError, ValueError):
        return "genesis"


def _script_for_task(task: str) -> str:
    """Which driver script implements `task`'s family -- rugiar_driver.py
    (this file, the default) for ordinary tasks, or rugiar_driver_gaze.py
    for the "target-aware" family (any task with cfg.rewards.target_aware =
    True, e.g. g1_gaze and future siblings). Dynamic (inspects the task's
    own cfg) rather than a hardcoded task-name list, so a new target-aware
    sibling task works here with no change to this function."""
    env_cfg, _ = task_registry.get_cfgs(name=task)
    if getattr(env_cfg.rewards, "target_aware", False):
        return "rugiar_driver_gaze.py"
    return "rugiar_driver.py"


def _bare_g1_policy_specs() -> list:
    """--policy specs for legacy checkpoints sitting directly in ./policies/
    (e.g. stable.pt, g1_crouch_stability.pt) -- NOT inside a policies/<name>/
    folder, so discover_local_policies() (which only scans those folders'
    meta.json) never finds them; they were only ever loadable via an
    explicit --policy at manual launch. Same auto-pickup convention
    docker-entrypoint.sh already uses for its own automatic launch. Only
    offered for --task g1: these predate the multi-task system entirely (all
    pretrained/legacy G1 checkpoints) and, unlike folder-based policies, have
    no sibling meta.json to check a task against -- g1_gaze's obs size
    happens to coincide with g1's (see _sibling_meta_task's docstring on why
    that coincidence is exactly the dangerous case), so blindly offering
    these to every family would risk a silent wrong-shape load for a
    NON-'g1' target instead of a safe no-op.

    Only *.pt, not *.onnx: unlike a self-contained torch checkpoint, an ONNX
    export can reference an external *.onnx.data weights file by relative
    name baked into the file itself at export time (see export_onnx()) --
    copying just the .onnx without its sibling .data breaks it, and this
    repo already has exactly that stale/incomplete case sitting in
    ./policies/ (confirmed: g1_crouch_stability.onnx references a
    policy_lstm_1.onnx.data that only exists under logs/.../exported/, not
    alongside it). One broken auto-picked-up file would crash the entire
    relaunch (main()'s policy-loading loop has no per-file try/except, by
    design -- an explicitly-requested --policy failing IS supposed to be a
    hard error). Safer to just not auto-offer the fragile format at all."""
    specs = []
    for path in sorted(glob.glob(os.path.join("policies", "*.pt"))):
        specs += ["--policy", f"{Path(path).stem}:{path}"]
    return specs


def _relaunch_for_family(cli: argparse.Namespace, new_task: str, adapter=None) -> None:
    """Self-relaunch: spawns a fresh driver process for `new_task` (no
    explicit --policy beyond legacy bare-file ones for 'g1', see
    _bare_g1_policy_specs() -- the new process auto-discovers that task's
    folder-based local policies itself, same mechanism this one used at its
    own startup), then exits THIS process so the new one can bind the same
    --control_port/--viser_port. See ControlService.switch_family()'s
    docstring for why this is a process-level operation rather than an
    in-process scene rebuild (Genesis's global, once-per-process simulator
    state). The browser's own WS client already retries the connection
    unconditionally every 1s on drop -- no proxy or coordination needed, it
    just reconnects once the new process is listening (~15-20s of Genesis
    startup, same as any fresh launch). `new_task`'s family may be a
    DIFFERENT script than this one (see _script_for_task()) -- each
    family/experiment has its own driver, deliberately not sharing this
    file's plumbing (see this file's module docstring).

    `adapter` (the CURRENT session's, if any -- optional so a test/smoke
    caller can omit it) is read for its LIVE operator_speed_limit -- an
    operator who already dialed this down mid-session should stay at that
    same limit after switching families, not silently snap back to
    whatever --cruise_limit this process happened to be launched with."""
    script = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), _script_for_task(new_task))
    cruise_limit = getattr(adapter, "operator_speed_limit", cli.cruise_limit)
    argv = [sys.executable, script, "--task", new_task,
            "--viser_port", str(cli.viser_port), "--speed", str(cli.speed),
            "--ramp_ticks", str(cli.ramp_ticks), "--cruise_limit", str(cruise_limit)]
    if new_task == "g1":
        argv += _bare_g1_policy_specs()
    if cli.control_port is not None:
        argv += ["--control_port", str(cli.control_port)]
    if cli.ball:
        argv.append("--ball")
    if cli.camera:
        argv.append("--camera")
    if cli.real:
        argv.append("--real")
        argv += ["--net_interface", cli.net_interface, "--robot_config", cli.robot_config]
    if cli.token:
        argv += ["--token", cli.token]
    print(f"[family switch] relaunching for task {new_task!r}: {' '.join(argv)}")
    sys.stdout.flush()  # os._exit() below skips normal interpreter cleanup, which would
    sys.stderr.flush()  # otherwise silently drop this line when stdout is a redirected file
    subprocess.Popen(argv, start_new_session=True)
    os._exit(0)  # immediate -- release the port now, no cleanup needed


def _sibling_meta_task(checkpoint_path: str) -> Optional[str]:
    """Same sibling-meta.json lookup as _sibling_meta_simulator(), for the
    'task' field. Unlike discover_local_policies() (only consulted for
    policies NOT named on the command line), an explicit `--policy name:path`
    is loaded unconditionally with no task check at all -- silently building
    a network shaped for THIS server's task out of a checkpoint trained for
    a DIFFERENT one whenever their obs/action sizes happen to coincide (they
    don't have to differ just because the tasks do). Returns None if there's
    no sibling meta.json, it doesn't parse, or it has no 'task' key -- a
    bare checkpoint path with no policies/<name>/ folder (e.g. a raw
    './policies/foo.pt') has nothing to check against, and that must not
    crash startup."""
    meta_path = os.path.join(os.path.dirname(checkpoint_path), "meta.json")
    try:
        with open(meta_path) as f:
            return json.load(f).get("task")
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Policy-switching demo on G1 (see legged_gym/control/)")
    parser.add_argument('--policy', action='append', default=[], dest='policy_specs',
                         help="name:/path/to/policy_lstm_1.pt — repeatable, first one is the default active. "
                              "Optional: any local policies/<name>/ folder trained for --task is auto-discovered "
                              "regardless (see discover_local_policies() below) — this is only for policies not "
                              "already registered that way (e.g. unitree_rl_gym's own pretrained checkpoints).")
    parser.add_argument('--task', type=str, default='g1',
                         help="registered task this server's Genesis scene (and every --policy's "
                              "obs/action space) is built for — e.g. 'g1' (walking) or 'g1_gaze'. "
                              "All --policy specs must have been trained on this same task.")
    parser.add_argument('--active', type=str, default=None, help="which --policy name starts active (default: first one given)")
    parser.add_argument('--ramp_ticks', type=int, default=15, help="control ticks to cross-fade over on a switch")
    parser.add_argument('--headless', action='store_true', default=False,
                         help="no viewer at all — runs a scripted smoke test (switch once, then exit)")
    parser.add_argument('--viser_port', type=int, default=9006,
                         help="Genesis's native viewer has a rasterizer indexing bug on this Mac/asset "
                              "combo — viser (web viewer) is the reliable way to actually watch this run.")
    parser.add_argument('--speed', type=float, default=0.35,
                         help="playback speed multiplier (1.0 = real-time 50Hz control rate)")
    parser.add_argument('--cruise_limit', type=float, default=1.0,
                         help="startup default for ControlService.set_operator_speed_limit() — caps every "
                              "set_command (web UI, examples/joystick_controller.py, any client) to this "
                              "fraction of the trained cfg.commands.ranges envelope. 1.0 (default) is the "
                              "full trained range, this repo's and upstream unitree_rl_gym's own "
                              "community-standard command envelope — pass e.g. 0.7 to start a session more "
                              "conservative than that. Live-adjustable afterward from the web UI's Command "
                              "panel or over the wire; this flag only sets where the session STARTS.")
    parser.add_argument('--control_port', type=int, default=None,
                         help="if set, starts a networked ControlServer (JSON-over-WebSocket at /ws, see "
                              "legged_gym/control/transport.py) on this port, exposing request_switch/"
                              "status/pause/resume/estop/restart/set_command/set_random_events to external "
                              "clients. Unless --headless, this port also serves the unified control web "
                              "(web/index.html: Docs/Simulator tabs + persistent controls panel + keyboard "
                              "shortcuts + a Stimuli panel for manual velocity commands) at "
                              "http://localhost:<control_port>/.")
    parser.add_argument('--ball', action='store_true', default=False,
                         help="spawn a physics-enabled ball prop next to the robot (Genesis only, for now)")
    parser.add_argument('--camera', action='store_true', default=False,
                         help="stream a robot-POV RGB camera feed to the control web at /camera.mjpg (Genesis "
                              "sim only, for now — see GenesisSimulator.get_camera_frame() and "
                              "legged_robot_config.py's sensor.rgb_camera_config). Incompatible with "
                              "--headless (no camera is built in that mode, and there's no web UI to stream to).")
    parser.add_argument('--real', action='store_true', default=False,
                         help="drive an actual robot over DDS (deploy_real/real_adapter.py::RealAdapter) "
                              "instead of the Genesis simulator. No Genesis env, no viser — see "
                              "docs/index.html §13 for the first-boot checklist. Incompatible with "
                              "--headless (a real robot's reset() blocks on a human at the physical remote; "
                              "there is no unattended smoke test).")
    parser.add_argument('--net_interface', type=str, default=None,
                         help="network interface DDS should bind to on the robot's onboard computer "
                              "(e.g. 'eth0', 'enp3s0') — required with --real.")
    parser.add_argument('--robot_config', type=str, default=None,
                         help="path to a deploy_real/configs/*.yaml (see g1.yaml) — required with --real.")
    parser.add_argument('--token', type=str, default=None,
                         help="shared-secret required on every /ws connection (query param ?token=...), "
                              "including the web UI, which forwards its own page's ?token=... — see "
                              "legged_gym/control/transport.py and docs/index.html §13. Strongly "
                              "recommended whenever --control_port is reachable from more than localhost, "
                              "which --real always is (the robot's own WiFi/LAN).")
    cli = parser.parse_args()

    if cli.real and cli.headless:
        raise ValueError("--real and --headless are mutually exclusive — a real robot's reset() blocks on "
                          "a human at the physical remote control, so there is no unattended smoke test.")
    if cli.camera and cli.headless:
        raise ValueError("--camera and --headless are mutually exclusive — no camera is built in headless "
                          "mode (see GenesisSimulator._create_envs()) and there's no web UI to stream to.")
    if cli.camera and cli.real:
        raise ValueError("--camera isn't wired up for --real yet — RealAdapter.get_camera_frame() always "
                          "returns None (see deploy_real/real_adapter.py).")
    if cli.real and not cli.net_interface:
        raise ValueError("--real requires --net_interface (the DDS network interface, e.g. 'eth0')")
    if cli.real and not cli.robot_config:
        raise ValueError("--real requires --robot_config (e.g. deploy_real/configs/g1.yaml)")

    policy_paths = parse_policy_args(cli.policy_specs)
    # active_name is resolved after local-policy discovery below (once
    # policy_paths is known to be non-empty) -- --policy is optional now,
    # see that argument's help text.

    # unitree_rl_gym's own pretrained checkpoints store hidden_state/cell_state as a fixed
    # (1, 1, 64) buffer -> batch size must be exactly 1 to use them, so num_envs=1 throughout.
    args = argparse.Namespace(
        task=cli.task, headless=True, cpu=True, num_envs=1, max_iterations=None,
        resume=False, sync_wandb=False, export_onnx=False, debug=False, load_run=None,
        ckpt=-1, use_joystick=False, joystick_type='xbox', follow_robot=False,
        viewer='viser', viser_port=cli.viser_port, motion_file=None, motion_out_dir=None,
        num_student=None,
    )

    env = None
    if cli.real:
        # No Genesis at all in this branch — deploy_real/real_adapter.py's
        # RealAdapter is the only thing that talks to the robot, over DDS.
        # env_cfg is still needed below (num_observations/num_actions for
        # load_policy, cfg.commands.ranges for the /config route) — get_cfgs
        # only returns cfg classes, it doesn't build a simulator.
        env_cfg, _ = task_registry.get_cfgs(name=args.task)

        from deploy_real.config import RobotConfig
        from deploy_real.real_adapter import RealAdapter
        robot_config = RobotConfig(cli.robot_config)
        adapter = RealAdapter(robot_config, cli.net_interface)
    else:
        if SIMULATOR == "genesis":
            backend = os.environ.get("GENESIS_BACKEND", "cpu").lower()
            if backend == "cuda":
                try:
                    gs.init(backend=gs.cuda, logging_level='warning')
                    print("Genesis initialised with CUDA backend.")
                except Exception as e:
                    print(f"Warning: CUDA backend failed ({e}), falling back to CPU.")
                    gs.init(backend=gs.cpu, logging_level='warning')
            else:
                gs.init(backend=gs.cpu, logging_level='warning')

        env_cfg, _ = task_registry.get_cfgs(name=args.task)
        if cli.ball:
            env_cfg.props.list = [default_ball_prop()]
        if cli.camera:
            env_cfg.sensor.add_rgb_camera = True

        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        adapter = SimAdapter(env, operator_speed_limit=cli.cruise_limit)

    hidden_size = 64  # matches G1RoughCfgPPO.policy.rnn_hidden_size

    # Lets the control web's "Create Policy" panel launch new training runs
    # (as subprocesses — see legged_gym/control/training.py) and, once one
    # finishes, hot-load the result here as a new switchable policy.
    training = TrainingManager()

    # Every policies/<name>/ folder finalize_policy() ever wrote for THIS
    # task is re-offered on every startup — not just whatever --policy
    # flags were typed this time. Without this, restarting the server (to
    # pick up new code, after a crash, ...) would "lose" every policy
    # trained via the UI in a PREVIOUS process's lifetime, even though
    # finalize_policy() specifically copies their checkpoints out of
    # scratch log_dir space so they'd survive exactly this — see
    # TrainingManager.discover_local_policies()'s docstring. --policy specs
    # win on a name collision (skip via `exclude`), same as any other
    # explicit-beats-implicit default.
    discovered = training.discover_local_policies(exclude=policy_paths.keys())
    for name, info in discovered.items():
        if info["task"] != args.task:
            continue  # a different task's obs/action space — loading it here would crash load_policy()
        policy_paths[name] = info["checkpoint"]

    if not policy_paths:
        raise ValueError(
            f"no policies to load: no --policy specs given, and no local policies/<name>/ folder is "
            f"registered for task {args.task!r} (checked ./policies/) — train one first "
            f"(e.g. `rugiar train --task {args.task}`) or pass --policy explicitly."
        )
    active_name = cli.active or next(iter(policy_paths))

    print("Loading policies:")
    policies = {}
    for name, path in policy_paths.items():
        checkpoint_task = _sibling_meta_task(path)
        if checkpoint_task is not None and checkpoint_task != args.task:
            raise ValueError(
                f"--policy '{name}' ({path}) was trained for task '{checkpoint_task}', but this "
                f"server is running --task {args.task!r}. Loading it anyway would silently build a "
                f"'{args.task}' network out of '{checkpoint_task}' weights whenever their obs/action "
                f"sizes happen to coincide -- pass --task {checkpoint_task!r} instead, or drop this "
                f"--policy."
            )
        policies[name] = load_policy(name, path, num_obs=env_cfg.env.num_observations,
                                      hidden_size=hidden_size, num_envs=adapter.num_envs)
        print(f"  '{name}' <- {path}{' (rediscovered from a previous run)' if name in discovered else ''}")
    policies["damping"] = damping_policy(adapter.num_envs, env_cfg.env.num_actions)

    supervisor = PolicySupervisor(policies, active=active_name, ramp_ticks=cli.ramp_ticks)
    safety = SafetyGovernor(supervisor, damping_policy_name="damping")

    # Every policy loaded above was trained on this same task's observation
    # space, so it's registered as a "clone from" source too — rediscovered
    # ones get their train_checkpoint back as well, so Clone-from keeps
    # working across a restart, not just the checkpoint itself.
    for name, path in policy_paths.items():
        train_checkpoint = discovered.get(name, {}).get("train_checkpoint")
        simulator = discovered.get(name, {}).get("simulator") or _sibling_meta_simulator(path)
        training.register_source(name, task=args.task, checkpoint=path,
                                  train_checkpoint=train_checkpoint, simulator=simulator)

    hidden_size_for_new_policies = hidden_size  # matches G1RoughCfgPPO.policy.rnn_hidden_size (see above)

    def _load_policy_for_refresh(name, path, task):
        # Rejects a task mismatch itself (see ControlService.refresh_local_policies()'s
        # docstring) — same filter startup already applies below at line ~127,
        # just re-expressed here since a rescan can surface a policy for a
        # DIFFERENT task than this process is running (e.g. a go2 policy
        # sitting in the same ./policies/ next to this g1 server).
        if task != args.task:
            raise ValueError(f"'{name}' is task '{task}', this server is running '{args.task}'")
        return load_policy(name, path, num_obs=env_cfg.env.num_observations,
                            hidden_size=hidden_size_for_new_policies, num_envs=adapter.num_envs,
                            description="Rediscovered from disk (refresh — trained outside this server)")

    service = ControlService(adapter, supervisor, safety, selector=None, training=training,
                              policy_loader=_load_policy_for_refresh, task_name=args.task)

    def drain_finished_training():
        """Call once per sim tick. Any job TrainingManager reports done gets
        loaded and registered into the running supervisor right here — the
        same 'web layer requests, sim-loop thread executes' boundary as
        restart_requested (see ControlService.restart()'s docstring) —
        loading a torch.jit module isn't safety-relevant, but it does touch
        the same `policies` dict the control loop reads every tick, so it
        belongs on this thread, not the socket thread."""
        for job in training.poll():
            try:
                # Copies both checkpoints out of rsl_rl's log_dir into their
                # own policies/<name>/ folder and registers the result as a
                # Clone-from source — see TrainingManager.finalize_policy()'s
                # docstring. Load THAT path, not job.policy_path, so what's
                # running matches what's registered.
                final_checkpoint = training.finalize_policy(
                    job.policy_name, task=job.task, checkpoint=job.policy_path,
                    train_checkpoint=job.train_checkpoint_path, job=job,
                )
                new_policy = load_policy(
                    job.policy_name, final_checkpoint,
                    num_obs=env_cfg.env.num_observations,
                    hidden_size=hidden_size_for_new_policies, num_envs=adapter.num_envs,
                    description=f"Trained via the control web ({job.command})",
                )
                supervisor.add_policy(new_policy)
                print(f"[training] '{job.policy_name}' finished and is now selectable "
                      f"(job {job.id}, exported to {final_checkpoint})")
            except Exception as e:  # noqa: BLE001 - a bad export must not crash the sim loop
                job.status = "failed"
                job.error = f"training finished but the policy failed to load: {e}"
                print(f"[training] job {job.id} ('{job.policy_name}') failed to load: {e}")

    control_server = None
    if cli.control_port is not None:
        control_server = ControlServer(service, port=cli.control_port, token=cli.token)
        if cli.token is None and cli.real:
            print("[rugiar_driver] WARNING: --real with no --token — the control socket is reachable "
                  "unauthenticated from anything on this robot's network. Pass --token to require a "
                  "shared secret (see docs/index.html §13).")

    viser_viewer = None

    if not cli.headless and not cli.real:
        # viser has nothing to render against a real robot (no Genesis env)
        # — see module docstring on why it has no robot-control GUI of its
        # own either way.
        viser_viewer = create_viser_viewer(env, port=cli.viser_port, show_command_sliders=False)
        print(f"Viser web viewer started at http://localhost:{cli.viser_port}")
        # No robot-control GUI added here on purpose — see module docstring.
        # viser's own Camera folder (Track robot / FOV) is all that's native
        # to the viewer and stays; --show_command_sliders=False also drops
        # viser's built-in (and, in this script, never-wired) velocity
        # sliders, which duplicated the unified web's Stimuli panel.

    if not cli.headless and control_server is not None:
        # Mount the unified control web (Docs/Simulator tabs + controls
        # panel + keyboard shortcuts — web/index.html) onto the SAME
        # FastAPI app/port as the /ws transport, per HANDOFF_control_web.md
        # §3-B: one process, one port, same-origin WS (no CORS). Routes
        # must be added before serve_in_thread(). Works the same whether
        # adapter is Sim or Real — the Simulator tab just has nothing to
        # show in --real mode (no viser_viewer above).
        repo_root = Path(__file__).resolve().parents[2]

        @control_server.app.get("/config")
        def _web_config():
            # command_ranges lets the web panel clamp its velocity
            # sliders to the exact envelope this policy was trained
            # across (env_cfg.commands.ranges) — see SimAdapter.set_command
            # / RealAdapter.set_command.
            ranges = env_cfg.commands.ranges
            return {
                "viser_port": cli.viser_port if not cli.real else None,
                # Lets the web panel show/hide its Camera section — a plain
                # <img src="/camera.mjpg"> otherwise has no way to know
                # whether anything will ever be published there (see
                # ControlServer._mjpeg_stream / --camera above).
                "camera_enabled": cli.camera,
                "command_ranges": {
                    "vx": list(ranges.lin_vel_x),
                    "vy": list(ranges.lin_vel_y),
                    "yaw": list(ranges.ang_vel_yaw),
                },
            }

        control_server.app.mount(
            "/docs", StaticFiles(directory=str(repo_root / "docs"), html=True), name="docs",
        )
        control_server.app.mount(
            "/", StaticFiles(directory=str(repo_root / "web"), html=True), name="web",
        )

    if control_server is not None:
        # Routes/mounts (if any — see the `if not cli.headless` block above)
        # must already be on control_server.app before this call.
        control_server.serve_in_thread()
        listening_at = f"ControlServer listening at ws://localhost:{cli.control_port}/ws"
        if not cli.headless:
            listening_at += f" — unified control web at http://localhost:{cli.control_port}/"
        print(listening_at)

    # RealAdapter.send_action() already sleeps config.control_dt internally
    # (matching deploy_real.py's own pacing) — an extra sleep here would just
    # slow the real control loop down further. cli.speed only makes sense as
    # a sim-playback knob.
    frame_dt = 0.0 if cli.real else (1 / 60.0) / max(cli.speed, 0.01)

    def run_headless_smoke_test():
        """No web UI at all: request one switch partway through, purely to
        prove the mechanism works end-to-end without a browser attached —
        this is the shape an autonomous on-robot process would drive."""
        other = next((n for n in policy_paths if n != active_name), None)
        if other is None:
            print(f"Only one policy loaded ('{active_name}') — running without a switch.")

        obs = adapter.get_observations()
        switched = False
        for i in range(80):
            if control_server is not None:
                control_server.drain_commands()
            if i == 40 and not switched and other is not None:
                print(f"[autonomous] requesting switch to '{other}' at step {i}")
                service.request_switch(other)
                switched = True
            action = service.tick(obs)
            adapter.send_action(action)
            obs = adapter.get_observations()
            if control_server is not None:
                control_server.publish_status(service.status())
            if i % 20 == 0:
                print(f"step {i:3d} | {service.status()}")
        print("Headless smoke test done.")

    if cli.headless:
        run_headless_smoke_test()
        return

    token_qs = f"?token={cli.token}" if cli.token else ""
    if control_server is not None and not cli.headless:
        url = f"http://localhost:{cli.control_port}{token_qs}"
        extra = "" if cli.real else f" {cli.viser_port} is the raw 3D view."
        print(f"\nOpen {url} — switch policies, pause/restart, E-STOP, and drive velocity commands live.{extra}")
        if cli.real:
            print("Share this SAME URL (with the token) with anyone building a home-made controller for "
                  "this robot — see docs/index.html §13 for the raw WebSocket protocol.")
    else:
        print(f"\nOpen http://localhost:{cli.viser_port} — pass --control_port to also get "
              f"the unified control web (policy switching, pause/restart, E-STOP, velocity commands).")
    obs = adapter.get_observations()
    camera_tick = 0
    camera_decimation = env_cfg.sensor.rgb_camera_config.decimation if cli.camera else None
    while True:
        t_start = time.perf_counter()

        if control_server is not None:
            control_server.drain_commands()

        if service.restart_requested:
            service.restart_requested = False
            adapter.reset()
            obs = adapter.get_observations()
            safety.reset()
            # reset() teleports the robot back to its default pose — without
            # this, viser's camera tracking computes one huge, stale delta
            # from wherever the robot used to be (see
            # ViserViewer.resync_camera_tracking()'s docstring), the same
            # "camera not centered, have to toggle Track robot" bug reported
            # after a restart.
            if viser_viewer is not None:
                viser_viewer.resync_camera_tracking()

        if service.family_switch_requested is not None:
            _relaunch_for_family(cli, service.family_switch_requested, adapter)  # never returns

        drain_finished_training()

        action = service.tick(obs)
        if action is not None:
            adapter.send_action(action)
            obs = adapter.get_observations()
            if viser_viewer is not None:
                viser_viewer.update_from_simulator(env, 0)

        if control_server is not None:
            control_server.publish_status(service.status())

            # A live video feed doesn't need control-loop rate (~50-200Hz) —
            # capture/encode/publish only every rgb_camera_config.decimation
            # ticks. get_camera_frame() returns None on any backend/config
            # that doesn't have a camera (see RobotAdapter.get_camera_frame's
            # docstring) — nothing to publish on those ticks.
            if camera_decimation is not None:
                camera_tick += 1
                if camera_tick % camera_decimation == 0:
                    frame = adapter.get_camera_frame()
                    if frame is not None:
                        control_server.publish_camera_frame(_encode_camera_frame_jpeg(frame))

        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


if __name__ == '__main__':
    main()
