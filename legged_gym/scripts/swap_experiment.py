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

Usage:
    python legged_gym/scripts/swap_experiment.py \
        --policy stable:/path/to/unitree_rl_gym/deploy/pre_train/g1/motion.pt \
        --policy cautious:logs/g1_cautious/<run>/exported/policy_lstm_1.pt \
        --active stable
"""
import argparse
import os
import time
from pathlib import Path

from fastapi.staticfiles import StaticFiles

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


def parse_policy_args(policy_args):
    """--policy name:/path/to/file.pt, repeatable."""
    policies = {}
    for spec in policy_args:
        name, path = spec.split(":", 1)
        policies[name] = path
    return policies


def main():
    parser = argparse.ArgumentParser(description="Policy-switching demo on G1 (see legged_gym/control/)")
    parser.add_argument('--policy', action='append', required=True, dest='policy_specs',
                         help="name:/path/to/policy_lstm_1.pt — repeatable, first one is the default active")
    parser.add_argument('--active', type=str, default=None, help="which --policy name starts active (default: first one given)")
    parser.add_argument('--ramp_ticks', type=int, default=15, help="control ticks to cross-fade over on a switch")
    parser.add_argument('--headless', action='store_true', default=False,
                         help="no viewer at all — runs a scripted smoke test (switch once, then exit)")
    parser.add_argument('--viser_port', type=int, default=9006,
                         help="Genesis's native viewer has a rasterizer indexing bug on this Mac/asset "
                              "combo — viser (web viewer) is the reliable way to actually watch this run.")
    parser.add_argument('--speed', type=float, default=0.35,
                         help="playback speed multiplier (1.0 = real-time 50Hz control rate)")
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
    cli = parser.parse_args()

    policy_paths = parse_policy_args(cli.policy_specs)
    active_name = cli.active or next(iter(policy_paths))

    # unitree_rl_gym's own pretrained checkpoints store hidden_state/cell_state as a fixed
    # (1, 1, 64) buffer -> batch size must be exactly 1 to use them, so num_envs=1 throughout.
    args = argparse.Namespace(
        task="g1", headless=True, cpu=True, num_envs=1, max_iterations=None,
        resume=False, sync_wandb=False, export_onnx=False, debug=False, load_run=None,
        ckpt=-1, use_joystick=False, joystick_type='xbox', follow_robot=False,
        viewer='viser', viser_port=cli.viser_port, motion_file=None, motion_out_dir=None,
        num_student=None,
    )

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

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    adapter = SimAdapter(env)

    hidden_size = 64  # matches G1RoughCfgPPO.policy.rnn_hidden_size
    print("Loading policies:")
    policies = {}
    for name, path in policy_paths.items():
        policies[name] = load_policy(name, path, num_obs=env_cfg.env.num_observations,
                                      hidden_size=hidden_size, num_envs=env.num_envs)
        print(f"  '{name}' <- {path}")
    policies["damping"] = damping_policy(env.num_envs, env_cfg.env.num_actions)

    supervisor = PolicySupervisor(policies, active=active_name, ramp_ticks=cli.ramp_ticks)
    safety = SafetyGovernor(supervisor, damping_policy_name="damping")

    # Lets the control web's "Create Policy" panel launch new training runs
    # (as subprocesses — see legged_gym/control/training.py) and, once one
    # finishes, hot-load the result here as a new switchable policy. Every
    # policy loaded via --policy above was trained on this same task's
    # observation space, so it's registered as a "clone from" source too —
    # see the poll loop below for how a *newly* trained policy gets
    # registered the same way once it completes.
    training = TrainingManager()
    for name, path in policy_paths.items():
        training.register_source(name, task=args.task, checkpoint=path)
    service = ControlService(adapter, supervisor, safety, selector=None, training=training)

    hidden_size_for_new_policies = hidden_size  # matches G1RoughCfgPPO.policy.rnn_hidden_size (see above)

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
                    hidden_size=hidden_size_for_new_policies, num_envs=env.num_envs,
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
        control_server = ControlServer(service, port=cli.control_port)

    viser_viewer = None

    if not cli.headless:
        viser_viewer = create_viser_viewer(env, port=cli.viser_port, show_command_sliders=False)
        print(f"Viser web viewer started at http://localhost:{cli.viser_port}")
        # No robot-control GUI added here on purpose — see module docstring.
        # viser's own Camera folder (Track robot / FOV) is all that's native
        # to the viewer and stays; --show_command_sliders=False also drops
        # viser's built-in (and, in this script, never-wired) velocity
        # sliders, which duplicated the unified web's Stimuli panel.

        if control_server is not None:
            # Mount the unified control web (Docs/Simulator tabs + controls
            # panel + keyboard shortcuts — web/index.html) onto the SAME
            # FastAPI app/port as the /ws transport, per HANDOFF_control_web.md
            # §3-B: one process, one port, same-origin WS (no CORS). Routes
            # must be added before serve_in_thread().
            repo_root = Path(__file__).resolve().parents[2]

            @control_server.app.get("/config")
            def _web_config():
                # command_ranges lets the web panel clamp its velocity
                # sliders to the exact envelope this policy was trained
                # across (env_cfg.commands.ranges) — see SimAdapter.set_command.
                ranges = env_cfg.commands.ranges
                return {
                    "viser_port": cli.viser_port,
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

    frame_dt = (1 / 60.0) / max(cli.speed, 0.01)

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

    if control_server is not None and not cli.headless:
        print(f"\nOpen http://localhost:{cli.control_port} — switch policies, pause/restart, "
              f"E-STOP, and drive velocity commands live. {cli.viser_port} is the raw 3D view.")
    else:
        print(f"\nOpen http://localhost:{cli.viser_port} — pass --control_port to also get "
              f"the unified control web (policy switching, pause/restart, E-STOP, velocity commands).")
    obs = adapter.get_observations()
    while True:
        t_start = time.perf_counter()

        if control_server is not None:
            control_server.drain_commands()

        if service.restart_requested:
            service.restart_requested = False
            adapter.reset()
            obs = adapter.get_observations()
            safety.reset()

        drain_finished_training()

        action = service.tick(obs)
        if action is not None:
            adapter.send_action(action)
            obs = adapter.get_observations()
            if viser_viewer is not None:
                viser_viewer.update_from_simulator(env, 0)

        if control_server is not None:
            control_server.publish_status(service.status())

        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


if __name__ == '__main__':
    main()
