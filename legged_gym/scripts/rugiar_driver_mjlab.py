"""
Third sibling driver: the same policy-switching / pause-restart / E-STOP /
live-telemetry control session as legged_gym/scripts/rugiar_driver.py, but
driving an **mjlab** (MuJoCo Warp) task instead of a Genesis one — phase 5
of docs/mjlab_migration.md.

Everything above the adapter is reused UNCHANGED: ControlService,
ControlServer (the same JSON-over-WebSocket protocol at /ws), the same
web/index.html UI on the same --control_port, PolicySupervisor's cross-fade
switching, SafetyGovernor's fall/NaN trip. The only genuinely new code is:
env construction (mjlab's registry + our own Rugiar-G1-Mimic task from the
repo-root mjlab_tasks/ package), MjlabAdapter
(legged_gym/control/mjlab_adapter.py), and the viewer bridge below.

Two things differ from its Genesis siblings, both structural:

1. **It runs in a different interpreter.** mjlab lives in `.venv-mjlab`
   (incompatible mujoco pin, plus an `rsl_rl` name collision with this
   repo's vendored copy — docs/mjlab_migration.md R1). Hence the sys.path
   reorder at the top of this file, and hence _relaunch_for_family() in
   BOTH directions picking an interpreter, not just a script.

2. **The viewer owns the loop, not us.** mjlab ships its own viser viewer
   (mjlab.viewer.viser.ViserPlayViewer — verified by reading
   .venv-mjlab/.../mjlab/viewer/viser/viewer.py; it takes an optional
   `viser_server`, which is how this file pins it to --viser_port), and
   that viewer's `run()` IS the sim loop: it calls `policy(obs)` and then
   `env.step()` itself, with its own real-time budget accumulator. So
   instead of our own `while True`, the per-tick control work
   (drain_commands -> service.tick -> publish_status) lives in the policy
   callback below — which the viewer calls on its own main thread, so
   ARCHITECTURE.md's hard invariant ("every ControlService call except the
   four cheap flag-setters must go through ControlServer's command queue,
   drained once per sim tick") is satisfied exactly as it is in
   rugiar_driver.py.

Reference-motion ghost overlay: mjlab renders it natively
(MotionCommandCfg.debug_vis + MjlabViserScene's debug visualization, on by
default in ViserPlayViewer.setup()) — nothing is reimplemented here.

NOT covered by tests/test_driver_family_parity.py, on purpose — see that
file's MJLAB_EXEMPT_DRIVER note: this driver shares no framework with its
Genesis siblings, so a text-identical-bodies check would be meaningless.

Usage:
    SIMULATOR=mjlab CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python \
        legged_gym/scripts/rugiar_driver_mjlab.py --control_port 9017
"""
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])

# R1 (docs/mjlab_migration.md): the repo root has to be ON sys.path (that's
# the only place `mjlab_tasks` and `legged_gym` live) but LAST, so PyPI
# rsl-rl-lib in .venv-mjlab/site-packages wins over this repo's vendored
# top-level rsl_rl/ package. Same reorder tests/conftest.py applies for the
# test session. Must run before any mjlab/rsl_rl/legged_gym import.
sys.path = [p for p in sys.path if p not in ("", ".", REPO_ROOT)] + [REPO_ROOT]

# legged_gym/__init__.py refuses to import without this, and 'mjlab' is the
# value that makes it skip the Genesis/Isaac import entirely (there is no
# Genesis in this venv, by design). setdefault, not assignment, so an
# explicitly-set value still surfaces as the error it should be.
os.environ.setdefault("SIMULATOR", "mjlab")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from typing import Optional  # noqa: E402

import torch  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import mjlab_tasks  # noqa: F401,E402 - import side effect: registers Rugiar-G1-Mimic
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks import registry  # noqa: E402

from legged_gym.control import (  # noqa: E402
    PolicySupervisor, SafetyGovernor, ControlService,
    load_policy, damping_policy, TrainingManager,
)
from legged_gym.control.mjlab_adapter import MjlabAdapter  # noqa: E402
from legged_gym.control.transport import ControlServer  # noqa: E402

DEFAULT_TASK = "Rugiar-G1-Mimic"
DEFAULT_MOTION = "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"


def parse_policy_args(policy_args):
    """--policy name:/path/to/checkpoint.onnx, repeatable. Same spec format
    as rugiar_driver.py's (kept identical so the two drivers' command lines
    read the same), but deliberately not shared code — see this module's
    docstring on the parity exemption."""
    policies = {}
    for spec in policy_args:
        name, path = spec.split(":", 1)
        policies[name] = path
    return policies


def _sibling_meta_task(checkpoint_path: str) -> Optional[str]:
    """The `task` field of the policies/<name>/meta.json sitting next to
    `checkpoint_path`, or None if there's no readable one. Same guard
    rugiar_driver.py applies to an explicit --policy: a checkpoint trained
    for a different task must not be silently loaded here just because its
    obs/action sizes happen to line up."""
    meta_path = os.path.join(os.path.dirname(checkpoint_path), "meta.json")
    try:
        with open(meta_path) as f:
            return json.load(f).get("task")
    except (OSError, ValueError):
        return None


def _script_for_task(task: str) -> str:
    """Which driver script implements `task`'s family, from THIS process's
    point of view. mjlab tasks come from mjlab's own registry; anything
    else is a Genesis/Isaac task belonging to one of the two legged_gym
    drivers. This process cannot tell g1 from g1_target (that needs
    legged_gym's task_registry, which needs Genesis, which is not installed
    in .venv-mjlab), so it hands every non-mjlab task to rugiar_driver.py —
    whose OWN _script_for_task() then re-dispatches to
    rugiar_driver_target.py if the task turns out to be target-aware. One
    extra ~15s relaunch in that one case, versus importing Genesis here,
    which simply isn't possible."""
    if task in registry.list_tasks():
        return "rugiar_driver_mjlab.py"
    return "rugiar_driver.py"


def _relaunch_for_family(cli: argparse.Namespace, new_task: str) -> None:
    """Self-relaunch for a different family, mirroring rugiar_driver.py's
    function of the same name (see its docstring for the full why: the new
    process rebinds the same --control_port/--viser_port, and the browser's
    WS client reconnects on its own). Picks the INTERPRETER as well as the
    script: leaving mjlab means switching back to the main `.venv` (with
    SIMULATOR=genesis), since neither venv can import the other's
    simulator. Returns without exiting if that venv is missing, leaving
    this session alive rather than killing it for a switch that can't
    work."""
    script_name = _script_for_task(new_task)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    env = os.environ.copy()
    if script_name == "rugiar_driver_mjlab.py":
        interpreter = sys.executable
    else:
        interpreter = os.path.join(REPO_ROOT, ".venv", "bin", "python")
        if not os.path.exists(interpreter):
            print(f"[family switch] cannot switch to {new_task!r}: no Genesis venv at {interpreter} "
                  f"-- staying on the current family.")
            return
        env["SIMULATOR"] = "genesis"
    argv = [interpreter, script, "--task", new_task,
            "--viser_port", str(cli.viser_port), "--ramp_ticks", str(cli.ramp_ticks)]
    if cli.control_port is not None:
        argv += ["--control_port", str(cli.control_port)]
    if cli.token:
        argv += ["--token", cli.token]
    if new_task == "g1" and script_name != "rugiar_driver_mjlab.py":
        # g1's own default: land in the ball+camera demo setup, same default
        # rugiar_driver.py's own _relaunch_for_family() applies -- see there.
        argv += ["--ball", "--camera"]
    print(f"[family switch] relaunching for task {new_task!r}: {' '.join(argv)}")
    sys.stdout.flush()  # os._exit() below skips normal interpreter cleanup, which would
    sys.stderr.flush()  # otherwise silently drop this line when stdout is a redirected file
    subprocess.Popen(argv, start_new_session=True, env=env)
    os._exit(0)  # immediate -- release the port now, no cleanup needed


def build_env(task: str, motion_file: Optional[str], device: str, debug_vis: bool):
    """One mjlab env, num_envs=1, built from the registered task's PLAY cfg
    (same construction tests/test_javier_checkpoints_track.py validated the
    checkpoints against in phase 4). `debug_vis` is mjlab's own native
    reference-motion ghost overlay — on for a viewer session, off headless
    (nothing to draw into)."""
    cfg = registry.load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    if "motion" in cfg.commands:
        if motion_file is not None:
            cfg.commands["motion"].motion_file = motion_file
        cfg.commands["motion"].debug_vis = debug_vis
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Policy-switching control session on an mjlab task (see legged_gym/control/)")
    parser.add_argument('--policy', action='append', default=[], dest='policy_specs',
                        help="name:/path/to/checkpoint.onnx — repeatable, first one is the default "
                             "active. Optional: every policies/<name>/ folder whose meta.json names "
                             "--task is auto-discovered regardless.")
    parser.add_argument('--task', type=str, default=DEFAULT_TASK,
                        help="registered mjlab task id (see mjlab_tasks/) this session is built for.")
    parser.add_argument('--active', type=str, default=None,
                        help="which --policy name starts active (default: first one given)")
    parser.add_argument('--ramp_ticks', type=int, default=15,
                        help="control ticks to cross-fade over on a switch")
    parser.add_argument('--headless', action='store_true', default=False,
                        help="no viewer at all — runs a scripted smoke test (switch once, then exit), "
                             "same shape as rugiar_driver.py's --headless")
    parser.add_argument('--viser_port', type=int, default=9006,
                        help="port for mjlab's own viser viewer (the 3D scene, including its native "
                             "reference-motion ghost overlay). Same flag/default as rugiar_driver.py.")
    parser.add_argument('--control_port', type=int, default=None,
                        help="if set, starts the ControlServer (JSON-over-WebSocket at /ws) on this "
                             "port and, unless --headless, serves the same unified control web "
                             "(web/index.html) at http://localhost:<control_port>/ that the Genesis "
                             "drivers serve.")
    parser.add_argument('--motion_file', type=str, default=DEFAULT_MOTION,
                        help="reference motion .npz the tracking task follows (mjlab format — see "
                             "legged_gym/scripts/process_reference_motion_mjlab.py). One clip per "
                             "session: switching dances means switching policies (R8).")
    parser.add_argument('--device', type=str, default='cpu',
                        help="torch/mujoco-warp device. CPU on macOS (evaluation-only there, R5).")
    parser.add_argument('--token', type=str, default=None,
                        help="shared secret required on every /ws connection (?token=...), same as "
                             "rugiar_driver.py's.")
    cli = parser.parse_args()

    env = build_env(cli.task, cli.motion_file, cli.device, debug_vis=not cli.headless)
    adapter = MjlabAdapter(env)
    num_obs = adapter.get_observations().shape[-1]
    num_actions = env.action_manager.total_action_dim
    print(f"[rugiar_driver_mjlab] task={cli.task!r} obs={num_obs} actions={num_actions} "
          f"motion={cli.motion_file}")

    training = TrainingManager()

    policy_paths = parse_policy_args(cli.policy_specs)
    discovered = training.discover_local_policies(exclude=policy_paths.keys())
    for name, info in discovered.items():
        if info["task"] != cli.task:
            continue  # a different task's obs/action space — see rugiar_driver.py's same filter
        policy_paths[name] = info["checkpoint"]

    if not policy_paths:
        raise ValueError(
            f"no policies to load: no --policy specs given, and no local policies/<name>/ folder is "
            f"registered for task {cli.task!r} (checked ./policies/)."
        )
    active_name = cli.active or next(iter(policy_paths))

    print("Loading policies:")
    policies = {}
    for name, path in policy_paths.items():
        checkpoint_task = _sibling_meta_task(path)
        if checkpoint_task is not None and checkpoint_task != cli.task:
            raise ValueError(
                f"--policy '{name}' ({path}) was trained for task '{checkpoint_task}', but this "
                f"server is running --task {cli.task!r}.")
        # hidden_size is irrelevant for these (Javier's are single-input
        # stateless ONNX — see policy.py's OnnxStatelessPolicy) but the
        # signature wants one; 0 keeps it honest rather than pretending
        # there's an LSTM here.
        policies[name] = load_policy(name, path, num_obs=num_obs, hidden_size=0,
                                     num_envs=adapter.num_envs)
        print(f"  '{name}' <- {path}")
    policies["damping"] = damping_policy(adapter.num_envs, num_actions)

    supervisor = PolicySupervisor(policies, active=active_name, ramp_ticks=cli.ramp_ticks)
    safety = SafetyGovernor(supervisor, damping_policy_name="damping")

    for name, path in policy_paths.items():
        info = discovered.get(name, {})
        training.register_source(name, task=cli.task, checkpoint=path,
                                 train_checkpoint=info.get("train_checkpoint"),
                                 simulator=info.get("simulator", "mjlab"),
                                 category=info.get("category"))

    def _load_policy_for_refresh(name, path, task):
        if task != cli.task:
            raise ValueError(f"'{name}' is task '{task}', this server is running '{cli.task}'")
        return load_policy(name, path, num_obs=num_obs, hidden_size=0, num_envs=adapter.num_envs,
                           description="Rediscovered from disk (refresh)")

    service = ControlService(adapter, supervisor, safety, selector=None, training=training,
                             policy_loader=_load_policy_for_refresh, task_name=cli.task)

    control_server = None
    if cli.control_port is not None:
        control_server = ControlServer(service, port=cli.control_port, token=cli.token)

    if not cli.headless and control_server is not None:
        repo_root = Path(REPO_ROOT)

        @control_server.app.get("/config")
        def _web_config():
            return {
                "viser_port": cli.viser_port,
                "camera_enabled": False,
                # No command_ranges: a tracking task has no velocity command
                # envelope at all (see MjlabAdapter's docstring). app.js
                # leaves its own defaults alone when this is absent/null, and
                # hides the Command panel outright off status()'s missing
                # `command` key — see applyStatus().
                "command_ranges": None,
            }

        control_server.app.mount(
            "/docs", StaticFiles(directory=str(repo_root / "docs"), html=True), name="docs",
        )
        control_server.app.mount(
            "/", StaticFiles(directory=str(repo_root / "web"), html=True), name="web",
        )

    if control_server is not None:
        control_server.serve_in_thread()
        listening_at = f"ControlServer listening at ws://localhost:{cli.control_port}/ws"
        if not cli.headless:
            listening_at += f" — unified control web at http://localhost:{cli.control_port}/"
        print(listening_at)

    zero_action = torch.zeros(adapter.num_envs, num_actions, device=cli.device)

    def run_headless_smoke_test():
        """No viewer at all: request one switch partway through, purely to
        prove the mechanism works end-to-end without a browser attached —
        deliberately the same shape (and step budget) as
        rugiar_driver.py's own run_headless_smoke_test()."""
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
                print(f"step {i:3d} | active={service.status()['active']} "
                      f"pending={service.status()['pending']} tripped={service.status()['safety_tripped']}")
        print("Headless smoke test done.")

    if cli.headless:
        try:
            run_headless_smoke_test()
        finally:
            env.close()
        return

    import viser  # noqa: E402 - only needed for the viewer path
    from mjlab.viewer.viser.viewer import ViserPlayViewer  # noqa: E402

    viser_server = viser.ViserServer(port=cli.viser_port, label="rugiar-mjlab")
    print(f"Viser web viewer started at http://localhost:{cli.viser_port}")

    viewer = None

    def control_tick(obs) -> torch.Tensor:
        """The viewer's policy callback — this IS our per-tick control loop
        (see this module's docstring). Runs on the viewer's main loop
        thread, once per env.step(), so every ControlService call here is
        on the sim thread, per legged_gym/control/ARCHITECTURE.md."""
        if control_server is not None:
            control_server.drain_commands()

        if service.restart_requested:
            service.restart_requested = False
            safety.reset()
            # Queued, not immediate: we're INSIDE the viewer's step, whose
            # caller is about to apply whatever we return. BaseViewer drains
            # its action queue at the top of the next tick, so the reset
            # lands cleanly between steps.
            if viewer is not None:
                viewer.request_reset()

        if service.family_switch_requested is not None:
            requested_task = service.family_switch_requested
            service.family_switch_requested = None
            _relaunch_for_family(cli, requested_task)  # normally never returns

        action = service.tick(obs[MjlabAdapter.OBS_GROUP])
        if control_server is not None:
            control_server.publish_status(service.status())
        # tick() returns None while paused — a zero action is exactly the
        # damping skill's own output (hold the default pose against the PD
        # controller), which is the right thing to feed a viewer that is
        # going to step the physics regardless.
        return zero_action if action is None else action

    viewer = ViserPlayViewer(env, control_tick, viser_server=viser_server)

    token_qs = f"?token={cli.token}" if cli.token else ""
    if control_server is not None:
        print(f"\nOpen http://localhost:{cli.control_port}{token_qs} — switch policies, pause/restart, "
              f"E-STOP. {cli.viser_port} is the raw 3D view.")
    try:
        viewer.run()
    finally:
        env.close()


if __name__ == '__main__':
    main()
