"""
ControlService — the one call surface. A viser button callback calls
`service.request_switch("cautious")`. An autonomous Selector loop calls the
exact same method. A future WebSocket/HTTP bridge for controlling a real,
UI-less robot from an external web app would also just call this method —
it would be a thin transport wrapper around this class, not a parallel
implementation.

Today this class is used in-process (see legged_gym/scripts/rugiar_driver.py):
viser's GUI callbacks call straight into it, no network hop needed for a
local sim demo. The moment you want to drive this from a *different*
process or machine (a real robot with no local display, an external web
app), wrap this same class with a tiny JSON-RPC-ish layer over WebSocket —
switch/status/pause/estop is the whole surface, per the architecture
write-up in the README.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import torch

from .adapter import RobotAdapter, Lifecycle
from .safety import SafetyGovernor
from .selector import Selector
from .supervisor import PolicySupervisor
from .training import TrainingManager


class ControlService:
    def __init__(
        self,
        adapter: RobotAdapter,
        supervisor: PolicySupervisor,
        safety: SafetyGovernor,
        selector: Optional[Selector] = None,
        training: Optional[TrainingManager] = None,
        policy_loader: Optional[Callable[[str, str, str], "Policy"]] = None,  # noqa: F821 - see refresh_local_policies()
        task_name: Optional[str] = None,
    ):
        self.adapter = adapter
        self.supervisor = supervisor
        self.safety = safety
        self.selector = selector
        self.training = training
        # Which registered task THIS process's Genesis scene was built for —
        # e.g. "g1" or "g1_gaze". Only used to answer list_families()'s
        # `current` field and to validate switch_family() requests; None on a
        # caller that doesn't pass it (e.g. RealAdapter setups, or tests)
        # simply means family-switching isn't offered.
        self.task_name = task_name
        # Set by switch_family() to the requested task name, drained (and
        # acted on) once per tick by rugiar_driver.py's main loop — same
        # "record intent, sim loop owns the actual work" split as
        # restart_requested below, except a family switch's "actual work" is
        # a process self-relaunch (see switch_family()'s docstring for why).
        self.family_switch_requested: Optional[str] = None
        # Turns a (name, checkpoint_path, task) triple into a loaded, in-process
        # Policy compatible with THIS running sim's obs/action space — see
        # refresh_local_policies() below. Lives outside ControlService/
        # TrainingManager on purpose: building a runtime Policy needs the
        # current env's obs size / hidden size / num_envs, which only
        # rugiar_driver.py's main() has in scope (same reason
        # drain_finished_training() there calls load_policy() directly
        # instead of this class doing it). None on a caller that never
        # wires one up (e.g. a future real-robot deployment with no local
        # training/refresh story) — refresh_local_policies() then no-ops.
        self.policy_loader = policy_loader
        self.paused = False
        # Restart only *records* intent, same shape as PolicySupervisor's
        # request_switch — the sim loop owns `obs` (the raw observation
        # tensor fed to policies) and must refresh it right after the reset,
        # so the actual adapter.reset()/safety.reset() calls stay in
        # rugiar_driver.py's loop rather than here. See restart()'s
        # docstring.
        self.restart_requested = False

    # ---- the "human or autonomous, same call" surface ----

    def request_switch(self, name: str) -> bool:
        return self.supervisor.request_switch(name)

    def delete_policy(self, name: str) -> None:
        """Discards a policy entirely — pulled from the switchable list AND
        the clone-from catalog, its exported checkpoint deleted from disk.
        For a training experiment that converged to something useless (the
        motivating case: a policy that never settled into a coherent gait —
        no reason to keep cluttering the panel or offering it as a
        fine-tuning base). Irreversible. Raises if `name` is 'damping', is
        currently active, or has a switch pending — see
        PolicySupervisor.remove_policy()'s docstring for why; the caller
        has to switch to something else first."""
        self.supervisor.remove_policy(name)
        if self.training is not None:
            self.training.forget_source(name)

    def refresh_local_policies(self) -> list:
        """Rescans ./policies/<name>/ for folders not yet loaded into THIS
        running process and loads/registers each one — the counterpart to
        drain_finished_training() in rugiar_driver.py, which only catches
        completions of jobs THIS process itself started. A policy trained
        by a separate process (e.g. the `rugiar` CLI, running independently
        of any server) writes straight to disk and is otherwise invisible
        here until the server restarts — see rugiar's SKILL.md
        "Picking up policies trained outside the web" for why. Returns the
        list of names actually added (empty if nothing new was found, or if
        no policy_loader/training was wired up at construction). Best-effort
        per policy — one bad/incompatible checkpoint (e.g. a different
        task's obs shape) is skipped rather than blocking the rest."""
        if self.training is None or self.policy_loader is None:
            return []
        known = set(self.supervisor.policies.keys())
        discovered = self.training.discover_local_policies(exclude=known)
        added = []
        for name, info in discovered.items():
            try:
                # policy_loader is responsible for rejecting a task mismatch
                # (a different obs/action space) itself — see
                # rugiar_driver.py's wiring; ObsSpec enforcement elsewhere
                # is only a warning, not a hard stop, so this can't rely on
                # load_policy() to catch it after the fact.
                new_policy = self.policy_loader(name, info["checkpoint"], info["task"])
            except Exception:  # noqa: BLE001 - one bad/mismatched checkpoint must not block the rest
                continue
            self.supervisor.add_policy(new_policy)
            self.training.register_source(
                name, task=info["task"], checkpoint=info["checkpoint"],
                train_checkpoint=info.get("train_checkpoint"),
                simulator=info.get("simulator", "genesis"),
                category=info.get("category"),
            )
            added.append(name)
        return added

    _NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

    def rename_policy(self, old_name: str, new_name: str) -> None:
        """Renames a policy everywhere its identity is recorded: the
        running supervisor (so it stays switchable/active under the new
        name, mid-switch or not — see PolicySupervisor.rename_policy()),
        and its `policies/<name>/` folder plus clone-from catalog entry
        (see TrainingManager.rename_policy()). Requires a dedicated
        training folder, same restriction as the delete path's
        forget_source() — `stable` and other bare --policy CLI sources
        have no folder to rename and are rejected by the training-side
        call below. Does the supervisor rename first: if the training-side
        folder rename then fails (name collision, no folder), the running
        policy is left renamed but the catalog/disk are untouched, which
        is recoverable (rename back) rather than the reverse — a folder
        renamed on disk with the supervisor still pointing at the old
        name, which would make the policy unswitchable."""
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("new name can't be empty")
        if not self._NAME_RE.match(new_name):
            raise ValueError("policy names may only contain letters, digits, '_', '-', '.'")
        if new_name == "damping":
            raise ValueError("'damping' is reserved and can't be used as a policy name")
        if new_name == old_name:
            return
        self.supervisor.rename_policy(old_name, new_name)
        if self.training is not None:
            self.training.rename_policy(old_name, new_name)

    def policy_info(self, name: str) -> Optional[dict]:
        """Backs the Policies list's "info" popup — everything known about
        how `name` came to exist: the exact command that trained it, what
        it was cloned from, entropy_coef, file paths, and (when available)
        the parsed `Mean action noise std` / `Mean reward` / `Mean episode
        length` trend plus final reward-term breakdown. Returns None rather
        than raising for a policy with no `policies/<name>/meta.json` (e.g.
        `stable`, an external checkpoint with no training history at all —
        see TrainingManager.finalize_policy()'s docstring) — the UI shows
        "no training info available" for that case instead of an error."""
        if self.training is None:
            return None
        try:
            return self.training.policy_info(name)
        except (FileNotFoundError, OSError):
            return None

    def status(self) -> dict:
        s = self.supervisor.status
        s["paused"] = self.paused
        s["safety_tripped"] = self.safety.tripped
        # Every user-selectable policy name — "damping" is the safety
        # fallback skill, not a switch target, so it's excluded the same
        # way rugiar_driver.py's viser panel already excludes it.
        s["policies"] = [name for name in self.supervisor.policies if name != "damping"]
        # Adapter-declared, not UI-hardcoded — see SimAdapter/RealAdapter's
        # backend_name/capabilities class attributes. Lets a control web
        # show the same panel for sim and real, graying out what the
        # current backend can't do (e.g. "restart" on real hardware).
        s["backend"] = getattr(self.adapter, "backend_name", "sim")
        s["current_task"] = self.task_name
        s["capabilities"] = getattr(self.adapter, "capabilities", {})
        # Optional — only SimAdapter exposes these today (see adapter.py's
        # command/random_events properties). Absent entirely for adapters
        # that don't support manual velocity/stimulus control.
        if hasattr(self.adapter, "command"):
            vx, vy, yaw = self.adapter.command
            s["command"] = {"vx": vx, "vy": vy, "yaw": yaw}
        if hasattr(self.adapter, "random_events"):
            s["random_events"] = self.adapter.random_events
        if hasattr(self.adapter, "episode_timeout_s"):
            s["episode_timeout_s"] = self.adapter.episode_timeout_s
        if hasattr(self.adapter, "operator_speed_limit"):
            s["operator_speed_limit"] = self.adapter.operator_speed_limit
        if self.training is not None:
            s["training_jobs"] = self.training.status()
        s["telemetry"] = self._telemetry()
        return s

    def _telemetry(self) -> Optional[dict]:
        """Live IMU-adjacent readout for the web panel — see RobotState's
        own field docstrings for what's a real sensor vs. simulator-only
        ground truth (surfaced verbatim in each field's 'source' below so
        the UI doesn't have to duplicate that knowledge). get_state() just
        reads already-populated tensors (no extra sim step), so this is
        cheap to call every tick. Only env 0 — rugiar_driver.py's live
        control demo always runs num_envs=1."""
        try:
            state = self.adapter.get_state()
        except Exception:  # noqa: BLE001 - a telemetry glitch must not break status()
            return None

        def scalar(t):
            return float(t[0]) if t is not None else None

        def vec3(t):
            return [float(x) for x in t[0]] if t is not None else None

        return {
            "base_height": {
                "value": scalar(state.base_height), "unit": "m", "source": "sim_ground_truth",
                "label": "Base height",
                "note": "Not measured by any real sensor — the simulator's own base z position. "
                        "Fine as a training-time target (training only ever runs in sim); would need "
                        "a different signal (e.g. leg kinematics) to ever inform a real-robot policy.",
            },
            "projected_gravity": {
                "value": vec3(state.projected_gravity), "unit": "g", "source": "sensor",
                "label": "Orientation (gravity in body frame)",
                "note": "What an IMU actually gives you: the gravity vector expressed in the robot's "
                        "own frame — upright is ~(0,0,-1); the same signal legged_robot.py's fall "
                        "detection watches.",
            },
            "base_ang_vel": {
                "value": vec3(state.base_ang_vel), "unit": "rad/s", "source": "sensor",
                "label": "Angular velocity (gyroscope)",
                "note": "Real IMU gyroscope reading — available on both sim and real hardware.",
            },
            "base_lin_vel": {
                "value": vec3(state.base_lin_vel), "unit": "m/s", "source": "sim_ground_truth",
                "label": "Linear velocity",
                "note": "Not directly sensed on real hardware (no IMU measures velocity, only "
                        "acceleration) — simulator ground truth only, None on RealAdapter.",
            },
        }

    # ---- training a new policy (see legged_gym/control/training.py) ----

    def _compatible_training_tasks(self) -> Optional[list]:
        """Only tasks whose observation space matches the currently-loaded
        policies are offered — a trained policy that doesn't match can't be
        hot-loaded into this running supervisor (PolicySupervisor requires
        one shared observation space; see supervisor.py's ObsSpec check).
        Returns None (meaning "don't filter") if the current env's obs count
        can't be determined."""
        env = getattr(self.adapter, "env", None)
        num_obs = getattr(getattr(env, "cfg", None), "env", None)
        num_obs = getattr(num_obs, "num_observations", None)
        if num_obs is None:
            return None
        from legged_gym.utils import task_registry
        compatible = []
        for name in task_registry.task_classes:
            try:
                env_cfg, _ = task_registry.get_cfgs(name)
            except Exception:  # noqa: BLE001 - a broken/unregistered cfg shouldn't break the catalog
                continue
            if getattr(env_cfg.env, "num_observations", None) == num_obs:
                compatible.append(name)
        return compatible

    def training_catalog(self) -> dict:
        """Everything the Create Policy panel needs to render its form and
        compose a command: trainable tasks compatible with this running sim,
        and every currently-loaded policy that's clonable (has a known
        checkpoint) as a fine-tuning base."""
        if self.training is None:
            raise NotImplementedError("no TrainingManager configured for this ControlService")
        return self.training.catalog(compatible_tasks=self._compatible_training_tasks())

    def start_training(self, policy_name: str, task: str, num_envs: int = 64,
                        max_iterations: Optional[int] = None, max_minutes: Optional[float] = None,
                        base_policy: Optional[str] = None,
                        cmd_vx: Optional[list] = None, cmd_vy: Optional[list] = None,
                        cmd_yaw: Optional[list] = None,
                        base_height_target: Optional[float] = None,
                        lin_vel_z_target: Optional[float] = None,
                        ang_vel_xy_target: Optional[float] = None,
                        orientation_tilt_target: Optional[float] = None,
                        push_robots: Optional[bool] = None,
                        max_push_vel_xy: Optional[float] = None,
                        push_interval_s: Optional[float] = None,
                        push_dir: Optional[str] = None,
                        entropy_coef: Optional[float] = None,
                        reward_scale_overrides: Optional[dict] = None,
                        backend: str = "local") -> str:
        """Launches a new training job; returns its job id. Training runs
        out-of-process (see TrainingManager) — this call returns immediately,
        the job's progress shows up in status()['training_jobs'], and the
        resulting policy is hot-loaded into the supervisor automatically once
        it finishes (see rugiar_driver.py's per-tick poll_finished_training()
        drain, mirroring how restart_requested is drained today).

        Time budget is either or both of max_iterations/max_minutes —
        whichever is hit first stops training (see web_train.py's chunked
        learn() loop). base_height_target/lin_vel_z_target/ang_vel_xy_target/
        orientation_tilt_target/push_robots/max_push_vel_xy/push_interval_s
        are the "stability target" knobs — override the task's own
        reward/domain-rand defaults, e.g. to hold a fine-tuning base's
        crouched height instead of the task's standing height.
        entropy_coef overrides PPO's exploration-noise bonus weight — watch
        for a run whose action noise std climbs instead of shrinks (visible
        live via web_train.py's log); that's this knob set too high for how
        weak the task's actual reward signal is, not a bug to fix elsewhere.
        reward_scale_overrides overrides any subset of the task's own
        <Cfg>.rewards.scales — see TrainingManager.task_defaults()'s
        'reward_scales' for the full set this task defines and its current
        default for each, and web_train.py's --reward_scale for the
        per-term sign convention (positive rewards more of that term,
        negative penalizes it). backend='kaggle' runs this same job on a
        Kaggle GPU kernel instead of a local CPU subprocess — see
        TrainingManager.start()/kaggle_backend.py; only available when
        system_info()['kaggle_available'] is true, and Clone-from
        (base_policy/from_checkpoint) isn't supported on it yet."""
        if self.training is None:
            raise NotImplementedError("no TrainingManager configured for this ControlService")
        return self.training.start(
            policy_name=policy_name, task=task, num_envs=num_envs,
            max_iterations=max_iterations, max_minutes=max_minutes,
            base_policy=base_policy, cmd_vx=cmd_vx, cmd_vy=cmd_vy, cmd_yaw=cmd_yaw,
            base_height_target=base_height_target,
            lin_vel_z_target=lin_vel_z_target, ang_vel_xy_target=ang_vel_xy_target,
            orientation_tilt_target=orientation_tilt_target,
            push_robots=push_robots,
            entropy_coef=entropy_coef,
            max_push_vel_xy=max_push_vel_xy, push_interval_s=push_interval_s,
            push_dir=push_dir,
            reward_scale_overrides=reward_scale_overrides,
            backend=backend,
        )

    def fuse_policies(self, names: list, out_name: str, weights: Optional[list] = None,
                       method: str = "weighted_average", export_task: Optional[str] = None) -> dict:
        """Merges 2+ already-registered local policies' weights into a new
        one — see TrainingManager.fuse_policies() for the actual merge/
        validation logic (this is a thin delegate, same style as
        start_training() above). Unlike start_training(), this runs to
        completion synchronously and returns the result directly — merging
        weights is a tensor op that takes seconds, not a multi-minute
        subprocess job, so there's no job id / polling story here.
        refresh_local_policies() afterward is what hot-loads the fused
        result into the running supervisor immediately, the same way a
        finished training job gets picked up (see poll_finished_training())
        — without it the new policy would only appear after the next
        connection/restart."""
        if self.training is None:
            raise NotImplementedError("no TrainingManager configured for this ControlService")
        result = self.training.fuse_policies(
            names, out_name, weights=weights, method=method, export_task=export_task)
        self.refresh_local_policies()
        return result

    def start_distillation(self, teacher: str, task: str, out_name: str,
                            rollout_steps: Optional[int] = None, bc_epochs: Optional[int] = None,
                            lr: Optional[float] = None, num_envs: Optional[int] = None,
                            method: str = "behavior_cloning") -> str:
        """Launches a new distillation job; returns its job id — same "runs
        out-of-process, returns immediately, progress shows up in
        status()['training_jobs'], hot-loaded automatically once it
        finishes" shape as start_training() above (see
        TrainingManager.start_distillation()'s docstring for why: a BC
        rollout+train run takes real wall-clock time, unlike fuse_policies()
        below). `teacher` may be ANY known local policy — crucially
        including ones with no train_checkpoint.pt at all (fuse_policies()
        requires one; this is the feature that doesn't), like `stable`. See
        distillation.DISTILL_METHODS (surfaced via training_catalog()'s
        `distill_methods` key) for the method options."""
        if self.training is None:
            raise NotImplementedError("no TrainingManager configured for this ControlService")
        kwargs = {}
        if rollout_steps is not None:
            kwargs["rollout_steps"] = rollout_steps
        if bc_epochs is not None:
            kwargs["bc_epochs"] = bc_epochs
        if lr is not None:
            kwargs["lr"] = lr
        if num_envs is not None:
            kwargs["num_envs"] = num_envs
        return self.training.start_distillation(teacher, task, out_name, method=method, **kwargs)

    def task_defaults(self, task: str) -> dict:
        """Reference values (e.g. the task's own default pelvis height) for
        the Create Policy panel's 'relative' target fields — see
        TrainingManager.task_defaults for what these are and aren't."""
        if self.training is None:
            raise NotImplementedError("no TrainingManager configured for this ControlService")
        return self.training.task_defaults(task)

    def system_info(self) -> dict:
        """What this server's machine actually is — CPU, RAM, GPU
        availability, simulator backend — for the web panel that shows it
        (the user explicitly didn't want to be guessing at what their
        hardware can handle)."""
        if self.training is None:
            raise NotImplementedError("no TrainingManager configured for this ControlService")
        info = self.training.system_info()
        info["control_backend"] = getattr(self.adapter, "backend_name", "sim")
        return info

    def estimate_training_time(self, num_envs: int, max_iterations: Optional[int] = None,
                                max_minutes: Optional[float] = None, backend: str = "local") -> dict:
        """(iterations, seconds) estimate for a would-be training job, from
        that BACKEND's own history of completed jobs (see
        TrainingManager.estimate — local and Kaggle are different throughput
        regimes, never pooled together) — called live as the Create Policy
        form's fields change, works whether the user filled in iterations,
        minutes, or both."""
        if self.training is None:
            raise NotImplementedError("no TrainingManager configured for this ControlService")
        return self.training.estimate(num_envs, max_iterations=max_iterations,
                                        max_minutes=max_minutes, backend=backend)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def restart(self) -> None:
        """Requests a reset to the default standing pose, keeping the
        currently active policy. Only records intent (mirrors
        PolicySupervisor.request_switch) — see __init__'s note on why the
        actual reset happens in the sim loop, not here."""
        self.restart_requested = True

    @staticmethod
    def _registered_task_names() -> list:
        """Genesis/Isaac task names from legged_gym's own task_registry —
        or an empty list when that import isn't available at all, which is
        the normal case under `.venv-mjlab` (mjlab runs from a separate venv
        with no Genesis and no vendored rsl_rl — see
        docs/mjlab_migration.md R1). An mjlab session's families then come
        entirely from the local policy catalog below, which is backend-
        agnostic (it just reads policies/<name>/meta.json)."""
        try:
            from legged_gym.utils import task_registry
        except ImportError:
            return []
        return list(task_registry.task_classes.keys())

    def _switchable_families(self) -> tuple:
        """(all task names, policies_per_task) — the union of every
        registered legged_gym task and every task some local policy was
        trained for. That union is what makes cross-BACKEND family
        switching work: `Rugiar-G1-Mimic` is an mjlab task and will never
        appear in legged_gym's task_registry, but Javier's checkpoints
        under policies/javier_mjlab_*/ declare it in their meta.json, so
        it shows up as a switchable family from a Genesis session (and,
        symmetrically, the Genesis families show up from an mjlab one).
        The driver decides which script/interpreter that task needs — see
        rugiar_driver.py's _script_for_task()."""
        policies_per_task: dict = {}
        if self.training is not None:
            for name, info in self.training.discover_local_policies().items():
                policies_per_task.setdefault(info["task"], []).append(name)
        tasks = set(self._registered_task_names()) | set(policies_per_task)
        if self.task_name is not None:
            tasks.add(self.task_name)
        return sorted(tasks), policies_per_task

    def list_families(self) -> dict:
        """Every registered task ('family'), and which local policies exist
        for each — what the control web's Family panel needs to render
        itself and know what's actually switchable-to (a task with zero
        local policies can't be switched to, see switch_family()). Unlike
        training_catalog()'s task list, this is NOT filtered to tasks whose
        obs shape matches the currently running one — the whole point here
        is offering tasks with a DIFFERENT shape, that's what a family
        switch is for."""
        tasks, policies_per_task = self._switchable_families()
        return {
            "tasks": tasks,
            "current": self.task_name,
            "policies_per_task": policies_per_task,
        }

    def switch_family(self, task: str) -> None:
        """Requests switching this ENTIRE session to a different registered
        task ('family') — e.g. from 'g1' (walking) to 'g1_gaze' (standing).
        Unlike request_switch() (swaps the active POLICY within the current
        task's already-built scene), this can't be done in-process: Genesis
        owns global, once-per-process simulator state (see
        legged_gym/control/training.py's module docstring on why training
        itself runs in a subprocess for exactly this reason) — there is no
        safe way to tear down one Genesis scene and build another inside
        this same process. So a family switch is a PROCESS-LEVEL operation:
        this only records intent (same 'sim loop owns the actual work' split
        as restart() above); rugiar_driver.py's main loop, seeing
        family_switch_requested set, self-relaunches a fresh process for the
        new task and exits this one — see that loop's comment for the full
        mechanism. Validates up front (task registered AND has at least one
        local policy to load) so a switch that can't succeed never kills a
        working session for nothing."""
        if self.training is None:
            raise NotImplementedError("no TrainingManager configured for this ControlService")
        tasks, policies_per_task = self._switchable_families()
        if task not in tasks:
            raise ValueError(f"unknown task '{task}'")
        if not policies_per_task.get(task):
            raise ValueError(f"no trained local policies for task '{task}' yet")
        self.family_switch_requested = task

    def set_episode_timeout(self, seconds: Optional[float] = None) -> None:
        """Configures/disables legged_robot.py's own timer-based episode
        reset — see SimAdapter.set_episode_timeout's docstring. None (the
        default from adapter construction) means the robot only ever resets
        via an explicit restart() or a real fall/contact-force trip. Raises
        if the current adapter doesn't support it (e.g. RealAdapter has no
        episode-timeout concept)."""
        fn = getattr(self.adapter, "set_episode_timeout", None)
        if fn is None:
            raise NotImplementedError(f"{type(self.adapter).__name__} does not support set_episode_timeout")
        fn(seconds)

    def set_operator_speed_limit(self, fraction: float) -> None:
        """Caps every set_command() call this session to `fraction` of the
        trained command envelope — see SimAdapter.set_operator_speed_limit's
        docstring for the full "why" (server-side, one enforced limit
        shared by every client: the web UI, examples/joystick_controller.py,
        and eventually other robots/clients on their own token-gated
        connections, instead of each reimplementing its own cap). 1.0 (no
        extra cap beyond the trained envelope) is the default, matching
        this repo's own — and upstream unitree_rl_gym's — community-standard
        command range. Values above 1.0 are allowed (up to
        adapter.OPERATOR_SPEED_LIMIT_MAX) for deliberate out-of-distribution
        experimentation in sim — the web UI shows an explicit warning past
        1.0 rather than silently allowing it. Raises if the current adapter
        doesn't support it — RealAdapter does not, on purpose: asking a
        real, physical robot to exceed its trained envelope is a separate,
        more deliberate decision this method does not make for you."""
        fn = getattr(self.adapter, "set_operator_speed_limit", None)
        if fn is None:
            raise NotImplementedError(f"{type(self.adapter).__name__} does not support set_operator_speed_limit")
        fn(fraction)

    def set_command(self, vx: float, vy: float, yaw: float) -> None:
        """Directly commands a target walking velocity (m/s, m/s, rad/s),
        overriding whatever domain-randomization command resampling would
        otherwise pick — see SimAdapter.set_command for the clamping and
        heading_command interaction. Raises if the current adapter doesn't
        support manual commands (e.g. a not-yet-built RealAdapter path)."""
        set_command_fn = getattr(self.adapter, "set_command", None)
        if set_command_fn is None:
            raise NotImplementedError(f"{type(self.adapter).__name__} does not support set_command")
        set_command_fn(vx, vy, yaw)

    def set_random_events(self, push_robots: bool, auto_commands: bool,
                           push_dir: Optional[str] = None) -> None:
        """Independently toggles random pushes and command auto-resampling
        — see SimAdapter.set_random_events. Turning both off is what lets
        you drive the robot deliberately instead of watching it react to
        the same randomized stressors used during training. push_dir biases
        push direction the same way Create Policy's training-time push_dir
        does (None/'behind'/'front'/'left'/'right') — kept as the same
        vocabulary on purpose, see Simulator.sample_push_vel_xy()."""
        fn = getattr(self.adapter, "set_random_events", None)
        if fn is None:
            raise NotImplementedError(f"{type(self.adapter).__name__} does not support set_random_events")
        fn(push_robots, auto_commands, push_dir)

    def estop(self) -> None:
        """Emergency stop. Trips safety (which forces the damping fallback —
        see SafetyGovernor.tick — every tick from here on, not just once),
        AND calls the adapter's own estop() if it has one. On SimAdapter
        that's just a lifecycle flag; on RealAdapter it's a real, immediate
        zero-torque write over DDS — the one action that must not depend on
        anything in this file working correctly."""
        self.safety.tripped = True
        adapter_estop = getattr(self.adapter, "estop", None)
        if adapter_estop is not None:
            adapter_estop()

    # ---- the per-tick driving loop ----

    def tick(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        """One control step. Returns None if paused (caller should hold
        position / not call adapter.send_action this tick).

        Note: while safety.tripped, this still returns an action — but
        safety.tick() below has already forced the supervisor onto the
        damping (zero-action) skill and refuses to confirm any other
        pending switch, so "still returns an action" means "returns the
        harmless hold-position action," not "keeps running whatever policy
        was active when it tripped." estop()/a NaN/a fall are read this way
        on purpose: freezing entirely (returning None) would leave real
        motors holding their last commanded torque, which is not obviously
        safer than a controlled, zero-target hold."""
        if self.paused:
            return None

        state = self.adapter.get_state()
        state = self.safety.tick(state)

        if not self.safety.tripped and self.selector is not None:
            proposed = self.selector.propose(state)
            if proposed is not None and proposed != self.supervisor.active_name:
                self.supervisor.request_switch(proposed)
                # Autonomous proposals still go through the SAME safety gate
                # as a human's request — being self-driven doesn't grant a
                # shortcut. safety.tick() above already tried to confirm any
                # pending switch this tick if it judged the moment safe.

        action = self.supervisor.step(obs)
        self.adapter.record(obs, action, state)
        return action
