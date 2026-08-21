#!/usr/bin/env python3
"""
S5+S6 of item 4 of HANDOFF_mimic_motion_library_ux.md (mjlab training
backend), against the contract frozen in docs/mjlab_training_contract.md:

  - TrainingManager.start() routes an mjlab task to legged_gym/scripts/
    mjlab_train.py under `.venv-mjlab/bin/python`, and a Genesis task to
    web_train.py under the Genesis interpreter, unchanged (§5b);
  - the mjlab subprocess env does NOT carry PYTHONPATH=<repo root> — the
    repo's vendored top-level rsl_rl/ would shadow .venv-mjlab's PyPI
    rsl-rl-lib (docs/mjlab_migration.md R1). This is the exact OPPOSITE of
    what the Genesis branch must do, so it gets a test of its own;
  - the knobs a motion-tracking task has no analogue for (velocity command
    envelope, stability targets, pushes) are rejected in start(), not left
    to the subprocess's exit code (§5c.3);
  - backend="kaggle" + an mjlab task is refused outright (§5, §5c.2);
  - finalize_policy() preserves an .onnx export's suffix — copying it to
    "checkpoint.pt" would make load_policy_backend(), which dispatches on
    the suffix, unable to load it (§5c.7);
  - parse_training_log() reads rsl-rl 5.x's "Mean action std:" /
    "Episode_Reward/<term>:" lines as well as the vendored rsl_rl's
    "Mean action noise std:" / "Mean episode rew_<term>:" (§3d).

Deliberately NOT covered here: a real training run. That takes ~1-3 min of
CPU and is validated by hand (see the contract's acceptance command); these
stay fast/mocked so the suite runs in both venvs.

Run: python -m pytest tests/test_mjlab_training_dispatch.py -q
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
# NOTE: deliberately no sys.path.insert(0, ROOT) here. conftest.py already
# puts the repo root on sys.path LAST, and putting it first would shadow
# .venv-mjlab's PyPI rsl-rl-lib with this repo's vendored rsl_rl/ — which
# makes `import mjlab` fail and silently turns every mjlab assertion below
# into "mjlab isn't installed" (docs/mjlab_migration.md R1).

from legged_gym.control import training as training_mod  # noqa: E402
from legged_gym.control.training import (  # noqa: E402
    TrainingManager, parse_training_log, training_backend_for_task,
)

MJLAB_TASK = "Rugiar-G1-Mimic"
MOTION = "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"


class _FakeProc:
    """Stand-in for the Popen a real start() would return — poll() returning
    None keeps the job 'running' so nothing tries to read a result file."""

    def poll(self):
        return None


def _pin_registries(mjlab_tasks, genesis_tasks):
    """Pins what each registry probe reports, so the dispatch assertions are
    about the DISPATCH LOGIC rather than about which simulator happens to be
    importable in the venv running the suite (and, in a full-suite run,
    whether some earlier test left a stubbed `legged_gym` in sys.modules —
    several do, which makes ambient registry state genuinely unreadable).
    Real-registry resolution is covered separately by TestBackendForTask."""
    return mock.patch.multiple(
        training_mod,
        _mjlab_registered_tasks=mock.Mock(return_value=mjlab_tasks),
        _legged_gym_registered_tasks=mock.Mock(return_value=genesis_tasks),
    )


def _start_capturing(tm: TrainingManager, **kwargs):
    """Runs start() with Popen mocked out; returns (job, argv, env)."""
    with mock.patch.object(training_mod.subprocess, "Popen",
                           return_value=_FakeProc()) as popen:
        job_id = tm.start(**kwargs)
    args, popen_kwargs = popen.call_args
    return tm.jobs[job_id], args[0], popen_kwargs["env"]


class TestBackendForTask(unittest.TestCase):
    """The decision table itself (contract §5a), driven off pinned registry
    probes so every row is exercised in BOTH venvs — including the two rows
    the venv running this suite can't reach on its own."""

    def test_mjlab_registry_is_authoritative_when_importable(self):
        with _pin_registries({MJLAB_TASK}, None):
            self.assertEqual(training_backend_for_task(MJLAB_TASK), "mjlab")
            self.assertEqual(training_backend_for_task("g1"), "genesis")

    def test_genesis_registry_decides_by_exclusion(self):
        with _pin_registries(None, {"g1", "g1_deepmimic"}):
            self.assertEqual(training_backend_for_task("g1"), "genesis")
            self.assertEqual(training_backend_for_task(MJLAB_TASK), "mjlab")

    def test_refuses_to_guess_when_neither_registry_is_readable(self):
        """Answering by exclusion off an unreadable registry would misroute
        every job — say so instead."""
        with _pin_registries(None, None):
            with self.assertRaises(ValueError) as ctx:
                training_backend_for_task("g1")
            self.assertIn("neither", str(ctx.exception))

    def test_empty_registry_counts_as_unreadable_not_as_no_tasks(self):
        """An empty task list means 'this process couldn't read the registry',
        not 'no task exists' — answering by exclusion off an empty set would
        misroute every job. Asserted on the probe itself, since that's where
        the normalization lives."""
        registry = pytest.importorskip("mjlab.tasks.registry")
        with mock.patch.object(registry, "list_tasks", return_value=[]):
            self.assertIsNone(training_mod._mjlab_registered_tasks())
        with mock.patch.object(registry, "list_tasks", return_value=[MJLAB_TASK]):
            self.assertEqual(training_mod._mjlab_registered_tasks(), {MJLAB_TASK})

    def test_resolves_against_the_real_registry_of_this_venv(self):
        try:
            resolved = training_backend_for_task(MJLAB_TASK)
        except ValueError:
            self.skipTest("neither registry readable in this session (stubbed by an earlier test)")
        self.assertEqual(resolved, "mjlab")
        self.assertEqual(training_backend_for_task("g1"), "genesis")


class TestTaskDefaultsShape(unittest.TestCase):
    """S10/S11 (HANDOFF_mimic_motion_library_ux.md) — task_defaults()'s
    'variables' dict is EMPTY (not full-keyed-with-None) for a task with no
    VARIABLE_REGISTRY concept at all (a motion-tracking task has no base
    height/tilt/etc target), and 'needs_motion_file' is read generically off
    the task's own cfg.commands rather than a task-name check — see
    TrainingManager.task_defaults()'s docstring and
    docs/mjlab_training_contract.md §2. This is what lets the Create Policy
    panel (web/app.js's refreshTaskFieldVisibility()) hide the Genesis-only
    field-groups and show the motion-clip picker without hardcoding
    'Rugiar-G1-Mimic' by name — asserted against the REAL registry of
    whichever venv runs this suite, not a mock, so a shape regression in
    either backend's branch is actually caught."""

    def test_mjlab_tracking_task_has_empty_variables_and_needs_motion(self):
        pytest.importorskip("mjlab")
        import mjlab_tasks  # noqa: F401 - registers Rugiar-G1-Mimic
        tm = TrainingManager()
        d = tm.task_defaults(MJLAB_TASK)
        self.assertEqual(d["variables"], {})
        self.assertTrue(d["needs_motion_file"])
        self.assertEqual(
            set(d["reward_scales"]),
            {"motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
             "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel",
             "action_rate_l2", "joint_limit", "self_collisions"},
        )

    def test_mjlab_non_tracking_task_does_not_need_motion(self):
        """Generic per-task detection (cfg.commands contains 'motion'), not a
        name check — a non-tracking mjlab task (no motion command term) must
        NOT be flagged as needing a clip, and keeps the same empty-variables
        shape (cartpole has no VARIABLE_REGISTRY concept either)."""
        pytest.importorskip("mjlab")
        import mjlab_tasks  # noqa: F401
        from mjlab.tasks import registry
        cartpole_tasks = [t for t in registry.list_tasks() if "Cartpole" in t]
        if not cartpole_tasks:
            self.skipTest("no cartpole task registered in this mjlab build")
        tm = TrainingManager()
        d = tm.task_defaults(cartpole_tasks[0])
        self.assertFalse(d["needs_motion_file"])

    def test_genesis_task_keeps_full_variables_and_no_motion_file(self):
        """The Genesis Create Policy flow (velocity envelope/push/stability
        targets) must be unaffected by the mjlab-shaped branch above."""
        try:
            import legged_gym.envs  # noqa: F401 - registers Genesis tasks
            from legged_gym.utils import task_registry  # noqa: F401
        except ImportError:
            self.skipTest("no Genesis in this venv")
        tm = TrainingManager()
        d = tm.task_defaults("g1")
        # The shape guarantee itself (S11's actual gating signal) — always
        # holds regardless of ambient sys.modules state.
        self.assertEqual(set(d["variables"]), set(TrainingManager.VARIABLE_REGISTRY))
        self.assertFalse(d["needs_motion_file"])
        if not d["reward_scales"] or any(v["reference"] is None for v in d["variables"].values()):
            # Same class of flake TestBackendForTask.
            # test_resolves_against_the_real_registry_of_this_venv already
            # documents: in a full-suite run, an earlier test may have left
            # a stubbed `legged_gym` in sys.modules, making env_cfg
            # resolution genuinely unreadable here — not a real regression.
            self.skipTest("ambient registry state unreadable in this run (stubbed legged_gym from an earlier test)")
        self.assertIn("tracking_lin_vel", d["reward_scales"])


@unittest.skipUnless(training_mod.MJLAB_PYTHON.exists(),
                     "no .venv-mjlab on this machine")
class TestMjlabDispatch(unittest.TestCase):
    def setUp(self):
        self.tm = TrainingManager()
        self.base = dict(policy_name="t_mjlab_dispatch", task=MJLAB_TASK,
                         num_envs=8, max_iterations=20, motion_file=MOTION)
        patcher = _pin_registries({MJLAB_TASK}, None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_routes_to_mjlab_train_under_the_mjlab_interpreter(self):
        job, argv, _ = _start_capturing(self.tm, **self.base)
        self.assertEqual(argv[0], str(training_mod.MJLAB_PYTHON))
        self.assertEqual(argv[2], str(training_mod.MJLAB_TRAIN_SCRIPT))
        self.assertIn("--motion_file", argv)
        self.assertEqual(job.simulator, "mjlab")

    def test_does_not_pass_genesis_only_flags(self):
        """web_train.py's --headless/--cpu prefix has no mjlab analogue —
        mjlab_train.py takes --device instead and never opens a viewer."""
        _, argv, _ = _start_capturing(self.tm, **self.base)
        self.assertNotIn("--cpu", argv)
        self.assertNotIn("--headless", argv)

    def test_repo_root_is_kept_off_pythonpath(self):
        """The R1 collision: with REPO_ROOT prepended, the repo's vendored
        rsl_rl/ shadows .venv-mjlab's PyPI rsl-rl-lib and the run dies."""
        _, _, env = _start_capturing(self.tm, **self.base)
        self.assertNotIn(str(training_mod.REPO_ROOT),
                         env.get("PYTHONPATH", "").split(os.pathsep))
        self.assertEqual(env["SIMULATOR"], "mjlab")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

    def test_inherited_pythonpath_entries_survive(self):
        """Only REPO_ROOT is stripped — an unrelated inherited entry is not
        this branch's business to drop."""
        with mock.patch.dict(os.environ, {
                "PYTHONPATH": os.pathsep.join([str(training_mod.REPO_ROOT), "/somewhere/else"])}):
            _, _, env = _start_capturing(self.tm, **self.base)
        self.assertEqual(env["PYTHONPATH"], "/somewhere/else")

    def test_kaggle_backend_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.tm.start(backend="kaggle", **self.base)
        self.assertIn("mjlab", str(ctx.exception).lower())

    def test_motion_file_is_required(self):
        args = dict(self.base)
        args.pop("motion_file")
        with self.assertRaises(ValueError) as ctx:
            self.tm.start(**args)
        self.assertIn("motion_file", str(ctx.exception))

    def test_inapplicable_knobs_are_rejected_up_front(self):
        for knob, value in (("cmd_vx", [-0.5, 0.5]), ("base_height_target", 0.5),
                            ("push_robots", True), ("push_dir", "behind")):
            with self.subTest(knob=knob):
                with self.assertRaises(ValueError) as ctx:
                    self.tm.start(**{**self.base, knob: value})
                self.assertIn(knob, str(ctx.exception))

    def test_reward_scale_overrides_validate_against_mjlab_terms(self):
        """mjlab's reward terms come from a dict of RewardTermCfg, not
        Genesis's <Cfg>.rewards.scales — a valid mimic term must pass and a
        Genesis-vocabulary one must fail."""
        pytest.importorskip("mjlab")
        job, argv, _ = _start_capturing(
            self.tm, **{**self.base, "reward_scale_overrides": {"motion_body_pos": 2.0}})
        self.assertIn("--reward_scale", argv)
        with self.assertRaises(ValueError):
            self.tm.start(**{**self.base, "policy_name": "t2",
                             "reward_scale_overrides": {"tracking_lin_vel": 1.0}})


class TestGenesisDispatchUnchanged(unittest.TestCase):
    def setUp(self):
        patcher = _pin_registries(None, {"g1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    @unittest.skipUnless(training_mod.GENESIS_PYTHON.exists(), "no .venv on this machine")
    def test_genesis_task_still_routes_to_web_train(self):
        tm = TrainingManager(python_exe=str(training_mod.GENESIS_PYTHON))
        job, argv, env = _start_capturing(
            tm, policy_name="t_genesis_dispatch", task="g1", num_envs=8, max_iterations=5)
        self.assertEqual(argv[2], str(training_mod.TRAIN_SCRIPT))
        self.assertIn("--headless", argv)
        self.assertIn("--cpu", argv)
        self.assertEqual(job.simulator, "genesis")
        # The Genesis branch's PYTHONPATH pin is load-bearing (a sibling
        # editable install would otherwise win) — it must NOT be dropped.
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(training_mod.REPO_ROOT))
        self.assertEqual(env["SIMULATOR"], "genesis")

    @unittest.skipUnless(training_mod.GENESIS_PYTHON.exists(), "no .venv on this machine")
    def test_genesis_job_from_an_mjlab_session_switches_interpreter(self):
        """A TrainingManager living in an mjlab session can't run web_train.py
        with its own interpreter — that venv has no Genesis."""
        tm = TrainingManager(python_exe=str(training_mod.MJLAB_PYTHON))
        _, argv, _ = _start_capturing(
            tm, policy_name="t_genesis_from_mjlab", task="g1", num_envs=8, max_iterations=5)
        self.assertEqual(argv[0], str(training_mod.GENESIS_PYTHON))


class TestFinalizePolicySuffix(unittest.TestCase):
    """§5c.7 — load_policy_backend() dispatches on the file suffix, so an
    .onnx export copied to checkpoint.pt is simply unloadable."""

    def _finalize(self, src_name: str) -> Path:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / src_name
            src.write_bytes(b"not-a-real-checkpoint")
            with mock.patch.object(training_mod, "POLICIES_DIR", Path(tmp) / "policies"):
                tm = TrainingManager()
                dest = tm.finalize_policy("t_suffix", "some_task", str(src), None)
            return Path(dest)

    def test_onnx_export_keeps_its_suffix(self):
        self.assertEqual(self._finalize("policy.onnx").name, "checkpoint.onnx")

    def test_pt_export_is_unchanged(self):
        self.assertEqual(self._finalize("policy.pt").name, "checkpoint.pt")


class TestParseTrainingLogBothStacks(unittest.TestCase):
    """§3d — one parser, two rsl_rl print formats."""

    MJLAB_BLOCK = """
                    Learning iteration 3/20

                       Mean action std: 1.00
                           Mean reward: -1.65
                   Mean episode length: 12.10
         Episode_Reward/motion_body_pos: 0.0193
          Episode_Reward/action_rate_l2: -0.1272
        Episode_Termination/time_out: 0.0000
         Metrics/motion/error_body_pos: 0.7207
"""
    GENESIS_BLOCK = """
                    Learning iteration 3/20

                 Mean action noise std: 0.85
                           Mean reward: 4.20
                   Mean episode length: 300.10
          Mean episode rew_tracking_lin_vel: 1.2500
"""

    def _parse(self, text: str) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            return parse_training_log(path)
        finally:
            os.unlink(path)

    def test_reads_rsl_rl_5x_mjlab_format(self):
        out = self._parse(self.MJLAB_BLOCK)
        self.assertEqual(out["final"]["noise_std"], 1.00)
        self.assertEqual(out["final"]["reward"], -1.65)
        self.assertEqual(out["final_reward_terms"]["motion_body_pos"], 0.0193)
        self.assertEqual(out["final_reward_terms"]["action_rate_l2"], -0.1272)
        # Termination/Metrics lines are not reward terms and must not appear.
        self.assertNotIn("time_out", out["final_reward_terms"])
        self.assertNotIn("error_body_pos", out["final_reward_terms"])

    def test_still_reads_the_vendored_genesis_format(self):
        out = self._parse(self.GENESIS_BLOCK)
        self.assertEqual(out["final"]["noise_std"], 0.85)
        self.assertEqual(out["final_reward_terms"]["tracking_lin_vel"], 1.25)


@unittest.skipUnless(training_mod.MJLAB_PYTHON.exists(), "no .venv-mjlab on this machine")
class TestMjlabTrainCli(unittest.TestCase):
    """The entrypoint's CLI contract (§2). Exercised through a real
    subprocess rather than by importing the module, because importing it
    pulls in mjlab — which only exists in the other venv, so an in-process
    test could never run under `.venv`."""

    SCRIPT = ROOT / "legged_gym" / "scripts" / "mjlab_train.py"

    def _run(self, *args) -> subprocess.CompletedProcess:
        env = {**os.environ, "SIMULATOR": "mjlab", "CUDA_VISIBLE_DEVICES": ""}
        env.pop("PYTHONPATH", None)
        return subprocess.run(
            [str(training_mod.MJLAB_PYTHON), str(self.SCRIPT), *args],
            capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=300)

    def test_rejects_inapplicable_genesis_flags(self):
        out = self._run("--task", MJLAB_TASK, "--name", "x", "--max_iterations", "1",
                        "--result_path", "/dev/null", "--cmd_vx_range", "-0.5", "0.5")
        self.assertEqual(out.returncode, 2)
        self.assertIn("--cmd_vx_range", out.stderr)
        self.assertIn("do not apply", out.stderr)

    def test_rejects_gpu_flag_pointing_at_device(self):
        out = self._run("--task", MJLAB_TASK, "--name", "x", "--max_iterations", "1",
                        "--result_path", "/dev/null", "--gpu")
        self.assertEqual(out.returncode, 2)
        self.assertIn("--device cuda:0", out.stderr)

    def test_requires_a_budget(self):
        out = self._run("--task", MJLAB_TASK, "--name", "x", "--result_path", "/dev/null")
        self.assertEqual(out.returncode, 2)
        self.assertIn("--max_iterations", out.stderr)

    def test_requires_a_motion_file_for_a_tracking_task(self):
        out = self._run("--task", MJLAB_TASK, "--name", "x", "--max_iterations", "1",
                        "--result_path", "/dev/null")
        self.assertEqual(out.returncode, 2)
        self.assertIn("--motion_file is required", out.stderr)

    def test_rejects_an_unknown_reward_term(self):
        out = self._run("--task", MJLAB_TASK, "--name", "x", "--max_iterations", "1",
                        "--motion_file", MOTION, "--result_path", "/dev/null",
                        "--reward_scale", "tracking_lin_vel", "1.0")
        self.assertEqual(out.returncode, 2)
        self.assertIn("unknown reward term", out.stderr)
        self.assertIn("motion_body_pos", out.stderr)  # lists the valid ones


if __name__ == "__main__":
    unittest.main()
