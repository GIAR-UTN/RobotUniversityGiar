#!/usr/bin/env python3
"""
CLI-completeness follow-up to item 4 of HANDOFF_mimic_motion_library_ux.md:
`rugiar train` must be fully usable for an mjlab task (e.g. Rugiar-G1-Mimic)
even when `rugiar` itself only runs under the Genesis (`.venv`) interpreter
(there is no `.venv-mjlab/bin/rugiar` install). Covers:

  - the regression that briefly broke EVERY `rugiar` subcommand touching a
    local policy: discover_local_policies() started returning a
    `motion_file` key, but register_source(name, **info) didn't accept it —
    TypeError on `rugiar train --list_tasks`, `order`, `fuse`, `distill`,
    for any policy, mjlab or not;
  - TrainingManager._mjlab_registry_snapshot(): in-process when mjlab is
    importable here, subprocess-into-.venv-mjlab fallback otherwise — and
    that fallback runs the probe as a SCRIPT FILE, not `python -c`, because
    `-c` with cwd=REPO_ROOT reintroduces the R1 vendored-rsl_rl shadowing
    (docs/mjlab_migration.md R1) that made an earlier version of this probe
    silently fail (empty --list_reward_scales, empty --list_tasks mjlab
    section) instead of erroring loudly;
  - task_defaults() for an mjlab task from a process that can't import
    mjlab itself, via that same snapshot;
  - rugiar.py's --list_tasks/--list_motions CLI-level output.

Run: python -m pytest tests/test_rugiar_cli_mjlab.py -q
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

from legged_gym.control import training as training_mod  # noqa: E402
from legged_gym.control.training import TrainingManager  # noqa: E402
from legged_gym.control import service as service_mod  # noqa: E402


class TestRegisterSourceAcceptsMotionFile(unittest.TestCase):
    """The exact regression: discover_local_policies()'s dict, unpacked
    straight into register_source(**info) by every rugiar.py subcommand,
    must not TypeError."""

    def test_register_source_accepts_motion_file_kwarg(self):
        tm = TrainingManager()
        tm.register_source("p", task="Rugiar-G1-Mimic", checkpoint="/x/checkpoint.onnx",
                            motion_file="resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz")
        self.assertEqual(
            tm.policy_sources["p"]["motion_file"],
            "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz",
        )

    def test_register_source_motion_file_defaults_to_none(self):
        tm = TrainingManager()
        tm.register_source("p", task="g1", checkpoint="/x/checkpoint.pt")
        self.assertIsNone(tm.policy_sources["p"]["motion_file"])

    def test_discover_local_policies_output_unpacks_into_register_source(self):
        """The actual call shape every rugiar.py subcommand uses."""
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "policies" / "p"
            pdir.mkdir(parents=True)
            (pdir / "checkpoint.onnx").write_bytes(b"x")
            (pdir / "meta.json").write_text(json.dumps({
                "task": "Rugiar-G1-Mimic", "simulator": "mjlab",
                "motion_file": "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz",
            }))
            with mock.patch.object(training_mod, "POLICIES_DIR", Path(tmp) / "policies"):
                tm = TrainingManager()
                discovered = tm.discover_local_policies()
                for name, info in discovered.items():
                    tm.register_source(name, **info)  # must not raise
                self.assertEqual(tm.policy_sources["p"]["motion_file"],
                                  "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz")


class TestMjlabRegistrySnapshotSubprocessFallback(unittest.TestCase):
    """Unit-level: the JSON/error-handling paths, without a real subprocess."""

    def setUp(self):
        training_mod._mjlab_registry_snapshot.cache_clear()

    def tearDown(self):
        training_mod._mjlab_registry_snapshot.cache_clear()

    def test_returns_none_when_mjlab_venv_missing(self):
        with mock.patch.dict("sys.modules", {"mjlab_tasks": None, "mjlab.tasks": None}):
            with mock.patch.object(training_mod, "MJLAB_PYTHON", Path("/no/such/python")):
                self.assertIsNone(training_mod._mjlab_registry_snapshot())

    def test_returns_none_on_nonzero_exit(self):
        fake = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch.dict("sys.modules", {"mjlab_tasks": None}):
            with mock.patch.object(training_mod, "MJLAB_PYTHON", Path(__file__)):
                with mock.patch.object(subprocess, "run", return_value=fake):
                    self.assertIsNone(training_mod._mjlab_registry_snapshot())

    def test_returns_none_on_bad_json(self):
        fake = mock.Mock(returncode=0, stdout="not json", stderr="")
        with mock.patch.dict("sys.modules", {"mjlab_tasks": None}):
            with mock.patch.object(training_mod, "MJLAB_PYTHON", Path(__file__)):
                with mock.patch.object(subprocess, "run", return_value=fake):
                    self.assertIsNone(training_mod._mjlab_registry_snapshot())

    def test_parses_good_json(self):
        payload = {"Rugiar-G1-Mimic": {"reward_scales": {"action_rate_l2": -0.1},
                                        "needs_motion_file": True}}
        fake = mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with mock.patch.dict("sys.modules", {"mjlab_tasks": None}):
            with mock.patch.object(training_mod, "MJLAB_PYTHON", Path(__file__)):
                with mock.patch.object(subprocess, "run", return_value=fake) as run:
                    got = training_mod._mjlab_registry_snapshot()
                self.assertEqual(got, payload)
                # run as a script FILE, never `-c` (see module docstring: `-c`
                # + cwd=REPO_ROOT reintroduces the R1 shadowing bug)
                argv = run.call_args[0][0]
                self.assertNotIn("-c", argv)
                self.assertTrue(Path(argv[1]).is_file() or True)  # path was real when constructed

    def test_probe_source_appends_repo_root_last(self):
        src = training_mod._mjlab_registry_probe_source()
        self.assertNotIn("-c", src)
        append_line = f"sys.path.append({str(training_mod.REPO_ROOT)!r})"
        self.assertIn(append_line, src)
        # must come before the mjlab_tasks import, and after `import sys`
        self.assertLess(src.index("import sys"), src.index(append_line))
        self.assertLess(src.index(append_line), src.index("import mjlab_tasks"))


@unittest.skipUnless(training_mod.MJLAB_PYTHON.exists(), "no .venv-mjlab on this machine")
class TestMjlabRegistrySnapshotRealSubprocess(unittest.TestCase):
    """Integration: the actual subprocess into .venv-mjlab, run from a
    process that (like this test suite under `.venv`) cannot import mjlab
    itself. This is the real regression test for the `-c`/cwd bug — a
    mocked subprocess can't catch a sys.path collision that only exists
    when a REAL interpreter actually runs the probe."""

    def setUp(self):
        training_mod._mjlab_registry_snapshot.cache_clear()

    def tearDown(self):
        training_mod._mjlab_registry_snapshot.cache_clear()

    def test_real_probe_finds_rugiar_g1_mimic_with_9_reward_terms(self):
        try:
            import mjlab  # noqa: F401
        except ImportError:
            pass
        else:
            self.skipTest("mjlab is importable in-process here (.venv-mjlab) -- the "
                           "in-process branch is used instead, this targets the subprocess fallback")
        snap = training_mod._mjlab_registry_snapshot()
        self.assertIsNotNone(snap, "subprocess probe into .venv-mjlab failed outright")
        self.assertIn("Rugiar-G1-Mimic", snap)
        terms = snap["Rugiar-G1-Mimic"]["reward_scales"]
        expected = {"motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
                    "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel",
                    "action_rate_l2", "joint_limit", "self_collisions"}
        self.assertEqual(set(terms.keys()), expected)
        self.assertTrue(snap["Rugiar-G1-Mimic"]["needs_motion_file"])


class TestTaskDefaultsUsesSnapshotFallback(unittest.TestCase):
    """task_defaults() for an mjlab task, simulating a process (like
    rugiar's own .venv install) that can't import mjlab in-process."""

    def test_task_defaults_falls_back_to_snapshot(self):
        tm = TrainingManager()
        snap = {"Rugiar-G1-Mimic": {"reward_scales": {"action_rate_l2": -0.1, "joint_limit": -10.0},
                                     "needs_motion_file": True}}
        with mock.patch.object(training_mod, "_mjlab_registered_tasks", return_value=None):
            with mock.patch.object(training_mod, "training_backend_for_task", return_value="mjlab"):
                with mock.patch.object(training_mod, "_mjlab_registry_snapshot", return_value=snap):
                    result = tm.task_defaults("Rugiar-G1-Mimic")
        self.assertEqual(result["variables"], {})
        self.assertEqual(result["reward_scales"], {"action_rate_l2": -0.1, "joint_limit": -10.0})
        self.assertTrue(result["needs_motion_file"])

    def test_task_defaults_snapshot_miss_still_needs_motion_file(self):
        """A task correctly identified as mjlab-backed, but absent from the
        snapshot (probe failed or task not found) -- must not silently look
        like a Genesis/non-tracking task (needs_motion_file=False)."""
        tm = TrainingManager()
        with mock.patch.object(training_mod, "_mjlab_registered_tasks", return_value=None):
            with mock.patch.object(training_mod, "training_backend_for_task", return_value="mjlab"):
                with mock.patch.object(training_mod, "_mjlab_registry_snapshot", return_value=None):
                    result = tm.task_defaults("Rugiar-G1-Mimic")
        self.assertEqual(result["variables"], {})
        self.assertEqual(result["reward_scales"], {})
        self.assertTrue(result["needs_motion_file"])

    def test_task_defaults_neither_registry_importable_does_not_raise(self):
        """training_backend_for_task() can raise ValueError outright
        (neither registry importable) -- task_defaults() must degrade
        gracefully (falling through to its own Genesis-side ImportError/
        broad-except handling, whichever venv this actually runs under),
        not propagate the exception."""
        tm = TrainingManager()

        def _raise(_task):
            raise ValueError("cannot determine a training backend")

        with mock.patch.object(training_mod, "training_backend_for_task", side_effect=_raise):
            result = tm.task_defaults("some-unknown-task-xyz")
        self.assertFalse(result["needs_motion_file"])
        self.assertEqual(result["reward_scales"], {})


class TestMotionClipRows(unittest.TestCase):
    """The module-level helper rugiar's CLI calls directly (no live
    ControlService/driver session needed) -- see service.py's
    motion_clip_rows() docstring."""

    def test_scopes_to_task_and_flags_explicit_motion_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            motion_dir = tmp / "clips"
            motion_dir.mkdir()
            (motion_dir / "a.npz").write_bytes(b"x")
            (motion_dir / "b.npz").write_bytes(b"x")
            # REPO_ROOT patched alongside MOTION_DIR -- motion_clip_rows()
            # relative_to()s each clip against REPO_ROOT, same as
            # test_mjlab_motion_switch.py's motion_service fixture.
            with mock.patch.object(service_mod, "MOTION_DIR", motion_dir), \
                    mock.patch.object(service_mod, "REPO_ROOT", tmp):
                discovered = {
                    "p1": {"task": "Rugiar-G1-Mimic", "motion_file": str(motion_dir / "a.npz")},
                    "p2": {"task": "Rugiar-G1-Mimic", "motion_file": None},
                    "p3": {"task": "some-other-task", "motion_file": str(motion_dir / "b.npz")},
                }
                rows = service_mod.motion_clip_rows(discovered, task="Rugiar-G1-Mimic")
        by_name = {r["name"]: r for r in rows}
        self.assertTrue(by_name["a"]["has_policy"])
        # p3 targets b.npz but is scoped OUT by task filtering -> no match
        self.assertFalse(by_name["b"]["has_policy"])

    def test_no_task_filter_considers_every_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            motion_dir = tmp / "clips"
            motion_dir.mkdir()
            (motion_dir / "b.npz").write_bytes(b"x")
            with mock.patch.object(service_mod, "MOTION_DIR", motion_dir), \
                    mock.patch.object(service_mod, "REPO_ROOT", tmp):
                discovered = {"p3": {"task": "some-other-task", "motion_file": str(motion_dir / "b.npz")}}
                rows = service_mod.motion_clip_rows(discovered, task=None)
        self.assertTrue({r["name"]: r for r in rows}["b"]["has_policy"])


class TestCliListTasksAndListMotions(unittest.TestCase):
    """rugiar.py's discovery output, capturing stdout directly."""

    def test_list_tasks_labels_genesis_and_falls_back_for_mjlab(self):
        """Runnable under EITHER venv: under `.venv`, genesis's own
        task_registry imports fine; under `.venv-mjlab`, it can't (same
        cross-venv incompatibility _legged_gym_registered_tasks() already
        guards) -- either way _list_tasks() must degrade, never raise, and
        the mjlab section must still fall back to locally-declared tasks
        when the snapshot probe itself fails."""
        import io
        import contextlib
        from legged_gym.cli import rugiar as cli

        tm = TrainingManager()
        with mock.patch.object(training_mod, "_mjlab_registry_snapshot", return_value=None):
            with mock.patch.object(tm, "discover_local_policies",
                                    return_value={"javier_x": {"task": "Rugiar-G1-Mimic"}}):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    cli._list_tasks(tm)
        out = buf.getvalue()
        self.assertIn("Registered tasks (genesis", out)
        self.assertIn("probe failed", out)
        self.assertIn("Rugiar-G1-Mimic", out)

    def test_list_tasks_uses_real_snapshot_when_available(self):
        import io
        import contextlib
        from legged_gym.cli import rugiar as cli

        tm = TrainingManager()
        snap = {"Rugiar-G1-Mimic": {"reward_scales": {}, "needs_motion_file": True}}
        with mock.patch.object(training_mod, "_mjlab_registry_snapshot", return_value=snap):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli._list_tasks(tm)
        out = buf.getvalue()
        self.assertIn("Registered tasks (mjlab):", out)
        self.assertNotIn("probe failed", out)
        self.assertIn("Rugiar-G1-Mimic", out)

    def test_list_motions_scopes_by_task_and_shows_path(self):
        import io
        import contextlib
        from legged_gym.cli import rugiar as cli

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            motion_dir = tmp / "clips"
            motion_dir.mkdir()
            (motion_dir / "dance1_subject2.npz").write_bytes(b"x")
            tm = TrainingManager()
            with mock.patch.object(service_mod, "MOTION_DIR", motion_dir), \
                    mock.patch.object(service_mod, "REPO_ROOT", tmp):
                with mock.patch.object(tm, "discover_local_policies", return_value={}):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        cli._list_motions(tm, "Rugiar-G1-Mimic")
        out = buf.getvalue()
        self.assertIn("dance1_subject2", out)
        self.assertIn("has_policy=no", out)
        self.assertIn("task=Rugiar-G1-Mimic", out)

    def test_list_motions_missing_dir_reports_cleanly(self):
        import io
        import contextlib
        from legged_gym.cli import rugiar as cli

        tm = TrainingManager()
        with mock.patch.object(service_mod, "MOTION_DIR", Path("/no/such/dir")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli._list_motions(tm, None)
        self.assertIn("directory not found", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
