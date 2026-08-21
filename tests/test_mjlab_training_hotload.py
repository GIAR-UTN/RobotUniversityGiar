#!/usr/bin/env python3
"""
Hot-loading a finished training job into a LIVE mjlab session — the follow-up
HANDOFF_mimic_motion_library_ux.md left open ("a training job started from a
live mjlab session needs a session restart before the new policy hot-loads").

The fix is rugiar_driver_mjlab.py's module-level drain_finished_training(),
called once per control tick from control_tick(). This file covers:

  1. the drain itself — a job TrainingManager reports done is finalized into
     policies/<name>/ AND added to the RUNNING supervisor, with no relaunch;
  2. the failure path — a job whose export won't load marks the JOB failed
     and never raises, because this runs inside the viewer's policy callback
     and an exception there takes down the whole session;
  3. that control_tick() actually calls it (asserted with `ast`, the same
     parse-don't-import approach tests/test_driver_family_parity.py uses —
     importing is fine here, but the CALL SITE is the thing that regressed
     before and it lives inside a closure no test can reach).

Deliberately NOT covered here: a real training run against a real driver
session. That takes minutes of CPU and is validated by hand (see this
change's report); these stay fast and run in the mjlab venv only.

Run with .venv-mjlab (conftest.py handles sys.path, R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_mjlab_training_hotload.py -q
"""
import ast
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO / "legged_gym/scripts/rugiar_driver_mjlab.py"

import legged_gym.control.training as training_mod  # noqa: E402
from legged_gym.control.training import TrainingJob, TrainingManager  # noqa: E402


def _load_driver_module():
    pytest.importorskip("mjlab")
    spec = importlib.util.spec_from_file_location("rugiar_driver_mjlab", DRIVER_PATH)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


@pytest.fixture(scope="module")
def driver():
    return _load_driver_module()


class FakeSupervisor:
    """Only add_policy() matters to the drain — the real PolicySupervisor
    needs loaded torch modules this test has no reason to build."""

    def __init__(self):
        self.added = []

    def add_policy(self, policy):
        self.added.append(policy)


class FakePolicy:
    def __init__(self, name):
        self.name = name


def _finished_job(tmp: Path, name="hotloaded") -> TrainingJob:
    """A job in exactly the state poll() hands back on success: status done,
    policy_path pointing at a real exported file on disk."""
    export = tmp / "logs" / "run" / "exported" / "policy.onnx"
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_bytes(b"not-a-real-onnx-but-finalize-only-copies-it")
    return TrainingJob(
        id="job1234", policy_name=name, task="Rugiar-G1-Mimic",
        command="rugiar train --task Rugiar-G1-Mimic --name " + name,
        log_path=str(tmp / "job.log"), result_path=str(tmp / "job.result.json"),
        progress_path=str(tmp / "job.progress.json"), started_at=1.0, finished_at=2.0,
        max_iterations=3, max_minutes=None, num_envs=8, iterations_done=3,
        status="done", policy_path=str(export), simulator="mjlab")


class _OneShotTraining(TrainingManager):
    """A real TrainingManager (so finalize_policy()/register_source() run for
    real) whose poll() reports one finished job exactly once, the way the
    live manager does on the tick after a subprocess exits."""

    def __init__(self, job):
        super().__init__()
        self._pending = [job]

    def poll(self):
        pending, self._pending = self._pending, []
        return pending


def test_drain_finalizes_and_hot_loads_without_a_relaunch(driver, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        monkeypatch.setattr(training_mod, "POLICIES_DIR", tmp / "policies")
        job = _finished_job(tmp)
        training = _OneShotTraining(job)
        supervisor = FakeSupervisor()

        loaded = driver.drain_finished_training(
            training, supervisor, lambda j, path: FakePolicy(j.policy_name))

        assert loaded == ["hotloaded"]
        # In the running process: selectable now, no restart involved.
        assert [p.name for p in supervisor.added] == ["hotloaded"]
        # On disk: the self-contained policies/<name>/ folder finalize_policy()
        # writes, with the .onnx suffix preserved (load_policy_backend()
        # dispatches on it).
        policy_dir = tmp / "policies" / "hotloaded"
        assert (policy_dir / "checkpoint.onnx").is_file()
        meta = json.loads((policy_dir / "meta.json").read_text())
        assert meta["task"] == "Rugiar-G1-Mimic"
        assert meta["simulator"] == "mjlab"
        # And in the clone-from catalog, same as a Genesis-session completion.
        assert "hotloaded" in training.policy_sources
        assert job.status == "done"


def test_drain_is_a_no_op_when_nothing_finished(driver):
    class _Idle(TrainingManager):
        def poll(self):
            return []

    supervisor = FakeSupervisor()
    assert driver.drain_finished_training(_Idle(), supervisor, lambda j, p: FakePolicy("x")) == []
    assert supervisor.added == []


def test_a_policy_that_wont_load_fails_the_job_instead_of_the_session(driver, monkeypatch):
    """This runs inside the viewer's policy callback — an exception here kills
    the whole control session, so a bad export must degrade to a failed JOB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        monkeypatch.setattr(training_mod, "POLICIES_DIR", tmp / "policies")
        job = _finished_job(tmp, name="broken")
        supervisor = FakeSupervisor()

        def _explode(j, path):
            raise RuntimeError("obs size mismatch")

        loaded = driver.drain_finished_training(_OneShotTraining(job), supervisor, _explode)

        assert loaded == []
        assert supervisor.added == []
        assert job.status == "failed"
        assert "obs size mismatch" in job.error


def test_control_tick_drains_training_every_tick():
    """The regression this whole change fixes was a MISSING CALL, not a broken
    helper — pin the call site itself. Parsed, not imported: control_tick is a
    closure inside main() that no test can reach at runtime."""
    tree = ast.parse(DRIVER_PATH.read_text(), filename=str(DRIVER_PATH))
    tick = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "control_tick"), None)
    assert tick is not None, "control_tick() is gone from rugiar_driver_mjlab.py"
    called = {n.func.id for n in ast.walk(tick)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "drain_finished_training" in called, (
        "control_tick() no longer drains finished training jobs — a policy trained from a live "
        "mjlab session would need a process restart to become selectable again")
