#!/usr/bin/env python3
"""
The pluggable training-backend registry in legged_gym/control/training.py —
TrainingBackend / BACKENDS / resolve_training_backend() / backend_for_job().

Its whole reason to exist is extensibility: adding a genuinely new place a
training job can run (a CUDA-enabled local venv, a second cloud provider)
must be ONE descriptor entry plus its own hooks, with NO edit to
TrainingManager.start()'s validation/dispatch. The load-bearing test here is
test_a_brand_new_backend_needs_no_core_change, which registers a synthetic
descriptor at runtime and drives a real start() through it end to end.

Everything else pins the three real backends' persisted/observable shape so
this refactor can't silently drift them:
  - which (requested backend, task stack) pair resolves to which descriptor;
  - the TrainingJob.backend / .simulator values written to meta.json;
  - the two rejection messages start() has always raised (unknown backend
    name; Kaggle asked for an mjlab task).

Behavioral dispatch assertions (argv/env/flags for genesis & mjlab) live in
tests/test_mjlab_training_dispatch.py and are deliberately NOT duplicated
here — that file is the contract, this one is the registry mechanism.

Run: python -m pytest tests/test_training_backend_registry.py -q
"""
import os
import unittest
from pathlib import Path
from unittest import mock

from legged_gym.control import backends as backends_mod  # noqa: E402
from legged_gym.control import training as training_mod  # noqa: E402
from legged_gym.control.training import (  # noqa: E402
    BACKENDS, REQUESTABLE_BACKENDS, TrainingBackend, TrainingJob, TrainingManager,
    backend_for_job, resolve_training_backend,
)

MJLAB_TASK = "Rugiar-G1-Mimic"
MOTION = "resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz"


class _FakeProc:
    def poll(self):
        return None


def _pin_registries(mjlab_tasks, genesis_tasks):
    """Same helper as tests/test_mjlab_training_dispatch.py's — pins what each
    registry probe reports so these assertions are about the REGISTRY, not
    about which simulator happens to be importable in the venv running the
    suite."""
    return mock.patch.multiple(
        training_mod,
        _mjlab_registered_tasks=mock.Mock(return_value=mjlab_tasks),
        _legged_gym_registered_tasks=mock.Mock(return_value=genesis_tasks),
    )


class TestRegistryShape(unittest.TestCase):
    def test_lookup_key_is_unique(self):
        """(requested_as, task_stack) IS the lookup key — a duplicate would
        make resolve_training_backend() silently prefer whichever entry was
        listed first."""
        keys = [(b.requested_as, b.task_stack) for b in BACKENDS]
        self.assertEqual(len(keys), len(set(keys)), f"duplicate backend lookup key in {keys}")

    def test_job_identity_is_unique_so_backend_for_job_is_unambiguous(self):
        """backend_for_job() recovers a descriptor from an in-flight job's own
        (backend, simulator) pair rather than a field stored on the job, which
        only works while that pair is unique."""
        ids = [(b.job_backend, b.simulator) for b in BACKENDS]
        self.assertEqual(len(ids), len(set(ids)), f"ambiguous (job_backend, simulator) in {ids}")

    def test_requestable_backends_are_derived_not_hardcoded(self):
        self.assertEqual(set(REQUESTABLE_BACKENDS), {b.requested_as for b in BACKENDS})
        self.assertEqual(set(REQUESTABLE_BACKENDS), {"local", "kaggle"})

    def test_local_backends_are_launchable_and_remote_ones_are_not(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend.id):
                if backend.remote:
                    # No interpreter/env of ours to build — the descriptor owns
                    # everything up to "the job is running".
                    self.assertIsNotNone(backend.launch_remote)
                    self.assertFalse(backend.accepts_local_checkpoint,
                                     "a remote backend can't read this machine's filesystem")
                else:
                    self.assertIsNotNone(backend.interpreter)
                    self.assertIsNotNone(backend.prepare_env)
                    self.assertTrue(Path(backend.script).is_file(),
                                    f"{backend.id}'s entrypoint {backend.script} doesn't exist")


class TestResolution(unittest.TestCase):
    def test_each_real_pair_resolves_to_its_own_descriptor(self):
        cases = [
            ("g1", "local", "local-genesis", "local", "genesis"),
            ("g1", "kaggle", "kaggle", "kaggle", "isaacgym"),
            (MJLAB_TASK, "local", "local-mjlab", "local", "mjlab"),
        ]
        for task, requested, backend_id, job_backend, simulator in cases:
            with self.subTest(task=task, requested=requested):
                with _pin_registries({MJLAB_TASK}, {"g1"}):
                    resolved = resolve_training_backend(task, requested)
                self.assertEqual(resolved.id, backend_id)
                # PERSISTED SHAPE — these two land in meta.json and in the
                # UI's job list; changing either breaks already-written jobs.
                self.assertEqual(resolved.job_backend, job_backend)
                self.assertEqual(resolved.simulator, simulator)

    def test_unknown_backend_name_message_is_unchanged(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_training_backend("g1", "gpu-farm")
        self.assertEqual(str(ctx.exception),
                         "unknown backend 'gpu-farm' — must be 'local' or 'kaggle'")

    def test_kaggle_refuses_an_mjlab_task_with_the_documented_message(self):
        with _pin_registries({MJLAB_TASK}, None):
            with self.assertRaises(ValueError) as ctx:
                resolve_training_backend(MJLAB_TASK, "kaggle")
        self.assertEqual(str(ctx.exception),
                         "the Kaggle backend doesn't support mjlab tasks (its bootstrap is "
                         "IsaacGym-specific)")

    def test_an_unreadable_registry_still_refuses_to_guess(self):
        with _pin_registries(None, None):
            with self.assertRaises(ValueError) as ctx:
                resolve_training_backend("g1", "local")
        self.assertIn("neither", str(ctx.exception))


class TestBackendForJob(unittest.TestCase):
    def _job(self, backend, simulator):
        return TrainingJob(
            id="x", policy_name="p", task="g1", command="", log_path="", result_path="",
            progress_path="", started_at=0.0, max_iterations=1, max_minutes=None, num_envs=1,
            backend=backend, simulator=simulator)

    def test_round_trips_every_registered_backend(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend.id):
                job = self._job(backend.job_backend, backend.simulator)
                self.assertIs(backend_for_job(job), backend)

    def test_an_unrecognized_pair_is_none_not_an_exception(self):
        """A job recorded by an older/newer build must not crash poll(), which
        runs on the sim loop — it falls back instead."""
        self.assertIsNone(backend_for_job(self._job("local", "isaacsim")))


@unittest.skipUnless(training_mod.GENESIS_PYTHON.exists(), "no .venv on this machine")
class TestExtensibility(unittest.TestCase):
    """The actual architectural claim: a new backend is a descriptor, not a
    new branch."""

    def test_a_brand_new_backend_needs_no_core_change(self):
        """Registers a synthetic 'local-nvidia'-shaped backend — a new
        requested_as name, its own interpreter/env/validation hooks — and
        drives a REAL TrainingManager.start() through it. start(), its
        validation block and its argv/env assembly are untouched by this
        test; if any of them still special-cased genesis/mjlab/kaggle, this
        could not pass."""
        rejected = []

        def _interpreter(manager, task):
            return "/usr/bin/pretend-cuda-python"

        def _prepare_env(env):
            env["SIMULATOR"] = "pretend"
            env["CUDA_VISIBLE_DEVICES"] = "0"

        def _validate(params):
            rejected.append(params["task"])

        fake = TrainingBackend(
            id="local-pretend-gpu",
            requested_as="pretend-gpu",
            task_stack="genesis",
            job_backend="local",
            simulator="pretend",
            command_prefix="rugiar train --backend pretend-gpu ",
            script=training_mod.TRAIN_SCRIPT,
            fixed_flags=("--device", "cuda:0"),
            interpreter=_interpreter,
            prepare_env=_prepare_env,
            validate_params=_validate,
        )
        # Patched on the registry package (where BACKENDS lives now) AND on
        # training (which imported both names into its own namespace).
        with mock.patch.object(backends_mod, "BACKENDS", BACKENDS + [fake]), \
                mock.patch.object(backends_mod, "REQUESTABLE_BACKENDS",
                                  REQUESTABLE_BACKENDS + ("pretend-gpu",)), \
                mock.patch.object(training_mod, "BACKENDS", BACKENDS + [fake]), \
                mock.patch.object(training_mod, "REQUESTABLE_BACKENDS",
                                  REQUESTABLE_BACKENDS + ("pretend-gpu",)), \
                _pin_registries(None, {"g1"}), \
                mock.patch.object(training_mod.subprocess, "Popen",
                                  return_value=_FakeProc()) as popen:
            tm = TrainingManager(python_exe=str(training_mod.GENESIS_PYTHON))
            job_id = tm.start(policy_name="t_new_backend", task="g1", num_envs=4,
                              max_iterations=2, backend="pretend-gpu")

        argv, popen_kwargs = popen.call_args[0][0], popen.call_args[1]
        job = tm.jobs[job_id]
        self.assertEqual(argv[0], "/usr/bin/pretend-cuda-python")
        self.assertEqual(argv[2], str(training_mod.TRAIN_SCRIPT))
        self.assertEqual(argv[3:5], ["--device", "cuda:0"])   # fixed_flags, in place
        self.assertEqual(popen_kwargs["env"]["SIMULATOR"], "pretend")
        self.assertEqual(popen_kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(job.simulator, "pretend")
        self.assertEqual(job.backend, "local")
        self.assertTrue(job.command.startswith("rugiar train --backend pretend-gpu "))
        self.assertEqual(rejected, ["g1"])  # the descriptor's own validation ran

    def test_an_unregistered_backend_name_is_still_refused(self):
        """The negative half: the registry is the ONLY thing that makes a
        backend name valid."""
        tm = TrainingManager()
        with self.assertRaises(ValueError):
            tm.start(policy_name="t_nope", task="g1", num_envs=4, max_iterations=2,
                     backend="pretend-gpu")


class TestEnvHooksStayOpposites(unittest.TestCase):
    """The two local backends' env preparation is not a shared block with an
    `if` in it — it's two exact opposites (docs/mjlab_migration.md R1), which
    is precisely why prepare_env is a per-descriptor hook."""

    def _prepare(self, backend_id, seed_pythonpath):
        backend = next(b for b in BACKENDS if b.id == backend_id)
        env = {"PYTHONPATH": seed_pythonpath} if seed_pythonpath is not None else {}
        backend.prepare_env(env)
        return env

    def test_genesis_prepends_repo_root(self):
        env = self._prepare("local-genesis", "/somewhere/else")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(training_mod.REPO_ROOT))
        self.assertEqual(env["SIMULATOR"], "genesis")

    def test_mjlab_strips_repo_root_but_keeps_the_rest(self):
        env = self._prepare(
            "local-mjlab", os.pathsep.join([str(training_mod.REPO_ROOT), "/somewhere/else"]))
        self.assertEqual(env["PYTHONPATH"], "/somewhere/else")
        self.assertEqual(env["SIMULATOR"], "mjlab")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

    def test_mjlab_drops_pythonpath_entirely_when_repo_root_was_all_of_it(self):
        env = self._prepare("local-mjlab", str(training_mod.REPO_ROOT))
        self.assertNotIn("PYTHONPATH", env)


if __name__ == "__main__":
    unittest.main()
