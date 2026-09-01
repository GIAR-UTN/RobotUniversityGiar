#!/usr/bin/env python3
"""
Unit tests for legged_gym/control/distillation.py (the pure BC algorithm)
and TrainingManager.start_distillation() (legged_gym/control/training.py,
the disk/job orchestration layer on top of it). Same package-stub trick as
tests/test_fusion.py — legged_gym/__init__.py unconditionally imports a
physics backend (genesis/isaacgym) as a side effect of package import, even
though nothing this test touches actually needs one.

distillation.py itself has zero legged_gym imports (it's pure torch), so
TestDistillationAlgorithm needs no stubbing at all — only
TestStartDistillation (which imports legged_gym.control.training) does.

Run directly: python tests/test_distillation.py
"""
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_package(dotted_name: str):
    if dotted_name in sys.modules:
        return
    module = types.ModuleType(dotted_name)
    module.__path__ = [str(ROOT / Path(*dotted_name.split(".")))]
    module.__package__ = dotted_name
    sys.modules[dotted_name] = module


for _pkg in ("legged_gym", "legged_gym.control"):
    _stub_package(_pkg)

from legged_gym.control import distillation
from legged_gym.control.training import TrainingManager
import legged_gym.control.training as training_mod

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent


class _FakeTeacherBackend:
    """Wraps a real ActorCritic/ActorCriticRecurrent the same way
    legged_gym.control.policy.load_policy_backend() would — deterministic
    act_inference() output, and a real (not no-op) whole-batch reset() so
    tests can actually observe collect_rollout() calling it on episode
    boundaries."""

    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.actor_critic.eval()
        self.reset_calls = 0

    def step(self, obs):
        with torch.no_grad():
            return self.actor_critic.act_inference(obs)

    def reset(self):
        self.reset_calls += 1
        if self.actor_critic.is_recurrent:
            self.actor_critic.memory_a.hidden_states = None
            self.actor_critic.memory_c.hidden_states = None


class _FakeEnv:
    """Just enough of the vectorized-env interface collect_rollout()/
    dagger_train() need (get_observations()/step()/commands/simulator.*) —
    real dynamics are irrelevant to testing the BC plumbing itself.
    `commands`/`simulator.base_lin_vel`/`simulator.base_ang_vel` back
    collect_rollout()'s ground_truth diagnostics (see its own docstring for
    why those are read straight off simulator state instead of decoded from
    obs). `done_at_steps`, if given, is a set of step indices (0-based,
    matching collect_rollout()'s own `step` counter) at which every env
    reports done=True, so tests can exercise the episode-boundary reset
    path."""

    def __init__(self, num_envs, num_obs, done_at_steps=frozenset()):
        self.num_envs, self.num_obs = num_envs, num_obs
        self.obs = torch.randn(num_envs, num_obs)
        self.commands = torch.randn(num_envs, 3)
        self.simulator = types.SimpleNamespace(
            base_lin_vel=torch.randn(num_envs, 3), base_ang_vel=torch.randn(num_envs, 3))
        self.done_at_steps = done_at_steps
        self._step = 0

    def get_observations(self):
        return self.obs

    def step(self, actions):
        self.obs = torch.randn(self.num_envs, self.num_obs)
        self.commands = torch.randn(self.num_envs, 3)
        self.simulator.base_lin_vel = torch.randn(self.num_envs, 3)
        self.simulator.base_ang_vel = torch.randn(self.num_envs, 3)
        dones = torch.full((self.num_envs,), self._step in self.done_at_steps, dtype=torch.bool)
        self._step += 1
        return self.obs, None, None, dones, {}


def _policy_cfg(hidden=(8,), rnn_hidden_size=8):
    return types.SimpleNamespace(
        actor_hidden_dims=list(hidden), critic_hidden_dims=list(hidden), activation="elu",
        init_noise_std=1.0, rnn_type="lstm", rnn_hidden_size=rnn_hidden_size, rnn_num_layers=1,
    )


class TestDistillationAlgorithm(unittest.TestCase):
    def test_check_dimensions_compatible_accepts_a_matching_teacher(self):
        torch.manual_seed(0)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        distillation.check_dimensions_compatible(teacher, num_obs=6, num_actions=3, num_envs=4)  # no raise

    def test_check_dimensions_compatible_rejects_wrong_obs_size(self):
        torch.manual_seed(0)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        with self.assertRaises(ValueError):
            distillation.check_dimensions_compatible(teacher, num_obs=10, num_actions=3, num_envs=4)

    def test_check_dimensions_compatible_rejects_wrong_action_size(self):
        class _WrongActionCountBackend:
            def step(self, obs):
                return torch.zeros(obs.shape[0], 99)

            def reset(self):
                pass

        with self.assertRaises(ValueError):
            distillation.check_dimensions_compatible(_WrongActionCountBackend(), num_obs=6, num_actions=3, num_envs=4)

    def test_build_student_non_recurrent_matches_target_dims(self):
        student = distillation.build_student(6, 6, 3, _policy_cfg(), is_recurrent=False)
        self.assertIsInstance(student, ActorCritic)
        self.assertFalse(student.is_recurrent)
        out = student.act_inference(torch.zeros(2, 6))
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_build_student_recurrent_matches_target_dims(self):
        student = distillation.build_student(6, 6, 3, _policy_cfg(), is_recurrent=True)
        self.assertIsInstance(student, ActorCriticRecurrent)
        self.assertTrue(student.is_recurrent)
        out = student.act_inference(torch.zeros(2, 6))
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_collect_rollout_returns_correctly_shaped_buffers(self):
        torch.manual_seed(0)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        env = _FakeEnv(num_envs=4, num_obs=6)
        obs_buf, action_buf, dones_buf, ground_truth = distillation.collect_rollout(env, teacher, num_steps=10)
        self.assertEqual(tuple(obs_buf.shape), (10, 4, 6))
        self.assertEqual(tuple(action_buf.shape), (10, 4, 3))
        self.assertEqual(tuple(dones_buf.shape), (10, 4))

    def test_collect_rollout_invokes_callback_every_step(self):
        torch.manual_seed(0)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        env = _FakeEnv(num_envs=2, num_obs=6)
        seen = []
        distillation.collect_rollout(env, teacher, num_steps=5, callback=lambda step, total: seen.append((step, total)))
        self.assertEqual(seen, [(i, 5) for i in range(5)])

    def test_collect_rollout_does_not_reset_teacher_absent_dones(self):
        torch.manual_seed(0)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        env = _FakeEnv(num_envs=1, num_obs=6)
        distillation.collect_rollout(env, teacher, num_steps=10)
        self.assertEqual(teacher.reset_calls, 0)

    def test_collect_rollout_resets_teacher_on_episode_boundaries(self):
        torch.manual_seed(0)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        env = _FakeEnv(num_envs=1, num_obs=6, done_at_steps={2, 5})
        obs_buf, action_buf, dones_buf, ground_truth = distillation.collect_rollout(env, teacher, num_steps=10)
        self.assertEqual(teacher.reset_calls, 2)
        self.assertTrue(bool(dones_buf[2, 0]))
        self.assertTrue(bool(dones_buf[5, 0]))
        self.assertFalse(bool(dones_buf[0, 0]))

    def test_bc_train_non_recurrent_loss_decreases(self):
        torch.manual_seed(0)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        env = _FakeEnv(num_envs=4, num_obs=6)
        obs_buf, action_buf, _, _ = distillation.collect_rollout(env, teacher, num_steps=20)
        student = distillation.build_student(6, 6, 3, _policy_cfg(), is_recurrent=False)
        losses = []
        final = distillation.bc_train(student, obs_buf, action_buf, epochs=25, lr=1e-2, num_mini_batches=2,
                                       callback=lambda epoch, loss: losses.append(loss))
        self.assertEqual(len(losses), 25)
        self.assertLess(final, losses[0])

    def test_bc_train_recurrent_loss_decreases_and_resets_hidden_state_each_epoch(self):
        torch.manual_seed(1)
        teacher = _FakeTeacherBackend(ActorCriticRecurrent(
            num_actor_obs=6, num_critic_obs=6, num_actions=3, actor_hidden_dims=[8], critic_hidden_dims=[8],
            rnn_type="lstm", rnn_hidden_size=8, rnn_num_layers=1))
        env = _FakeEnv(num_envs=3, num_obs=6)
        obs_buf, action_buf, dones_buf, _ = distillation.collect_rollout(env, teacher, num_steps=30)
        student = distillation.build_student(6, 6, 3, _policy_cfg(), is_recurrent=True)
        losses = []
        final = distillation.bc_train(student, obs_buf, action_buf, epochs=12, lr=1e-2, chunk_len=10,
                                       dones_buf=dones_buf,
                                       callback=lambda epoch, loss: losses.append(loss))
        self.assertEqual(len(losses), 12)
        self.assertLess(final, losses[0])
        # A fresh epoch starts from a zeroed hidden state (student.reset() at the top of
        # bc_train()'s recurrent branch) — after training, an explicit reset() must still
        # zero it out, proving the reset call is wired to the real memory modules.
        student.reset(dones=torch.ones(3, dtype=torch.bool))
        self.assertTrue(all(torch.all(h == 0) for h in student.memory_a.hidden_states))

    def test_bc_train_recurrent_resets_hidden_state_at_episode_boundaries(self):
        obs_buf = torch.randn(6, 2, 6)
        action_buf = torch.randn(6, 2, 3)
        # env 0 "resets" after step 2; env 1 never does.
        dones_buf = torch.zeros(6, 2, dtype=torch.bool)
        dones_buf[2, 0] = True

        # Two identically-initialized students trained on the SAME buffers —
        # one told about the episode boundary, one not. If mid-chunk resets
        # are wired up, the boundary changes what the recurrent student
        # actually learns, so the two must diverge.
        torch.manual_seed(2)
        student_a = distillation.build_student(6, 6, 3, _policy_cfg(), is_recurrent=True)
        torch.manual_seed(2)
        student_b = distillation.build_student(6, 6, 3, _policy_cfg(), is_recurrent=True)
        distillation.bc_train(student_a, obs_buf, action_buf, epochs=1, chunk_len=6, dones_buf=dones_buf)
        distillation.bc_train(student_b, obs_buf, action_buf, epochs=1, chunk_len=6, dones_buf=None)
        params_a = torch.cat([p.detach().flatten() for p in student_a.parameters()])
        params_b = torch.cat([p.detach().flatten() for p in student_b.parameters()])
        self.assertFalse(torch.allclose(params_a, params_b))

    def test_dagger_train_returns_correctly_shaped_aggregated_buffers(self):
        torch.manual_seed(3)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        env = _FakeEnv(num_envs=2, num_obs=6)
        student = distillation.build_student(6, 6, 3, _policy_cfg(rnn_hidden_size=8), is_recurrent=False)
        final_loss, obs_buf, action_buf, dones_buf, ground_truth, loss_curve, round_diagnostics = distillation.dagger_train(
            env, teacher, student, num_rounds=3, round_steps=5, bc_epochs=2, lr=1e-2)
        # 3 rounds x 5 steps each, aggregated (not just the latest round) — see this
        # function's own docstring on why vanilla DAgger keeps every round's data.
        self.assertEqual(tuple(obs_buf.shape), (15, 2, 6))
        self.assertEqual(tuple(action_buf.shape), (15, 2, 3))
        self.assertEqual(tuple(dones_buf.shape), (15, 2))
        self.assertEqual(tuple(ground_truth["commands"].shape), (15, 2, 3))
        self.assertIsInstance(final_loss, float)
        # loss_curve: one point per (round, epoch) — 3 rounds x 2 epochs each.
        self.assertEqual(len(loss_curve), 6)
        self.assertEqual([p["round"] for p in loss_curve], [0, 0, 1, 1, 2, 2])
        self.assertEqual([p["epoch"] for p in loss_curve], [0, 1, 0, 1, 0, 1])
        # round_diagnostics: one summary per round, computed from THAT round's own
        # (obs, teacher-relabeled action) pairs, not the aggregate.
        self.assertEqual(len(round_diagnostics), 3)
        self.assertEqual([r["round"] for r in round_diagnostics], [0, 1, 2])
        for r in round_diagnostics:
            self.assertIn("action_abs_mean", r)
            self.assertIn("commanded_lin_vel_x", r)

    def test_dagger_train_marks_round_boundaries_as_done(self):
        # Round boundaries reset both backends' hidden state but env.step() itself never
        # reports a done there — dagger_train() must force that boundary into dones_buf
        # itself (see its own docstring) so bc_train()'s chunked hidden-state resets stay
        # aligned with what collection actually did.
        torch.manual_seed(4)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        env = _FakeEnv(num_envs=1, num_obs=6)  # never reports done on its own
        student = distillation.build_student(6, 6, 3, _policy_cfg(), is_recurrent=False)
        _, _, _, dones_buf, _, _, _ = distillation.dagger_train(
            env, teacher, student, num_rounds=3, round_steps=4, bc_epochs=1, lr=1e-2)
        self.assertTrue(bool(dones_buf[0, 0]))    # round 0 boundary
        self.assertTrue(bool(dones_buf[4, 0]))    # round 1 boundary
        self.assertTrue(bool(dones_buf[8, 0]))    # round 2 boundary
        self.assertFalse(bool(dones_buf[1, 0]))

    def test_dagger_train_beta_zero_round_steps_env_with_student_actions(self):
        # With beta0=0 (never the teacher), every action actually fed to env.step() must be
        # the STUDENT's — proves the mix genuinely branches on beta rather than always
        # leaning on the teacher regardless of the requested fraction. lr=0.0 keeps the
        # student's weights (and thus its action for a given obs) fixed across the single
        # round's retrain, so the comparison below isn't chasing a moving target.
        torch.manual_seed(5)
        teacher = _FakeTeacherBackend(ActorCritic(num_actor_obs=6, num_critic_obs=6, num_actions=3,
                                                    actor_hidden_dims=[8], critic_hidden_dims=[8]))
        student = distillation.build_student(6, 6, 3, _policy_cfg(), is_recurrent=False)

        class _SpyEnv(_FakeEnv):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.seen_actions = []

            def step(self, actions):
                self.seen_actions.append(actions.detach().clone())
                return super().step(actions)

        env = _SpyEnv(num_envs=4, num_obs=6)
        # lr=0.0 keeps the student's weights fixed across the round's retrain, so replaying
        # the SAME obs sequence dagger_train() recorded (obs_buf) through the (unmoved)
        # student afterward reproduces exactly what should have been fed to env.step() if
        # (and only if) the mix picked the student's action at every one of these steps.
        _, obs_buf, _, _, _, _, _ = distillation.dagger_train(
            env, teacher, student, num_rounds=1, round_steps=6, bc_epochs=1, lr=0.0, beta0=0.0)

        self.assertEqual(len(env.seen_actions), 6)
        student.reset(dones=torch.ones(4, dtype=torch.bool))
        with torch.no_grad():
            for t in range(6):
                expected = student.act_inference(obs_buf[t])
                self.assertTrue(torch.allclose(expected, env.seen_actions[t]))


class TestStartDistillation(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_policies_dir = training_mod.POLICIES_DIR
        self._orig_jobs_dir = training_mod.JOBS_DIR
        training_mod.POLICIES_DIR = Path(self._tmp.name) / "policies"
        training_mod.JOBS_DIR = Path(self._tmp.name) / "jobs"
        training_mod.POLICIES_DIR.mkdir(parents=True)
        training_mod.JOBS_DIR.mkdir(parents=True)

    def tearDown(self):
        training_mod.POLICIES_DIR = self._orig_policies_dir
        training_mod.JOBS_DIR = self._orig_jobs_dir
        self._tmp.cleanup()

    def _mgr(self, sources: dict) -> TrainingManager:
        mgr = TrainingManager.__new__(TrainingManager)
        mgr.policy_sources = dict(sources)
        mgr.jobs = {}
        mgr._procs = {}
        mgr._log_files = {}
        mgr.python_exe = sys.executable
        return mgr

    def test_unknown_teacher_raises(self):
        mgr = self._mgr({})
        with self.assertRaises(ValueError):
            mgr.start_distillation("nope", "g1", "out")

    def test_teacher_without_checkpoint_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": None, "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "out")

    def test_output_name_collision_raises(self):
        (training_mod.POLICIES_DIR / "taken").mkdir(parents=True)
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "taken")

    def test_reserved_name_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "damping")

    def test_unknown_method_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "out", method="nope")

    def test_planned_but_unavailable_method_raises(self):
        # dagger is real now (distillation.DISTILL_METHODS["dagger"]["available"] is True)
        # — exercise the "planned, not implemented" rejection path itself instead, via a
        # temporary fake entry, so this guard stays covered even with every real method available.
        from legged_gym.control import distillation
        distillation.DISTILL_METHODS["_test_planned"] = {"label": "x", "description": "x", "available": False}
        try:
            mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
            with self.assertRaises(ValueError):
                mgr.start_distillation("t", "g1", "out", method="_test_planned")
        finally:
            del distillation.DISTILL_METHODS["_test_planned"]

    def test_dagger_method_is_accepted(self):
        import unittest.mock as mock
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        fake_proc = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        with mock.patch.object(training_mod.subprocess, "Popen", return_value=fake_proc) as popen:
            job_id = mgr.start_distillation(
                "t", "g1", "out", method="dagger", dagger_rounds=3, dagger_beta0=0.8, dagger_beta_decay=0.4)
        argv = popen.call_args.args[0]
        self.assertIn("--dagger_rounds", argv)
        self.assertEqual(argv[argv.index("--dagger_rounds") + 1], "3")
        self.assertIn("--dagger_beta0", argv)
        self.assertIn("--dagger_beta_decay", argv)
        self.assertEqual(mgr.jobs[job_id].distill_method, "dagger")
        # max_iterations must be the TOTAL bc epochs across all rounds (dagger_rounds *
        # bc_epochs), not just one round's — see start_distillation()'s own comment on why.
        self.assertEqual(mgr.jobs[job_id].max_iterations, 3 * 20)  # bc_epochs defaults to 20
        mgr._log_files[job_id].close()

    def test_dagger_invalid_rounds_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "out", method="dagger", dagger_rounds=0)

    def test_dagger_invalid_beta_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "out", method="dagger", dagger_beta0=1.5)

    def test_empty_out_name_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "  ")

    def test_nonpositive_rollout_steps_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "out", rollout_steps=0)

    def test_nonpositive_bc_epochs_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "out", bc_epochs=-1)

    def test_nonpositive_lr_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "out", lr=0)

    def test_nonpositive_num_envs_raises(self):
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with self.assertRaises(ValueError):
            mgr.start_distillation("t", "g1", "out", num_envs=0)

    def test_teacher_with_no_train_checkpoint_is_allowed(self):
        # The whole point of distillation — unlike fuse_policies(), a teacher
        # with checkpoint.pt but NO train_checkpoint.pt (e.g. 'stable') must
        # be accepted, not rejected. subprocess.Popen() is faked out (no real
        # simulator available in this test environment) so this only
        # exercises start_distillation()'s own validation/bookkeeping, not
        # web_distill.py itself.
        import unittest.mock as mock
        mgr = self._mgr({"stable": {"task": "g1", "checkpoint": "/some/checkpoint.pt", "train_checkpoint": None}})
        fake_proc = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        with mock.patch.object(training_mod.subprocess, "Popen", return_value=fake_proc) as popen:
            job_id = mgr.start_distillation("stable", "g1", "stable_distilled", num_envs=2)
        popen.assert_called_once()
        argv = popen.call_args.args[0]
        self.assertIn("--teacher_checkpoint", argv)
        self.assertIn("/some/checkpoint.pt", argv)
        self.assertIn(job_id, mgr.jobs)
        self.assertEqual(mgr.jobs[job_id].job_type, "distill")
        self.assertEqual(mgr.jobs[job_id].teacher_policy, "stable")
        self.assertEqual(mgr.jobs[job_id].policy_name, "stable_distilled")
        mgr._log_files[job_id].close()

    def test_gpu_flag_replaces_cpu_in_argv(self):
        """gpu=True must run web_distill.py with --gpu (never --cpu), and
        surface as --gpu on the job's copyable rugiar command — the distill
        counterpart of the local-nvidia backend's --gpu dispatch."""
        import unittest.mock as mock
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        fake_proc = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        with mock.patch("legged_gym.control.cuda_utils.cuda_is_usable",
                        return_value=(True, "")) as probe, \
                mock.patch.object(training_mod.subprocess, "Popen", return_value=fake_proc) as popen:
            job_id = mgr.start_distillation("t", "g1", "out_gpu", num_envs=2, gpu=True)
        probe.assert_called_once()  # the preflight ran before anything was launched
        argv = popen.call_args.args[0]
        self.assertIn("--gpu", argv)
        self.assertNotIn("--cpu", argv)
        self.assertIn("--gpu", mgr.jobs[job_id].command)
        mgr._log_files[job_id].close()

    def test_cpu_is_the_default(self):
        import unittest.mock as mock
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        fake_proc = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        with mock.patch.object(training_mod.subprocess, "Popen", return_value=fake_proc) as popen:
            job_id = mgr.start_distillation("t", "g1", "out_cpu", num_envs=2)
        argv = popen.call_args.args[0]
        self.assertIn("--cpu", argv)
        self.assertNotIn("--gpu", argv)
        self.assertNotIn("--gpu", mgr.jobs[job_id].command)
        mgr._log_files[job_id].close()

    def test_gpu_refused_when_cuda_is_unusable(self):
        """An enumerated-but-unusable GPU must be a clean up-front rejection
        (the same cuda_is_usable trap the local-nvidia backend guards), never
        a launch followed by a mid-`gs.init` crash."""
        import unittest.mock as mock
        mgr = self._mgr({"t": {"task": "g1", "checkpoint": "x", "train_checkpoint": None}})
        with mock.patch("legged_gym.control.cuda_utils.cuda_is_usable",
                        return_value=(False, "torch.cuda.is_available() is False")), \
                mock.patch.object(training_mod.subprocess, "Popen") as popen:
            with self.assertRaises(ValueError) as ctx:
                mgr.start_distillation("t", "g1", "out", num_envs=2, gpu=True)
        self.assertIn("GPU", str(ctx.exception))
        popen.assert_not_called()  # refused before anything was launched
        self.assertEqual(mgr.jobs, {})


if __name__ == "__main__":
    unittest.main()
