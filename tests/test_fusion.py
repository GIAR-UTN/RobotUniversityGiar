#!/usr/bin/env python3
"""
Unit tests for legged_gym/control/fusion.py (the pure merge/architecture
algorithm) and TrainingManager.fuse_policies() (legged_gym/control/
training.py, the disk/policies/<name>/ orchestration layer on top of it).
Same package-stub trick as tests/test_delete_policy.py — legged_gym/
__init__.py unconditionally imports a physics backend (genesis/isaacgym) as
a side effect of package import, even though nothing this test touches
actually needs one.

TestFusePolicies additionally fakes `legged_gym.utils.task_registry` (a
tiny in-test module with just the get_cfgs() this code needs) instead of
importing the real one, for the same reason — the real task_registry
transitively needs `legged_gym.envs`, which needs a real simulator
installed. It also swaps fusion.export_actor_critic() for a version that
calls the REAL legged_gym.utils.helpers.PolicyExporter directly instead of
going through legged_gym.scripts.play.export_policy() (which does `from
legged_gym.envs import *` at module level) — so these tests still produce
and load a genuine TorchScript checkpoint, just without exercising the
task_type-dispatch line in export_actor_critic() itself (that line is a
straight pass-through to play.export_policy(), already exercised by every
normal training run in this repo).

Run directly: python tests/test_fusion.py
"""
import json
import sys
import tempfile
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

from legged_gym.control import fusion
from legged_gym.control.training import TrainingManager
import legged_gym.control.training as training_mod

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent


def _tiny_actor_critic(seed: int, num_actor_obs=6, num_critic_obs=6, num_actions=2, hidden=(8,)):
    torch.manual_seed(seed)
    return ActorCritic(num_actor_obs=num_actor_obs, num_critic_obs=num_critic_obs, num_actions=num_actions,
                        actor_hidden_dims=list(hidden), critic_hidden_dims=list(hidden))


def _write_policy(policies_dir: Path, name: str, actor_critic, task: str = "g1") -> Path:
    policy_dir = policies_dir / name
    policy_dir.mkdir(parents=True)
    torch.save({"model_state_dict": actor_critic.state_dict(), "optimizer_state_dict": {},
                "iter": 100, "infos": {}}, policy_dir / "train_checkpoint.pt")
    (policy_dir / "checkpoint.pt").write_bytes(b"fake - fuse_policies() never reads a source's checkpoint.pt")
    with open(policy_dir / "meta.json", "w") as f:
        json.dump({"task": task}, f)
    return policy_dir


class TestMergeStateDicts(unittest.TestCase):
    def test_uniform_average_of_identical_dicts_matches_the_original(self):
        sd = {"w": torch.tensor([1.0, 2.0, 3.0])}
        merged = fusion.merge_state_dicts([sd, sd], [1, 1])
        self.assertTrue(torch.allclose(merged["w"], sd["w"]))

    def test_weighted_average_matches_hand_computed_values(self):
        sd_a = {"w": torch.tensor([0.0, 0.0])}
        sd_b = {"w": torch.tensor([10.0, 20.0])}
        merged = fusion.merge_state_dicts([sd_a, sd_b], [3, 1])  # normalized: 0.75/0.25
        self.assertTrue(torch.allclose(merged["w"], torch.tensor([2.5, 5.0])))

    def test_raises_on_missing_key(self):
        with self.assertRaises(ValueError):
            fusion.merge_state_dicts([{"w": torch.zeros(2)}, {}], [1, 1])

    def test_raises_on_shape_mismatch(self):
        with self.assertRaises(ValueError):
            fusion.merge_state_dicts([{"w": torch.zeros(2)}, {"w": torch.zeros(3)}], [1, 1])

    def test_raises_on_non_positive_total_weight(self):
        with self.assertRaises(ValueError):
            fusion.merge_state_dicts([{"w": torch.zeros(2)}, {"w": torch.zeros(2)}], [1, -1])

    def test_raises_with_fewer_than_two_state_dicts(self):
        with self.assertRaises(ValueError):
            fusion.merge_state_dicts([{"w": torch.zeros(2)}], [1])


class TestInferArchitecture(unittest.TestCase):
    def test_recovers_plain_actor_critic_dims(self):
        ac = ActorCritic(num_actor_obs=10, num_critic_obs=10, num_actions=4,
                          actor_hidden_dims=[16, 16], critic_hidden_dims=[16, 16])
        arch = fusion.infer_architecture(ac.state_dict())
        self.assertEqual(arch, {
            "is_recurrent": False, "num_actions": 4,
            "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
            "num_actor_obs": 10, "num_critic_obs": 10,
        })

    def test_recovers_recurrent_dims(self):
        rec = ActorCriticRecurrent(num_actor_obs=8, num_critic_obs=8, num_actions=3,
                                    actor_hidden_dims=[16], critic_hidden_dims=[16],
                                    rnn_type="lstm", rnn_hidden_size=12, rnn_num_layers=1)
        arch = fusion.infer_architecture(rec.state_dict())
        self.assertTrue(arch["is_recurrent"])
        self.assertEqual(arch["rnn_type"], "lstm")
        self.assertEqual(arch["rnn_hidden_size"], 12)
        self.assertEqual(arch["rnn_num_layers"], 1)
        self.assertEqual(arch["num_actor_obs"], 8)
        self.assertEqual(arch["num_actions"], 3)

    def test_recovers_gru_recurrent_dims(self):
        rec = ActorCriticRecurrent(num_actor_obs=5, num_critic_obs=5, num_actions=2,
                                    actor_hidden_dims=[8], critic_hidden_dims=[8],
                                    rnn_type="gru", rnn_hidden_size=6, rnn_num_layers=1)
        arch = fusion.infer_architecture(rec.state_dict())
        self.assertEqual(arch["rnn_type"], "gru")

    def test_raises_on_non_actor_critic_state_dict(self):
        with self.assertRaises(ValueError):
            fusion.infer_architecture({"unrelated": torch.zeros(2)})


class TestArchitecturesCompatible(unittest.TestCase):
    def test_matching_architectures_are_compatible(self):
        ac1 = _tiny_actor_critic(1)
        ac2 = _tiny_actor_critic(2)
        a = fusion.infer_architecture(ac1.state_dict())
        b = fusion.infer_architecture(ac2.state_dict())
        self.assertIsNone(fusion.architectures_compatible(a, b))

    def test_differing_hidden_dims_are_incompatible(self):
        a = fusion.infer_architecture(_tiny_actor_critic(1, hidden=(8,)).state_dict())
        b = fusion.infer_architecture(_tiny_actor_critic(2, hidden=(16,)).state_dict())
        self.assertIsNotNone(fusion.architectures_compatible(a, b))

    def test_differing_obs_dims_are_incompatible(self):
        a = fusion.infer_architecture(_tiny_actor_critic(1, num_actor_obs=6).state_dict())
        b = fusion.infer_architecture(_tiny_actor_critic(2, num_actor_obs=9).state_dict())
        self.assertIsNotNone(fusion.architectures_compatible(a, b))

    def test_recurrent_vs_non_recurrent_is_incompatible(self):
        a = fusion.infer_architecture(_tiny_actor_critic(1).state_dict())
        rec = ActorCriticRecurrent(num_actor_obs=6, num_critic_obs=6, num_actions=2,
                                    actor_hidden_dims=[8], critic_hidden_dims=[8], rnn_hidden_size=8)
        b = fusion.infer_architecture(rec.state_dict())
        self.assertIsNotNone(fusion.architectures_compatible(a, b))


class TestRebasinAlign(unittest.TestCase):
    def test_permutation_is_a_pure_relabeling_output_is_unchanged(self):
        ac = _tiny_actor_critic(1, hidden=(8, 8))
        ref = _tiny_actor_critic(2, hidden=(8, 8)).state_dict()
        aligned = fusion.rebasin_align(ref, ac.state_dict())

        arch = fusion.infer_architecture(ac.state_dict())
        rebuilt = fusion.build_actor_critic(arch, aligned)
        obs = torch.randn(4, 6)
        self.assertTrue(torch.allclose(rebuilt.actor(obs), ac.actor(obs), atol=1e-5))

    def test_aligning_a_policy_with_itself_yields_the_identity_permutation(self):
        ac = _tiny_actor_critic(1, hidden=(8, 8))
        sd = ac.state_dict()
        aligned = fusion.rebasin_align(sd, sd)
        for key, value in sd.items():
            self.assertTrue(torch.allclose(aligned[key], value), key)

    def test_recurrent_permutation_is_a_pure_relabeling_output_is_unchanged(self):
        torch.manual_seed(1)
        rec = ActorCriticRecurrent(num_actor_obs=6, num_critic_obs=6, num_actions=2,
                                    actor_hidden_dims=[8], critic_hidden_dims=[8],
                                    rnn_type="lstm", rnn_hidden_size=8, rnn_num_layers=1)
        torch.manual_seed(2)
        ref = ActorCriticRecurrent(num_actor_obs=6, num_critic_obs=6, num_actions=2,
                                    actor_hidden_dims=[8], critic_hidden_dims=[8],
                                    rnn_type="lstm", rnn_hidden_size=8, rnn_num_layers=1).state_dict()

        aligned = fusion.rebasin_align(ref, rec.state_dict())
        arch = fusion.infer_architecture(rec.state_dict())
        rebuilt = fusion.build_actor_critic(arch, aligned)

        obs = torch.randn(3, 6)
        self.assertTrue(torch.allclose(rebuilt.act_inference(obs), rec.act_inference(obs), atol=1e-4))

    def test_recurrent_permutation_also_works_for_gru(self):
        torch.manual_seed(1)
        rec = ActorCriticRecurrent(num_actor_obs=6, num_critic_obs=6, num_actions=2,
                                    actor_hidden_dims=[8], critic_hidden_dims=[8],
                                    rnn_type="gru", rnn_hidden_size=8, rnn_num_layers=1)
        torch.manual_seed(2)
        ref = ActorCriticRecurrent(num_actor_obs=6, num_critic_obs=6, num_actions=2,
                                    actor_hidden_dims=[8], critic_hidden_dims=[8],
                                    rnn_type="gru", rnn_hidden_size=8, rnn_num_layers=1).state_dict()

        aligned = fusion.rebasin_align(ref, rec.state_dict())
        arch = fusion.infer_architecture(rec.state_dict())
        rebuilt = fusion.build_actor_critic(arch, aligned)

        obs = torch.randn(3, 6)
        self.assertTrue(torch.allclose(rebuilt.act_inference(obs), rec.act_inference(obs), atol=1e-4))

    def test_aligning_a_recurrent_policy_with_itself_yields_the_identity_permutation(self):
        rec = ActorCriticRecurrent(num_actor_obs=6, num_critic_obs=6, num_actions=2,
                                    actor_hidden_dims=[8], critic_hidden_dims=[8],
                                    rnn_type="lstm", rnn_hidden_size=8, rnn_num_layers=1)
        sd = rec.state_dict()
        aligned = fusion.rebasin_align(sd, sd)
        for key, value in sd.items():
            self.assertTrue(torch.allclose(aligned[key], value), key)


class TestFusePolicies(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_policies_dir = training_mod.POLICIES_DIR
        training_mod.POLICIES_DIR = Path(self._tmp.name)

        self._fake_env_cfg = types.SimpleNamespace(env=types.SimpleNamespace(num_observations=6, num_actions=2))
        self._fake_train_cfg = types.SimpleNamespace(
            policy=types.SimpleNamespace(activation="elu"),
            runner=types.SimpleNamespace(load_run=-1, checkpoint=-1),
        )
        fake_task_registry = types.ModuleType("legged_gym.utils.task_registry")
        fake_task_registry.get_cfgs = lambda name: (self._fake_env_cfg, self._fake_train_cfg)
        _stub_package("legged_gym.utils")
        sys.modules["legged_gym.utils.task_registry"] = fake_task_registry
        # `from legged_gym.utils import task_registry` resolves via getattr() first, which
        # finds the REAL task_registry if some other test module (e.g. one requiring the
        # real simulator) already imported legged_gym.utils before this test ran — the
        # sys.modules entry above alone isn't enough in that case. Force the attribute too.
        utils_module = sys.modules["legged_gym.utils"]
        self._had_real_task_registry_attr = hasattr(utils_module, "task_registry")
        self._orig_task_registry_attr = getattr(utils_module, "task_registry", None)
        utils_module.task_registry = fake_task_registry

        from legged_gym.utils.helpers import PolicyExporter, PolicyExporterLSTM

        def _fake_export(actor_critic, out_dir, env_cfg, train_cfg, task_type, export_onnx=False):
            # Same dispatch play.export_policy() does — a recurrent actor_critic needs
            # PolicyExporterLSTM (bundles the LSTM in), not plain PolicyExporter (which
            # would silently drop it and export an MLP expecting hidden-sized input).
            if getattr(actor_critic, "is_recurrent", False):
                PolicyExporterLSTM(actor_critic).export(out_dir, env_cfg, export_onnx=export_onnx)
            else:
                PolicyExporter(actor_critic).export(out_dir, env_cfg, export_onnx=export_onnx, train_cfg=train_cfg)
            produced = list(Path(out_dir).glob("*.pt"))
            return str(produced[0])

        self._orig_export = fusion.export_actor_critic
        fusion.export_actor_critic = _fake_export

    def tearDown(self):
        training_mod.POLICIES_DIR = self._orig_policies_dir
        fusion.export_actor_critic = self._orig_export
        sys.modules.pop("legged_gym.utils.task_registry", None)
        utils_module = sys.modules.get("legged_gym.utils")
        if utils_module is not None:
            if self._had_real_task_registry_attr:
                utils_module.task_registry = self._orig_task_registry_attr
            else:
                del utils_module.task_registry
        self._tmp.cleanup()

    def _mgr(self, sources: dict) -> TrainingManager:
        mgr = TrainingManager.__new__(TrainingManager)
        mgr.policy_sources = dict(sources)
        return mgr

    def _source(self, policy_dir: Path, task: str = "g1") -> dict:
        return {"task": task, "checkpoint": str(policy_dir / "checkpoint.pt"),
                "train_checkpoint": str(policy_dir / "train_checkpoint.pt")}

    def test_fuses_two_policies_into_a_loadable_checkpoint(self):
        dir_a = _write_policy(training_mod.POLICIES_DIR, "a", _tiny_actor_critic(1))
        dir_b = _write_policy(training_mod.POLICIES_DIR, "b", _tiny_actor_critic(2))
        mgr = self._mgr({"a": self._source(dir_a), "b": self._source(dir_b)})

        result = mgr.fuse_policies(["a", "b"], "fused", weights=[0.5, 0.5])

        self.assertEqual(result["name"], "fused")
        self.assertEqual(result["warnings"], [])
        fused_dir = training_mod.POLICIES_DIR / "fused"
        self.assertTrue((fused_dir / "checkpoint.pt").is_file())
        self.assertTrue((fused_dir / "train_checkpoint.pt").is_file())

        loaded = torch.jit.load(str(fused_dir / "checkpoint.pt"))
        out = loaded(torch.zeros(1, 6))
        self.assertEqual(tuple(out.shape), (1, 2))

        with open(fused_dir / "meta.json") as f:
            meta = json.load(f)
        self.assertEqual(meta["task"], "g1")
        self.assertEqual(meta["trained_via"], "fusion")
        self.assertEqual(meta["fusion"]["method"], "weighted_average")
        self.assertEqual([s["name"] for s in meta["fusion"]["sources"]], ["a", "b"])
        self.assertIn("fused", mgr.policy_sources)
        self.assertEqual(mgr.policy_sources["fused"]["task"], "g1")

    def test_merging_a_policy_with_itself_reproduces_its_own_weights(self):
        ac = _tiny_actor_critic(3)
        policy_dir = _write_policy(training_mod.POLICIES_DIR, "solo", ac)
        mgr = self._mgr({"solo": self._source(policy_dir)})

        mgr.fuse_policies(["solo", "solo"], "solo_x2", weights=[1, 1])

        fused_state = torch.load(training_mod.POLICIES_DIR / "solo_x2" / "train_checkpoint.pt",
                                  map_location="cpu", weights_only=False)["model_state_dict"]
        for key, value in ac.state_dict().items():
            self.assertTrue(torch.allclose(fused_state[key], value), key)

    def test_unknown_source_raises(self):
        mgr = self._mgr({})
        with self.assertRaises(ValueError):
            mgr.fuse_policies(["nope", "also_nope"], "out")

    def test_source_without_train_checkpoint_raises(self):
        mgr = self._mgr({
            "a": {"task": "g1", "checkpoint": "x", "train_checkpoint": None},
            "b": {"task": "g1", "checkpoint": "x", "train_checkpoint": None},
        })
        with self.assertRaises(ValueError):
            mgr.fuse_policies(["a", "b"], "out")

    def test_needs_at_least_two_policies(self):
        dir_a = _write_policy(training_mod.POLICIES_DIR, "a", _tiny_actor_critic(1))
        mgr = self._mgr({"a": self._source(dir_a)})
        with self.assertRaises(ValueError):
            mgr.fuse_policies(["a"], "out")

    def test_output_name_collision_raises(self):
        dir_a = _write_policy(training_mod.POLICIES_DIR, "a", _tiny_actor_critic(1))
        dir_b = _write_policy(training_mod.POLICIES_DIR, "b", _tiny_actor_critic(2))
        _write_policy(training_mod.POLICIES_DIR, "taken", _tiny_actor_critic(1))
        mgr = self._mgr({"a": self._source(dir_a), "b": self._source(dir_b)})
        with self.assertRaises(ValueError):
            mgr.fuse_policies(["a", "b"], "taken")

    def test_mismatched_architectures_raise(self):
        dir_small = _write_policy(training_mod.POLICIES_DIR, "small", _tiny_actor_critic(1, hidden=(8,)))
        dir_big = _write_policy(training_mod.POLICIES_DIR, "big", _tiny_actor_critic(2, hidden=(32,)))
        mgr = self._mgr({"small": self._source(dir_small), "big": self._source(dir_big)})
        with self.assertRaises(ValueError):
            mgr.fuse_policies(["small", "big"], "out")

    def test_differing_tasks_succeed_with_a_warning(self):
        dir_a = _write_policy(training_mod.POLICIES_DIR, "a", _tiny_actor_critic(1), task="g1")
        dir_b = _write_policy(training_mod.POLICIES_DIR, "b", _tiny_actor_critic(2), task="g1_other")
        mgr = self._mgr({"a": self._source(dir_a, task="g1"), "b": self._source(dir_b, task="g1_other")})

        result = mgr.fuse_policies(["a", "b"], "mixed")

        self.assertTrue(any("different tasks" in w for w in result["warnings"]), result["warnings"])

    def test_unknown_fusion_method_raises(self):
        dir_a = _write_policy(training_mod.POLICIES_DIR, "a", _tiny_actor_critic(1))
        dir_b = _write_policy(training_mod.POLICIES_DIR, "b", _tiny_actor_critic(2))
        mgr = self._mgr({"a": self._source(dir_a), "b": self._source(dir_b)})
        with self.assertRaises(ValueError):
            mgr.fuse_policies(["a", "b"], "out", method="nope")

    def test_git_rebasin_fuses_two_policies_into_a_loadable_checkpoint(self):
        dir_a = _write_policy(training_mod.POLICIES_DIR, "a", _tiny_actor_critic(1))
        dir_b = _write_policy(training_mod.POLICIES_DIR, "b", _tiny_actor_critic(2))
        mgr = self._mgr({"a": self._source(dir_a), "b": self._source(dir_b)})

        result = mgr.fuse_policies(["a", "b"], "rebased", method="git_rebasin")

        fused_dir = training_mod.POLICIES_DIR / "rebased"
        loaded = torch.jit.load(str(fused_dir / "checkpoint.pt"))
        out = loaded(torch.zeros(1, 6))
        self.assertEqual(tuple(out.shape), (1, 2))
        with open(fused_dir / "meta.json") as f:
            meta = json.load(f)
        self.assertEqual(meta["fusion"]["method"], "git_rebasin")
        self.assertEqual(result["warnings"], [])

    def test_git_rebasin_fuses_two_recurrent_policies_into_a_loadable_checkpoint(self):
        torch.manual_seed(1)
        rec_a = ActorCriticRecurrent(num_actor_obs=6, num_critic_obs=6, num_actions=2,
                                      actor_hidden_dims=[8], critic_hidden_dims=[8],
                                      rnn_type="lstm", rnn_hidden_size=8, rnn_num_layers=1)
        torch.manual_seed(2)
        rec_b = ActorCriticRecurrent(num_actor_obs=6, num_critic_obs=6, num_actions=2,
                                      actor_hidden_dims=[8], critic_hidden_dims=[8],
                                      rnn_type="lstm", rnn_hidden_size=8, rnn_num_layers=1)
        dir_a = _write_policy(training_mod.POLICIES_DIR, "a", rec_a)
        dir_b = _write_policy(training_mod.POLICIES_DIR, "b", rec_b)
        mgr = self._mgr({"a": self._source(dir_a), "b": self._source(dir_b)})

        result = mgr.fuse_policies(["a", "b"], "rebased_rnn", method="git_rebasin")

        fused_dir = training_mod.POLICIES_DIR / "rebased_rnn"
        loaded = torch.jit.load(str(fused_dir / "checkpoint.pt"))
        # PolicyExporterLSTM's forward is (obs, hidden_state, cell_state) -> (action, h, c) —
        # the exported jit module bundles the LSTM in, unlike the plain (non-recurrent) exporter.
        out, _, _ = loaded(torch.zeros(1, 6), torch.zeros(1, 1, 8), torch.zeros(1, 1, 8))
        self.assertEqual(tuple(out.shape), (1, 2))
        with open(fused_dir / "meta.json") as f:
            meta = json.load(f)
        self.assertEqual(meta["fusion"]["method"], "git_rebasin")
        self.assertEqual(result["warnings"], [])

    def test_failed_fuse_does_not_leave_a_partial_policy_folder(self):
        dir_a = _write_policy(training_mod.POLICIES_DIR, "a", _tiny_actor_critic(1))
        dir_b = _write_policy(training_mod.POLICIES_DIR, "b", _tiny_actor_critic(2))
        mgr = self._mgr({"a": self._source(dir_a), "b": self._source(dir_b)})

        def _broken_export(*args, **kwargs):
            raise RuntimeError("simulated export failure")
        fusion.export_actor_critic = _broken_export

        with self.assertRaises(RuntimeError):
            mgr.fuse_policies(["a", "b"], "broken")
        self.assertFalse((training_mod.POLICIES_DIR / "broken").exists())


if __name__ == "__main__":
    unittest.main()
