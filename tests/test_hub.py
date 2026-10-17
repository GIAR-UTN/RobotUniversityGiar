#!/usr/bin/env python3
"""
Unit tests for legged_gym/control/hub.py — the Hugging Face Hub mirror of
./policies/<name>/. huggingface_hub is replaced by an in-memory fake (one
dict per repo), so these need neither the package, a network nor an account.
hub.py is loaded straight from its file: it imports nothing from legged_gym,
and importing the legged_gym package itself would pull in a physics backend.

Run directly: python tests/test_hub.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rugiar_hub", ROOT / "legged_gym/control/hub.py")
hub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hub)


class FakeHub:
    """Stands in for the huggingface_hub module: repos are {path_in_repo: bytes}."""

    def __init__(self):
        self.repos = {}
        fake = self

        class HfApi:
            def create_repo(self, repo_id, repo_type, private, exist_ok):
                fake.repos.setdefault(repo_id, {})

            def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, commit_message):
                for f in Path(folder_path).rglob("*"):
                    if f.is_file():
                        fake.repos[repo_id][f"{path_in_repo}/{f.relative_to(folder_path)}"] = f.read_bytes()

            def list_repo_files(self, repo_id, repo_type):
                return list(fake.repos[repo_id])

        def snapshot_download(repo_id, repo_type, allow_patterns, local_dir):
            prefix = allow_patterns[0].rstrip("*")
            for path, data in fake.repos.get(repo_id, {}).items():
                if path.startswith(prefix):
                    out = Path(local_dir) / path
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)
            return local_dir

        self.module = types.SimpleNamespace(HfApi=HfApi, snapshot_download=snapshot_download)


def _make_policy(policies_dir: Path, name: str, checkpoint: bytes = b"weights"):
    d = policies_dir / name
    d.mkdir(parents=True)
    (d / "checkpoint.pt").write_bytes(checkpoint)
    (d / "meta.json").write_text(json.dumps({"task": "g1"}))


class TestHub(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.policies = self.tmp / "policies"
        self.policies.mkdir()
        self.fake = FakeHub()
        self.patch = mock.patch.object(hub, "_hf", lambda: self.fake.module)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.tmp)

    def test_push_uploads_the_policy_folder_under_its_own_name(self):
        _make_policy(self.policies, "walker")
        hub.push_policy("walker", "org/policies", policies_dir=self.policies)
        self.assertEqual(self.fake.repos["org/policies"]["walker/checkpoint.pt"], b"weights")
        self.assertIn("walker/meta.json", self.fake.repos["org/policies"])

    def test_push_refuses_an_incomplete_policy(self):
        (self.policies / "half").mkdir()
        with self.assertRaises(hub.HubError):
            hub.push_policy("half", "org/policies", policies_dir=self.policies)

    def test_list_returns_only_policy_folders(self):
        _make_policy(self.policies, "a")
        _make_policy(self.policies, "b")
        hub.push_policy("a", "org/p", policies_dir=self.policies)
        hub.push_policy("b", "org/p", policies_dir=self.policies)
        self.fake.repos["org/p"]["README.md"] = b"x"
        self.assertEqual(hub.list_remote_policies("org/p"), ["a", "b"])

    def test_pull_round_trips_and_protects_local_copies(self):
        _make_policy(self.policies, "walker", b"v1")
        hub.push_policy("walker", "org/p", policies_dir=self.policies)
        shutil.rmtree(self.policies / "walker")

        hub.pull_policy("walker", "org/p", policies_dir=self.policies)
        self.assertEqual((self.policies / "walker/checkpoint.pt").read_bytes(), b"v1")

        with self.assertRaises(hub.HubError):
            hub.pull_policy("walker", "org/p", policies_dir=self.policies)
        (self.policies / "walker/checkpoint.pt").write_bytes(b"local edit")
        hub.pull_policy("walker", "org/p", policies_dir=self.policies, overwrite=True)
        self.assertEqual((self.policies / "walker/checkpoint.pt").read_bytes(), b"v1")

    def test_pull_of_a_missing_policy_leaves_nothing_behind(self):
        self.fake.repos["org/p"] = {}
        with self.assertRaises(hub.HubError):
            hub.pull_policy("ghost", "org/p", policies_dir=self.policies)
        self.assertFalse((self.policies / "ghost").exists())

    def test_repo_comes_from_env_when_not_given(self):
        _make_policy(self.policies, "walker")
        with mock.patch.dict(os.environ, {hub.REPO_ENV: "org/env-repo"}):
            hub.push_policy("walker", policies_dir=self.policies)
        self.assertIn("walker/meta.json", self.fake.repos["org/env-repo"])
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(hub.HubError):
                hub.push_policy("walker", policies_dir=self.policies)


class TestAutoPush(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _make_policy(self.tmp, "walker")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_disabled_without_a_repo(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(hub, "push_policy") as push:
            self.assertIsNone(hub.push_after_save("walker", policies_dir=self.tmp))
        push.assert_not_called()

    def test_a_failed_upload_never_raises(self):
        def boom(*a, **k):
            raise ConnectionError("no network")
        with mock.patch.dict(os.environ, {hub.REPO_ENV: "org/p"}), \
                mock.patch.object(hub, "push_policy", boom):
            self.assertIsNone(hub.push_after_save("walker", policies_dir=self.tmp))

    def test_missing_package_is_reported_not_raised(self):
        with mock.patch.dict(os.environ, {hub.REPO_ENV: "org/p"}), \
                mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            self.assertIsNone(hub.push_after_save("walker", policies_dir=self.tmp))


if __name__ == "__main__":
    unittest.main()
