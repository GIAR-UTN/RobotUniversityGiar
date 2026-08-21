"""S1 (HANDOFF_mimic_motion_library_ux.md): ControlService.list_motions()/
switch_motion(), and the --motion_file-dropping bug fix in
legged_gym/scripts/rugiar_driver_mjlab.py's family-switch relaunch.

Two halves:

1. list_motions()/switch_motion() themselves — pure ControlService logic
   (glob a directory, cross-reference TrainingManager.discover_local_
   policies(), validate/resolve a path). Exercised against a lightweight
   FakeAdapter (only `backend_name` matters to these two methods — see
   ControlService._require_motion_capable()) and real temp
   motion-clip/policies directories (monkeypatched module-level constants,
   same pattern tests/test_rename_policy.py and tests/test_fusion.py
   already use for training_mod.POLICIES_DIR) rather than mocks, so the
   real glob/discovery/path-validation code actually runs. No mjlab env
   build needed for this half.

2. _argv_for_family_switch()/_argv_for_motion_switch() — the actual argv
   rebuild rugiar_driver_mjlab.py's self-relaunch performs, split out from
   _relaunch_for_family()/_relaunch_for_motion() (which end in
   subprocess.Popen()+os._exit() and so can't be unit tested directly)
   specifically so this bug fix is verifiable: the old argv rebuild never
   included --motion_file at all, so relaunching into another mjlab-hosted
   family silently reverted to the hardcoded default clip. Needs mjlab
   importable only because importing the driver script itself does.

Run with .venv-mjlab (conftest.py handles sys.path, R1):

    CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest \
        tests/test_mjlab_motion_switch.py -q
"""
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

import legged_gym.control.service as service_mod  # noqa: E402
import legged_gym.control.training as training_mod  # noqa: E402
from legged_gym.control.service import ControlService  # noqa: E402
from legged_gym.control.training import TrainingManager  # noqa: E402


class FakeAdapter:
    """Only `backend_name` is read by list_motions()/switch_motion() —
    everything else about a real MjlabAdapter/SimAdapter is irrelevant to
    these two methods, so a real env build isn't needed to test them."""

    def __init__(self, backend_name):
        self.backend_name = backend_name


# ---- half 1: list_motions() / switch_motion() ----

@pytest.fixture
def motion_service(monkeypatch, tmp_path):
    # list_motions()/switch_motion() report clip paths relative to
    # REPO_ROOT (that's what --motion_file/DEFAULT_MOTION expect — see
    # rugiar_driver_mjlab.py) — patch REPO_ROOT itself to this fixture's
    # own tmp_path so that still holds for a motion dir that lives outside
    # the real repo tree, instead of leaking test fixtures into it.
    motion_dir = tmp_path / "_motion_clips"
    policies_dir = tmp_path / "_policies"
    motion_dir.mkdir()
    policies_dir.mkdir()

    (motion_dir / "dance1_subject2.npz").touch()
    (motion_dir / "unmatched_clip.npz").touch()

    policy_dir = policies_dir / "javier_mjlab_dance1_subject2"
    policy_dir.mkdir()
    (policy_dir / "checkpoint.onnx").touch()
    (policy_dir / "meta.json").write_text(json.dumps({"task": "Rugiar-G1-Mimic"}))

    # A policy for a DIFFERENT task must not count as a match for either
    # clip, even if its name happens to collide.
    other_task_dir = policies_dir / "genesis_dance1_subject2_lookalike"
    other_task_dir.mkdir()
    (other_task_dir / "checkpoint.pt").touch()
    (other_task_dir / "meta.json").write_text(json.dumps({"task": "g1_deepmimic"}))

    monkeypatch.setattr(service_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service_mod, "MOTION_DIR", motion_dir)
    monkeypatch.setattr(training_mod, "POLICIES_DIR", policies_dir)

    adapter = FakeAdapter("mjlab")
    service = ControlService(
        adapter, supervisor=object(), safety=object(),
        training=TrainingManager(), task_name="Rugiar-G1-Mimic",
        motion_file="resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz",
    )
    return service, motion_dir


def test_list_motions_flags_clip_with_and_without_a_local_policy(motion_service):
    service, _ = motion_service
    result = service.list_motions()
    by_name = {c["name"]: c for c in result["clips"]}
    assert set(by_name) == {"dance1_subject2", "unmatched_clip"}
    assert by_name["dance1_subject2"]["has_policy"] is True
    assert by_name["unmatched_clip"]["has_policy"] is False
    # Not disabled either way -- list_motions() itself carries no such
    # flag; has_policy is purely informational (the web UI must not use it
    # to disable a row, unlike familyButtonRow's hasPolicies gate).
    assert result["current"] == service.motion_file


def test_list_motions_prefers_explicit_motion_file_over_heuristic(motion_service, monkeypatch, tmp_path):
    """A policy whose meta.json carries the new explicit `motion_file` field
    (written by TrainingManager.finalize_policy() since HANDOFF Item 2's
    follow-up) must cross-reference by that EXACT path, not by name-
    substring guessing -- proven here by a policy folder name that would
    NOT match its clip under the old heuristic at all."""
    service, motion_dir = motion_service
    policies_dir = training_mod.POLICIES_DIR
    explicit_dir = policies_dir / "totally_unrelated_policy_name"
    explicit_dir.mkdir()
    (explicit_dir / "checkpoint.onnx").touch()
    # Relative to the fixture's (patched) REPO_ROOT -- same convention
    # list_motions() itself uses for each clip's own "path" field.
    (explicit_dir / "meta.json").write_text(json.dumps({
        "task": "Rugiar-G1-Mimic",
        "motion_file": str((motion_dir / "unmatched_clip.npz").relative_to(service_mod.REPO_ROOT)),
    }))

    result = service.list_motions()
    by_name = {c["name"]: c for c in result["clips"]}
    # unmatched_clip's name shares nothing with "totally_unrelated_policy_name"
    # -- only the explicit meta.json field can explain this being True.
    assert by_name["unmatched_clip"]["has_policy"] is True


def test_list_motions_falls_back_to_heuristic_for_legacy_policies_with_no_field(motion_service):
    """javier_mjlab_dance1_subject2 (the fixture's baseline policy) has no
    `motion_file` key in its meta.json at all -- exactly what a policy
    trained before this change looks like. It must still match via the old
    name-substring heuristic, unchanged."""
    service, _ = motion_service
    result = service.list_motions()
    by_name = {c["name"]: c for c in result["clips"]}
    assert by_name["dance1_subject2"]["has_policy"] is True


def test_list_motions_ignores_a_same_named_policy_for_a_different_task(motion_service):
    """genesis_dance1_subject2_lookalike is registered for 'g1_deepmimic',
    not 'Rugiar-G1-Mimic' -- it must not itself flip has_policy (the real
    match here comes from javier_mjlab_dance1_subject2, which IS for the
    right task)."""
    service, _ = motion_service
    result = service.list_motions()
    by_name = {c["name"]: c for c in result["clips"]}
    assert by_name["unmatched_clip"]["has_policy"] is False


def test_switch_motion_records_relative_path_intent(motion_service):
    service, motion_dir = motion_service
    clip_path = motion_dir / "unmatched_clip.npz"
    assert service.motion_switch_requested is None
    service.switch_motion(str(clip_path))
    assert service.motion_switch_requested is not None
    assert service.motion_switch_requested.endswith("unmatched_clip.npz")


def test_switch_motion_rejects_path_outside_motion_dir(motion_service):
    service, _ = motion_service
    with pytest.raises(ValueError):
        service.switch_motion("/etc/passwd")
    assert service.motion_switch_requested is None


def test_switch_motion_rejects_nonexistent_clip(motion_service):
    service, motion_dir = motion_service
    with pytest.raises(ValueError):
        service.switch_motion(str(motion_dir / "does_not_exist.npz"))


def test_genesis_backend_raises_not_implemented_cleanly():
    """S1's guard: Genesis has no reference-motion-clip concept at all --
    both methods must raise NotImplementedError (which ControlServer's
    _dispatch() turns into a clean JSON error reply, not an unhandled
    exception surfacing as a 500)."""
    adapter = FakeAdapter("sim")
    service = ControlService(adapter, supervisor=object(), safety=object(),
                             training=TrainingManager())
    with pytest.raises(NotImplementedError):
        service.list_motions()
    with pytest.raises(NotImplementedError):
        service.switch_motion("whatever.npz")


# ---- half 2: the relaunch argv rebuild (the actual bug fix) ----

def _load_driver_module():
    pytest.importorskip("mjlab")
    spec = importlib.util.spec_from_file_location(
        "rugiar_driver_mjlab", REPO / "legged_gym/scripts/rugiar_driver_mjlab.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


@pytest.fixture(scope="module")
def driver():
    return _load_driver_module()


def _cli(**overrides):
    ns = argparse.Namespace(
        task="Rugiar-G1-Mimic", viser_port=9006, ramp_ticks=15,
        control_port=9017, token=None, motion_file=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_family_switch_preserves_motion_file_into_an_mjlab_task(driver):
    """Regression test for the bug S1 fixed: relaunching into another
    mjlab-hosted family used to silently drop --motion_file, reverting to
    the hardcoded DEFAULT_MOTION clip regardless of what was actually
    playing before the switch."""
    active_clip = "resources/reference_motion/unitree_g1/mjlab_run/g1moves_B_DadDance.npz"
    cli = _cli(motion_file=active_clip)
    argv, _env = driver._argv_for_family_switch(cli, "Rugiar-G1-Mimic")
    assert "--motion_file" in argv
    assert argv[argv.index("--motion_file") + 1] == active_clip


def test_family_switch_with_no_motion_file_omits_the_flag(driver):
    cli = _cli(motion_file=None)
    argv, _env = driver._argv_for_family_switch(cli, "Rugiar-G1-Mimic")
    assert "--motion_file" not in argv


def test_family_switch_into_a_genesis_task_does_not_pass_motion_file(driver):
    """rugiar_driver.py (the Genesis sibling) has no --motion_file flag at
    all -- passing it would be a hard argparse error on the far end."""
    cli = _cli(motion_file="resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz")
    result = driver._argv_for_family_switch(cli, "g1")
    if result is None:
        pytest.skip("no Genesis .venv on this machine to relaunch into")
    argv, _env = result
    assert "--motion_file" not in argv


def test_motion_switch_argv_targets_same_task_new_clip_same_interpreter(driver):
    cli = _cli(motion_file="resources/reference_motion/unitree_g1/mjlab_run/dance1_subject2.npz")
    new_clip = "resources/reference_motion/unitree_g1/mjlab_run/g1moves_B_DadDance.npz"
    argv = driver._argv_for_motion_switch(cli, new_clip)
    assert argv[0] == sys.executable
    assert argv[argv.index("--task") + 1] == "Rugiar-G1-Mimic"
    assert argv[argv.index("--motion_file") + 1] == new_clip
