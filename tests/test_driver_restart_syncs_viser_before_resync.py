#!/usr/bin/env python3
"""
Regression test for the "positions look wrong / viser flickers and zoom
misbehaves right after a restart" bug reported against the obstacle_course
scenario.

Root cause: `service.restart_requested` handling in rugiar_driver.py /
rugiar_driver_target.py called `adapter.reset()` then
`viser_viewer.resync_camera_tracking()`, but never pushed the post-reset
robot/prop transforms into viser itself. `resync_camera_tracking()` only
invalidates the camera's stale anchor (`_camera_track_last_base_pos`) so the
*next* tracking tick snaps cleanly -- it does not touch mesh positions or
`_last_base_pos` (see ViserViewer.update_from_simulator(), the only place
that does). Those stay stale until `service.tick(obs)` next returns a
non-None action and the ordinary per-tick `viser_viewer.update_from_simulator`
call further down the loop runs -- which can be several ticks later (paused
policy, countdown, decimation). In that window the scene still shows the
pre-reset pose while the camera has already re-anchored, which is exactly
"positions don't look right" + a flicker/zoom jump.

Fix: call `viser_viewer.update_from_simulator(env, 0)` immediately after
`adapter.reset()`/`obs = adapter.get_observations()`, BEFORE
`resync_camera_tracking()`, so both the scene and `_last_base_pos` are
already current by the time tracking re-anchors.

This test parses (does not import) both Genesis driver scripts with `ast` --
same technique as test_driver_family_parity.py -- to avoid a Genesis/torch
runtime dependency, and asserts that inside the `if service.restart_requested:`
block, `viser_viewer.update_from_simulator(...)` is called strictly before
`viser_viewer.resync_camera_tracking()`.

Run directly: python tests/test_driver_restart_syncs_viser_before_resync.py
"""
import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DRIVERS = [
    REPO_ROOT / "legged_gym" / "scripts" / "rugiar_driver.py",
    REPO_ROOT / "legged_gym" / "scripts" / "rugiar_driver_target.py",
]


def _restart_block_call_order(path: Path):
    """Returns, in source order, the dotted call names (e.g.
    'viser_viewer.resync_camera_tracking') made anywhere inside the
    `if service.restart_requested:` block of `path`."""
    tree = ast.parse(path.read_text(), filename=str(path))

    def is_restart_test(test_node):
        return (
            isinstance(test_node, ast.Attribute)
            and test_node.attr == "restart_requested"
            and isinstance(test_node.value, ast.Name)
            and test_node.value.id == "service"
        )

    restart_ifs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and is_restart_test(node.test)
    ]
    assert len(restart_ifs) == 1, (
        f"expected exactly one 'if service.restart_requested:' block in {path.name}, "
        f"found {len(restart_ifs)}"
    )

    calls = []
    for node in ast.walk(restart_ifs[0]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name):
                calls.append(f"{owner.id}.{node.func.attr}")
    return calls


class DriverRestartSyncsViserBeforeResyncTest(unittest.TestCase):
    def test_update_from_simulator_precedes_resync_on_restart(self):
        for path in DRIVERS:
            with self.subTest(driver=path.name):
                calls = _restart_block_call_order(path)
                self.assertIn(
                    "viser_viewer.update_from_simulator",
                    calls,
                    f"{path.name}'s restart handling never refreshes viser's scene "
                    "(viser_viewer.update_from_simulator) after adapter.reset() -- "
                    "the viewer will keep showing pre-reset positions until the next "
                    "policy tick.",
                )
                self.assertIn(
                    "viser_viewer.resync_camera_tracking",
                    calls,
                    f"{path.name}'s restart handling no longer calls "
                    "viser_viewer.resync_camera_tracking() -- see ae3649c.",
                )
                update_idx = calls.index("viser_viewer.update_from_simulator")
                resync_idx = calls.index("viser_viewer.resync_camera_tracking")
                self.assertLess(
                    update_idx,
                    resync_idx,
                    f"{path.name} calls resync_camera_tracking() before "
                    "update_from_simulator() on restart -- the camera would re-anchor "
                    "to a still-stale _last_base_pos, reproducing the post-restart "
                    "flicker/wrong-position bug.",
                )


if __name__ == "__main__":
    unittest.main()
