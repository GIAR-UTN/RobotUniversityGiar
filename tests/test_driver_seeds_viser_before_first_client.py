#!/usr/bin/env python3
"""
Regression test for "camera and robot aren't centered on first load /
everything renders far away" — reported for obstacle_course, where the
track doesn't start at world origin, but reproducible on any scenario that
spawns off-origin.

Root cause (confirmed empirically, not just by code reading — see the
session that added this test): `env.simulator.base_pos` only becomes
world-frame (`env_origins` included) once
`GenesisSimulator.reset_root_states()` has actually run for that env --
that's the only place `base_pos += env_origins` happens. But
`task_registry.make_env()` never calls `env.reset()` itself, so right
after env construction `base_pos` still sits at its raw, un-offset spawn
position (e.g. near local (0, 0)) even though the scenario's props/terrain
are already placed in the true world frame (e.g. obstacle_course's single-
env `env_origins` is ~(-50, -50, 0) on a 'plane' terrain — see
GenesisSimulator._get_env_origins). A client connecting before the first
proper reset sees `ViserViewer._on_client_connect` place the camera near
whatever stale/pre-reset `_last_base_pos` it has -- near local (0, 0) --
while the robot mesh and every prop render 50+ units away. Same family of
bug as the already-fixed post-restart case (see
test_driver_restart_syncs_viser_before_resync.py), but at startup instead
of on a restart_requested tick, so that fix didn't cover it.

An earlier attempt at this fix manually added `env.simulator.env_origins`
to `base_pos` inside `ViserViewer.update_from_simulator()` -- WRONG: once a
real reset has run, `base_pos` already includes `env_origins`, so that
manual addition double-counted it (empirically caught: a freshly-connected
client's camera anchor came out at ~(-100, -100) instead of ~(-50, -50)).
Do not reintroduce that -- see update_from_simulator()'s own comment.

Fix: call `adapter.reset()` BEFORE the first `viser_viewer.
update_from_simulator(env, 0)` seed call, immediately after
`create_viser_viewer(...)` and before the main loop / before the web
server starts accepting connections -- so both `base_pos` (now genuinely
world-frame) and `_last_base_pos` already reflect the env's real,
post-reset pose before any client can possibly connect.

This test parses (does not import) both Genesis driver scripts with `ast`
— same technique as test_driver_family_parity.py — to avoid a Genesis/
torch runtime dependency, and asserts that inside main()'s top-level
statements (i.e. not nested inside the main loop's `while True:`),
`adapter.reset()` is called, followed by `viser_viewer.
update_from_simulator(...)`, in that order, before the loop.

Run directly: python tests/test_driver_seeds_viser_before_first_client.py
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


def _find_main(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("no def main() found")


def _is_call(node, dotted_name: str) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    owner = node.func.value
    return (
        isinstance(owner, ast.Name)
        and owner.id == dotted_name.split(".")[0]
        and node.func.attr == dotted_name.split(".")[1]
    )


def _pre_loop_call_order(path: Path) -> list:
    """Dotted call names, in source order, seen anywhere in main()'s
    top-level statements BEFORE the top-level `while True:` loop."""
    tree = ast.parse(path.read_text(), filename=str(path))
    main_fn = _find_main(tree)

    calls = []
    for stmt in main_fn.body:
        if isinstance(stmt, ast.While):
            break
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Name):
                    calls.append(f"{owner.id}.{node.func.attr}")
    return calls


class DriverSeedsViserBeforeFirstClientTest(unittest.TestCase):
    def test_reset_then_viser_seeded_before_main_loop(self):
        for path in DRIVERS:
            with self.subTest(driver=path.name):
                calls = _pre_loop_call_order(path)
                self.assertIn(
                    "adapter.reset",
                    calls,
                    f"{path.name} never calls adapter.reset() before its main `while "
                    "True:` loop -- env.simulator.base_pos stays at its raw, un-offset "
                    "construction-time value (env_origins not yet applied) until a real "
                    "reset runs, so a client connecting before that sees the camera "
                    "placed far from the robot/scenario.",
                )
                self.assertIn(
                    "viser_viewer.update_from_simulator",
                    calls,
                    f"{path.name} never calls viser_viewer.update_from_simulator(...) "
                    "before its main `while True:` loop -- a client connecting before "
                    "the first policy tick sees a stale/empty scene.",
                )
                reset_idx = calls.index("adapter.reset")
                seed_idx = calls.index("viser_viewer.update_from_simulator")
                self.assertLess(
                    reset_idx,
                    seed_idx,
                    f"{path.name} calls viser_viewer.update_from_simulator(...) before "
                    "adapter.reset() -- base_pos would still be pre-reset (env_origins "
                    "not yet applied), seeding viser with the wrong position.",
                )


if __name__ == "__main__":
    unittest.main()
