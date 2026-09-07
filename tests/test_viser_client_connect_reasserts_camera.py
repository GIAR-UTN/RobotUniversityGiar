#!/usr/bin/env python3
"""
Regression test for "parpadeo y luego apareció lejos" (flicker, then it
settled far away) -- a client's camera snapping to the robot at connect
time, then losing that position to a stale/in-flight tracking tick shortly
after.

Root cause: `ViserViewer._on_client_connect` and the main sim loop's own
`_apply_camera_tracking` both write the same client's `camera.position`/
`look_at`, each under `_camera_track_lock` -- but the lock only prevents
them from writing at the same instant, not in a particular order. A single
write from `_on_client_connect` can still be overwritten a moment later by
a tracking tick that was already in flight when the client connected.

An earlier fix attempt re-asserted the camera position exactly ONCE, after
a single fixed 0.5s delay. That is not reliable: reproduced on a session
where this machine's load average was observed as high as ~300, where
0.5s of wall-clock time was not consistently enough for the client/
scheduler to settle, and the bug came back.

Fix: `_on_client_connect` now spawns a background thread that re-asserts
the camera position/look_at several times over a few seconds after
connect (see `_reassert_while_settling` in viser_viewer.py), so our value
is the one that lands last regardless of how long the client, network, or
OS scheduler takes to settle -- not just whichever side wins a single
fixed-delay race.

This test parses (does not import) viser_viewer.py with `ast` -- same
technique as test_driver_family_parity.py -- to avoid a Genesis/torch
runtime dependency, and asserts that `_on_client_connect`:
  1. spawns a background thread (`threading.Thread(...).start()`), and
  2. that thread's target function re-asserts the camera position inside
     a loop iterating over more than one delay value (not a single
     one-shot `time.sleep(...)` reassert).

Run directly: python tests/test_viser_client_connect_reasserts_camera.py
"""
import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VISER_VIEWER = REPO_ROOT / "legged_gym" / "utils" / "viser_viewer.py"


def _find_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"no {class_name}.{method_name}() found")


def _spawns_background_thread(method: ast.FunctionDef) -> bool:
    """True iff the method contains `threading.Thread(...).start()`."""
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "Thread"
        ):
            return True
    return False


def _nested_functions(method: ast.FunctionDef) -> list:
    return [n for n in ast.walk(method) if isinstance(n, ast.FunctionDef) and n is not method]


def _reasserts_over_multiple_delays(fn: ast.FunctionDef) -> bool:
    """True iff `fn` loops over more than one delay value, sleeping and
    re-writing `client.camera.position` on each iteration -- not a single
    one-shot `time.sleep(...)` followed by one write."""
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
            if len(node.iter.elts) < 2:
                continue
            has_sleep = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sleep"
                for n in ast.walk(node)
            )
            has_camera_write = any(
                isinstance(n, ast.Attribute) and n.attr == "position"
                for n in ast.walk(node)
            )
            if has_sleep and has_camera_write:
                return True
    return False


class ViserClientConnectReassertsCameraTest(unittest.TestCase):
    def test_on_client_connect_reasserts_camera_over_multiple_delays(self):
        tree = ast.parse(VISER_VIEWER.read_text(), filename=str(VISER_VIEWER))
        on_connect = _find_method(tree, "ViserViewer", "_on_client_connect")

        self.assertTrue(
            _spawns_background_thread(on_connect),
            "_on_client_connect no longer spawns a background thread to re-assert the "
            "camera position after the initial (racy) override -- see the "
            "'parpadeo y luego aparecio lejos' bug this guards against.",
        )

        nested = _nested_functions(on_connect)
        self.assertTrue(
            any(_reasserts_over_multiple_delays(fn) for fn in nested),
            "_on_client_connect's background reassert no longer loops over multiple "
            "delay values -- a single fixed-delay reassert was proven unreliable under "
            "load (observed load average ~300) and let the far-camera bug come back.",
        )


if __name__ == "__main__":
    unittest.main()
