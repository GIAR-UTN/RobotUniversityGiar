#!/usr/bin/env python3
"""
Guards against legged_gym/scripts/rugiar_driver.py and its sibling
rugiar_driver_gaze.py drifting apart on the logic that is NOT specific to
the "target-aware" family (see both files' module docstrings — they are a
deliberately duplicated pair, one script per task family, rather than a
shared module, because each family is treated as its own architecturally
independent EXPERIMENT).

That duplication is a real maintenance hazard: a bug fix or behavior change
made to a shared helper (e.g. _sibling_meta_simulator, _relaunch_for_family)
in one file is easy to forget to mirror into the other, and nothing else
would catch it.

This test parses (does not import) both files with `ast` and asserts that
every function on the whitelist below — the ones with no gaze-specific
reason to differ — has textually identical bodies in both files once each
function's own docstring is stripped out (docstrings are allowed to differ,
e.g. to name "this file" vs "the sibling file"; see _script_for_task).
Parsing instead of importing means this test has no Genesis/torch runtime
dependency, unlike most of this project's other rugiar_driver tests.

Deliberately NOT covered here: main()'s body as a whole (it legitimately
differs — the gaze driver's target-aware obs injection is interleaved into
otherwise-shared control flow) and the shared portions of the module
docstrings themselves. Both files carry a cross-reference comment instead
(see their module docstrings / top-of-file comments) telling a future editor
to mirror non-gaze changes made inside main() by hand.

Run directly: python tests/test_driver_family_parity.py
"""
import ast
import sys
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DRIVER = REPO_ROOT / "legged_gym" / "scripts" / "rugiar_driver.py"
DRIVER_GAZE = REPO_ROOT / "legged_gym" / "scripts" / "rugiar_driver_gaze.py"

# Top-level (or nested, e.g. drain_finished_training inside main()) function
# names that have no target-aware reason to differ between the two drivers.
# main() itself is excluded on purpose -- see module docstring above.
SHARED_FUNCTIONS = [
    "_encode_camera_frame_jpeg",
    "parse_policy_args",
    "_sibling_meta_simulator",
    "_script_for_task",
    "_bare_g1_policy_specs",
    "_relaunch_for_family",
    "_sibling_meta_task",
    "drain_finished_training",
]


def _function_bodies(path: Path) -> dict:
    """name -> dedented source of the function with its own leading
    docstring (if any) stripped, for every def in `path` whose name is in
    SHARED_FUNCTIONS. Docstrings are stripped because they're allowed to
    name the file itself (e.g. "this file" vs "the sibling file")."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    bodies = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in SHARED_FUNCTIONS:
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]  # drop the docstring statement
        if not body:
            bodies[node.name] = ""
            continue
        start = body[0].lineno
        end = body[-1].end_lineno
        segment = "\n".join(source.splitlines()[start - 1:end])
        bodies[node.name] = textwrap.dedent(segment).strip()
    return bodies


class DriverFamilyParityTest(unittest.TestCase):
    def test_shared_functions_are_textually_identical(self):
        base_bodies = _function_bodies(DRIVER)
        gaze_bodies = _function_bodies(DRIVER_GAZE)

        missing_in_base = set(SHARED_FUNCTIONS) - set(base_bodies)
        missing_in_gaze = set(SHARED_FUNCTIONS) - set(gaze_bodies)
        self.assertFalse(
            missing_in_base, f"expected functions missing from rugiar_driver.py: {missing_in_base}"
        )
        self.assertFalse(
            missing_in_gaze,
            f"expected functions missing from rugiar_driver_gaze.py: {missing_in_gaze}",
        )

        for name in SHARED_FUNCTIONS:
            self.assertEqual(
                base_bodies[name],
                gaze_bodies[name],
                f"{name}() has drifted between rugiar_driver.py and rugiar_driver_gaze.py -- "
                "this function is meant to be non-gaze-specific and kept identical in both "
                "drivers (see this test's module docstring). If the change is intentional and "
                "target-aware-specific, either rename/move it out of SHARED_FUNCTIONS in "
                f"{Path(__file__).name} or mirror the change into the other file.",
            )


if __name__ == "__main__":
    unittest.main()
