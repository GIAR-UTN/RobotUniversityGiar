"""Pushes the repo-root sys.path entry to the end instead of the front.
Needed under .venv-mjlab so PyPI rsl-rl-lib (in site-packages) resolves
ahead of this repo's vendored top-level rsl_rl/ package, while this
repo's own mjlab_tasks/ package -- only available via the repo root --
still resolves. See docs/mjlab_migration.md R1. A no-op under the main
.venv, which has no competing rsl_rl in site-packages.

pytest inserts the repo root as an ABSOLUTE path (not "" or "."), so the
filter has to match on that, not on the bare-string forms `-I` cleanly
strips -- confirmed by hand (an "" / "." filter alone didn't catch it).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if p not in ("", ".", _REPO_ROOT)] + [_REPO_ROOT]
