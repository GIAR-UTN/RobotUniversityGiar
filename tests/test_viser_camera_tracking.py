#!/usr/bin/env python3
"""
Unit test for ViserViewer._apply_camera_tracking() (legged_gym/utils/viser_viewer.py)
— pure vector math, no viser server/websocket or physics backend needed.

Regression test for the "Track robot" bug reported against the control web:
with tracking on, every simulator tick used to hard-reset each connected
client's camera to `base_pos + fixed_offset`, discarding any zoom/orbit the
user had just done — scroll-wheel zoom and touch-pinch looked broken because
the very next tick snapped the camera back. _apply_camera_tracking() now
translates each client's *current* camera by how far the robot moved since
the last tick instead, so a user's chosen distance/angle survives.

legged_gym/__init__.py (and legged_gym.utils/.envs/.simulator's own
__init__.py side effects) eagerly import a real physics backend
(Genesis/IsaacGym/IsaacLab) — none of which this test needs, since
_apply_camera_tracking() only touches numpy arrays and a `server.get_clients()`
duck-typed stand-in. So, as in test_push_direction.py, we install empty
namespace-package stand-ins for the intermediate packages and let Python's
normal file-based import resolve viser_viewer.py through them.

Run directly: python tests/test_viser_camera_tracking.py
"""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_package(dotted_name: str):
    if dotted_name in sys.modules:
        return
    module = types.ModuleType(dotted_name)
    module.__path__ = [str(ROOT / Path(*dotted_name.split(".")))]
    module.__package__ = dotted_name
    sys.modules[dotted_name] = module


for _pkg in ("legged_gym", "legged_gym.utils"):
    _stub_package(_pkg)
# The real legged_gym/__init__.py normally defines this; viser_viewer.py
# imports it directly (`from legged_gym import LEGGED_GYM_ROOT_DIR`).
sys.modules["legged_gym"].LEGGED_GYM_ROOT_DIR = str(ROOT)

from legged_gym.utils.viser_viewer import ViserViewer


class _FakeCamera:
    def __init__(self, position, look_at):
        self.position = np.array(position, dtype=float)
        self.look_at = np.array(look_at, dtype=float)


class _FakeClient:
    def __init__(self, position, look_at):
        self.camera = _FakeCamera(position, look_at)


class _FakeServer:
    def __init__(self):
        self._clients = {}

    def get_clients(self):
        return self._clients


def _make_viewer():
    """A bare ViserViewer with only the attributes _apply_camera_tracking()
    touches — bypasses __init__ (mesh loading, real viser server) entirely."""
    viewer = object.__new__(ViserViewer)
    viewer.server = _FakeServer()
    viewer._camera_offset = np.array([2.0, 2.0, 1.5])
    viewer._camera_look_at_offset = np.array([0.0, 0.0, 0.3])
    viewer._camera_track_last_base_pos = None
    return viewer


class TestCameraTracking(unittest.TestCase):
    def test_first_call_snaps_to_default_offset(self):
        viewer = _make_viewer()
        client = _FakeClient(position=[9.0, 9.0, 9.0], look_at=[9.0, 9.0, 9.0])
        viewer.server._clients[0] = client

        base_pos = np.array([0.0, 0.0, 0.0])
        viewer._apply_camera_tracking(base_pos)

        np.testing.assert_allclose(client.camera.position, base_pos + viewer._camera_offset)
        np.testing.assert_allclose(client.camera.look_at, base_pos + viewer._camera_look_at_offset)

    def test_tracking_preserves_user_zoom_and_orbit(self):
        """The actual regression: a user-adjusted camera (zoomed in, orbited
        to a non-default angle) must be translated, not reset, as the robot
        keeps moving."""
        viewer = _make_viewer()
        client = _FakeClient(position=[0.0, 0.0, 0.0], look_at=[0.0, 0.0, 0.0])
        viewer.server._clients[0] = client

        base_pos_0 = np.array([0.0, 0.0, 0.0])
        viewer._apply_camera_tracking(base_pos_0)

        # Simulate the user zooming/orbiting client-side; viser syncs this
        # back to the server, so the next tick sees it via client.camera.
        zoomed_position = np.array([0.5, 0.5, 0.4])  # much closer than the default offset
        zoomed_look_at = np.array([0.0, 0.0, 0.3])
        client.camera.position = zoomed_position.copy()
        client.camera.look_at = zoomed_look_at.copy()

        base_pos_1 = np.array([1.0, 0.0, 0.0])  # robot walked 1m forward
        viewer._apply_camera_tracking(base_pos_1)

        delta = base_pos_1 - base_pos_0
        np.testing.assert_allclose(client.camera.position, zoomed_position + delta)
        np.testing.assert_allclose(client.camera.look_at, zoomed_look_at + delta)

        # The bug this replaces: naively resetting to base_pos + fixed offset
        # would have produced this instead — assert we did NOT reproduce it.
        naive_reset_position = base_pos_1 + viewer._camera_offset
        self.assertFalse(np.allclose(client.camera.position, naive_reset_position))

    def test_reenabling_tracking_snaps_back_to_default(self):
        """Turning 'Track robot' off then back on (cb_tracking.on_update)
        resets _camera_track_last_base_pos to None — the next tick should
        snap to the default offset again, not translate from a stale base
        position accumulated while tracking was off."""
        viewer = _make_viewer()
        client = _FakeClient(position=[0.0, 0.0, 0.0], look_at=[0.0, 0.0, 0.0])
        viewer.server._clients[0] = client

        viewer._apply_camera_tracking(np.array([5.0, 5.0, 0.0]))
        client.camera.position = np.array([99.0, 99.0, 99.0])  # user flew far away

        # ... tracking gets disabled, robot moves a lot, tracking re-enabled:
        viewer._camera_track_last_base_pos = None
        new_base_pos = np.array([50.0, 0.0, 0.0])
        viewer._apply_camera_tracking(new_base_pos)

        np.testing.assert_allclose(client.camera.position, new_base_pos + viewer._camera_offset)

    def test_no_clients_is_a_noop(self):
        viewer = _make_viewer()
        # Should not raise with zero connected clients.
        viewer._apply_camera_tracking(np.array([1.0, 2.0, 3.0]))
        self.assertIsNotNone(viewer._camera_track_last_base_pos)

    def test_multiple_clients_each_keep_their_own_offset(self):
        viewer = _make_viewer()
        near_client = _FakeClient(position=[0.5, 0.0, 0.5], look_at=[0.0, 0.0, 0.3])
        far_client = _FakeClient(position=[5.0, 5.0, 3.0], look_at=[0.0, 0.0, 0.3])
        viewer.server._clients[0] = near_client
        viewer.server._clients[1] = far_client
        # Seed last_base_pos directly (not via _apply_camera_tracking) so the
        # two clients' distinct starting positions survive — the "first
        # call" (None) branch would otherwise snap both to the same default.
        viewer._camera_track_last_base_pos = np.array([0.0, 0.0, 0.0])

        near_pos_before = near_client.camera.position.copy()
        far_pos_before = far_client.camera.position.copy()

        delta = np.array([2.0, 0.0, 0.0])
        viewer._apply_camera_tracking(delta)

        np.testing.assert_allclose(near_client.camera.position, near_pos_before + delta)
        np.testing.assert_allclose(far_client.camera.position, far_pos_before + delta)
        # The two clients' distinct zoom levels must not have converged.
        self.assertGreater(
            np.linalg.norm(far_client.camera.position - near_client.camera.position),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
