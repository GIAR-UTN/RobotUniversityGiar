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
import threading
import time
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
    """Mirrors viser's own CameraHandle.position setter side effect: setting
    .position ALSO shifts .look_at by the same delta (viser's own docstring:
    "position updates translate both the camera and its look_at point
    together"). This is not a nicety -- a fake that just stored position/
    look_at as bare attributes would silently pass even if
    _apply_camera_tracking() additionally added delta to .look_at itself,
    which is exactly the double-application bug this fake exists to catch."""

    def __init__(self, position, look_at, up_direction=(0.0, 1.0, 0.0)):
        self._position = np.array(position, dtype=float)
        self._look_at = np.array(look_at, dtype=float)
        # Real viser's default (before any explicit set) is the BROWSER's
        # own reported up, which starts out three.js's Y-up -- not this
        # scene's actual +Z-up convention. Defaulting the fake the same way
        # lets a test catch code that forgets to set it explicitly.
        self._up_direction = np.array(up_direction, dtype=float)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        value = np.array(value, dtype=float)
        offset = value - self._position
        self._position = value
        self._look_at = self._look_at + offset

    @property
    def look_at(self):
        return self._look_at

    @look_at.setter
    def look_at(self, value):
        self._look_at = np.array(value, dtype=float)

    @property
    def up_direction(self):
        return self._up_direction

    @up_direction.setter
    def up_direction(self, value):
        self._up_direction = np.array(value, dtype=float)


class _FakeClient:
    def __init__(self, position, look_at, up_direction=(0.0, 1.0, 0.0)):
        self.camera = _FakeCamera(position, look_at, up_direction)


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
    viewer._last_base_pos = np.zeros(3)
    viewer._camera_track_lock = threading.Lock()
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

    def test_reconnecting_client_snaps_to_current_robot_position(self):
        """Regression for the "reload the page and the camera/robot aren't
        centered, have to toggle Track robot" bug: on_client_connect used to
        place a fresh client's camera at a fixed offset from the ORIGIN,
        not from wherever the robot actually was after walking away from it
        — so a client reconnecting mid-session (e.g. a browser reload) saw
        an empty patch of ground instead of the robot until the next
        tracking tick (or never, if tracking was off) caught up.
        _on_client_connect must use _last_base_pos, kept current every tick
        by update()/update_from_simulator() regardless of whether tracking
        is enabled."""
        viewer = _make_viewer()
        viewer._last_base_pos = np.array([12.0, -4.0, 0.8])  # robot walked far from origin

        client = _FakeClient(position=[0.0, 0.0, 0.0], look_at=[0.0, 0.0, 0.0])
        viewer._on_client_connect(client)

        np.testing.assert_allclose(client.camera.position, viewer._last_base_pos + viewer._camera_offset)
        np.testing.assert_allclose(client.camera.look_at, viewer._last_base_pos + viewer._camera_look_at_offset)
        self.assertIsNone(viewer._camera_track_last_base_pos)

    def test_reconnecting_client_gets_scene_up_direction_not_browser_default(self):
        """Regression for "no shaking anymore, but the camera lands badly
        aimed after a reload -- pointed near the horizon instead of down at
        the robot -- and still needs a Track-robot toggle to fix." Position/
        look_at were already correct; the missing piece was up_direction.
        A fresh client's CameraHandle starts out from the BROWSER's own
        first-reported camera state (three.js's Y-up default, still
        mid-stabilization right after connect) rather than this scene's
        actual +Z-up convention, and viser's internal _update_wxyz() derives
        the final camera orientation from position + look_at + up_direction
        together -- a stale/wrong up_direction alone is enough to tip the
        camera near the horizon even with the right position and target.
        _on_client_connect must pin up_direction to +Z explicitly instead of
        trusting whatever the client happened to report first."""
        viewer = _make_viewer()
        viewer._last_base_pos = np.array([12.0, -4.0, 0.8])

        client = _FakeClient(
            position=[0.0, 0.0, 0.0], look_at=[0.0, 0.0, 0.0], up_direction=[0.0, 1.0, 0.0]
        )
        viewer._on_client_connect(client)

        np.testing.assert_allclose(client.camera.up_direction, [0.0, 0.0, 1.0])

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

    def test_connect_and_tracking_tick_race_dont_interleave(self):
        """Regression for the "reload the page and the camera keeps
        bouncing" bug: viser fires _on_client_connect on its own websocket
        thread, concurrently with _apply_camera_tracking running on the main
        sim-loop thread. Both write a client's camera.position THEN
        camera.look_at as two separate statements; without a lock the two
        threads' statements can interleave (thread A's position write,
        thread B's position write, thread B's look_at write, thread A's
        look_at write), leaving position and look_at sourced from different
        base_pos snapshots -- a camera pointed at the wrong spot, which
        reads as the reported bouncing/glitching right after a reload.
        _camera_track_lock must make each writer's position+look_at pair
        atomic with respect to the other.

        Uses a Barrier (not a sleep) inside the camera's position setter to
        force a deterministic rendezvous between the two threads: if
        _camera_track_lock actually serializes the two call sites, the
        second thread can never reach the barrier while the first holds the
        lock, so the barrier always times out (harmless, swallowed below)
        and no interleaving happens. If the lock were missing, both threads
        would reach the barrier together every iteration, guaranteeing the
        interleave this test exists to catch."""

        barrier = threading.Barrier(2)

        class _RendezvousCamera(_FakeCamera):
            @_FakeCamera.position.setter
            def position(self, value):
                _FakeCamera.position.fset(self, value)
                try:
                    barrier.wait(timeout=0.02)
                except threading.BrokenBarrierError:
                    pass
                finally:
                    barrier.reset()

        viewer = _make_viewer()
        client = _FakeClient(position=[0.0, 0.0, 0.0], look_at=[0.0, 0.0, 0.0])
        client.camera = _RendezvousCamera(position=[0.0, 0.0, 0.0], look_at=[0.0, 0.0, 0.0])
        viewer.server._clients[0] = client

        errors = []
        ITERS = 15
        expected_diff = viewer._camera_offset - viewer._camera_look_at_offset

        def connect_loop():
            for i in range(ITERS):
                try:
                    viewer._last_base_pos = np.array([float(i), 0.0, 0.0])
                    viewer._on_client_connect(client)
                    np.testing.assert_allclose(
                        client.camera.position - client.camera.look_at, expected_diff
                    )
                except Exception as exc:  # pragma: no cover - surfaced via errors
                    errors.append(exc)

        def tracking_loop():
            for i in range(ITERS):
                try:
                    with viewer._camera_track_lock:
                        viewer._camera_track_last_base_pos = None
                    viewer._apply_camera_tracking(np.array([0.0, float(i), 0.0]))
                    np.testing.assert_allclose(
                        client.camera.position - client.camera.look_at, expected_diff
                    )
                except Exception as exc:  # pragma: no cover - surfaced via errors
                    errors.append(exc)

        t1 = threading.Thread(target=connect_loop)
        t2 = threading.Thread(target=tracking_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()
