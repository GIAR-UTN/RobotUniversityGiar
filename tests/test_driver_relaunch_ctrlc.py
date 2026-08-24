#!/usr/bin/env python3
"""
Regression test for the "port 9017 stays in use after Ctrl+C" bug.

Root cause: _spawn_or_exec() in the three driver scripts
(legged_gym/scripts/rugiar_driver.py, rugiar_driver_target.py,
rugiar_driver_mjlab.py) relaunched a family/motion switch by spawning the
new driver as a DETACHED child (start_new_session=True) and os._exit()ing
the old process -- except when PID 1 (Docker). A detached child gets its
OWN new session, which the terminal's SIGINT-to-foreground-process-group
never reaches, so after any family/motion switch the driver survived
Ctrl+C and kept running, holding --control_port/--viser_port. The next
launch then died with "ControlServer failed to bind ws://0.0.0.0:<port>
(port likely already in use)".

The fix: _spawn_or_exec() always replaces THIS process IN PLACE with
os.execve() -- same PID/session/process group, so Ctrl+C keeps working.
All listening sockets are non-inheritable (PEP 446: socket.socket()
defaults to non-inheritable), so execve closes them and the new driver
rebinds the same ports without EADDRINUSE. spawn-detached + os._exit() is
kept ONLY as a fallback for when execve itself fails (missing interpreter,
etc.).

This test loads _spawn_or_exec's source from each driver via ast (like
test_driver_family_parity.py) and runs it in a sandbox with fake
os/sys/subprocess, so no Genesis/mjlab import is needed:

  - the primary path must reach os.execve(argv[0], argv, env) and must
    NOT fall through to subprocess.Popen (the old detached-spawn bug);
  - the primary path must not even consult os.getpid() (the sandbox's
    FakeOS has no getpid, so an AttributeError here means the execve is
    still gated on PID 1 -- i.e. the terminal-detach bug is back);
  - when os.execve raises OSError, the fallback must Popen with
    start_new_session=True + env and os._exit(0).

Run directly: python tests/test_driver_relaunch_ctrlc.py
"""
import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DRIVERS = [
    REPO_ROOT / "legged_gym" / "scripts" / "rugiar_driver.py",
    REPO_ROOT / "legged_gym" / "scripts" / "rugiar_driver_target.py",
    REPO_ROOT / "legged_gym" / "scripts" / "rugiar_driver_mjlab.py",
]


def _spawn_or_exec_source(path: Path) -> str:
    """Raw source of the _spawn_or_exec def in `path`, or raise if absent --
    parsing (not importing) keeps this test free of any simulator runtime
    dependency."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_spawn_or_exec":
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"_spawn_or_exec not found in {path}")


class _ExecveReturned(Exception):
    """Sentinel the FakeOS raises from execve on success -- execve never
    returns, and the test must not silently continue past it."""


class _FakeOS:
    def __init__(self, execve_error=None):
        self.execve_calls = []
        self.exit_calls = []
        self._execve_error = execve_error
        # Deliberately NO getpid(): the fixed code must not gate execve on
        # os.getpid() == 1 (the old terminal-detach path). Calling it would
        # AttributeError, failing the test.
    def execve(self, *args):
        if self._execve_error is not None:
            raise self._execve_error
        self.execve_calls.append(args)
        raise _ExecveReturned
    def _exit(self, code):
        self.exit_calls.append(code)


class _FakeSys:
    class stdout:
        flush = staticmethod(lambda: None)
    class stderr:
        flush = staticmethod(lambda: None)


class _FakeSubprocess:
    def __init__(self):
        self.popen_calls = []
    def Popen(self, argv, **kwargs):
        self.popen_calls.append((argv, kwargs))


def _run_spawn_or_exec(path: Path, os_, sys_, subprocess_):
    ns = {"os": os_, "sys": sys_, "subprocess": subprocess_}
    exec(compile(_spawn_or_exec_source(path), str(path), "exec"), ns)
    return ns["_spawn_or_exec"]


class DriverRelaunchCtrlCTest(unittest.TestCase):
    def _test_primary_path(self, path):
        os_ = _FakeOS()
        fn = _run_spawn_or_exec(path, os_, _FakeSys(), _FakeSubprocess())
        argv = ["/interp", "/script", "--task", "g1"]
        env = {"SIMULATOR": "genesis"}
        with self.assertRaises(_ExecveReturned):
            fn(argv, env)
        self.assertEqual(os_.execve_calls, [(argv[0], argv, env)],
                         f"{path.name}: execve must be reached in-place (not PID-1-gated)")
        self.assertEqual(os_.exit_calls, [],
                         f"{path.name}: must not os._exit() after a successful execve")

    def test_execve_in_place_in_every_driver(self):
        for path in DRIVERS:
            with self.subTest(driver=path.name):
                self._test_primary_path(path)

    def test_primary_path_never_detached_spawns(self):
        """The bug it's guarding: the old body reached
        subprocess.Popen(start_new_session=True) + os._exit(0) for every
        non-PID-1 process. The fixed body must not call Popen on success."""
        for path in DRIVERS:
            with self.subTest(driver=path.name):
                os_ = _FakeOS()
                subprocess_ = _FakeSubprocess()
                fn = _run_spawn_or_exec(path, os_, _FakeSys(), subprocess_)
                with self.assertRaises(_ExecveReturned):
                    fn(["py", "s", "--task", "g1"], {})
                self.assertEqual(
                    subprocess_.popen_calls, [],
                    f"{path.name}: a successful execve must not fall through to a "
                    "detached Popen -- that is the Ctrl+C-stale-port bug",
                )

    def test_falls_back_to_detached_spawn_only_on_execve_failure(self):
        for path in DRIVERS:
            with self.subTest(driver=path.name):
                os_ = _FakeOS(execve_error=OSError("no such interpreter"))
                subprocess_ = _FakeSubprocess()
                fn = _run_spawn_or_exec(path, os_, _FakeSys(), subprocess_)
                argv = ["/missing/interp", "/script"]
                env = {"SIMULATOR": "genesis"}
                fn(argv, env)  # must not raise: fallback exits the process
                self.assertEqual(
                    subprocess_.popen_calls,
                    [(argv, {"start_new_session": True, "env": env})],
                    f"{path.name}: execve failure must fall back to spawn-detached",
                )
                self.assertEqual(os_.exit_calls, [0])


if __name__ == "__main__":
    unittest.main()
