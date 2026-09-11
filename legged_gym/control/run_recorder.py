"""RunRecorder — writes a competition run's manifest + vectorized trajectory
to disk (runs/<scenario>/<team_id>/<run_id>/).

This is evidence, not a verdict: the score that decides the INNOVATON
ranking is never read back out of manifest.json as-submitted — it gets
recomputed by re-simulating trajectory.jsonl (state + commands) on
organizer-controlled infra. A team's fork can write whatever it wants here;
what it writes just stops being trusted the moment it's re-simulated. See
the RuGIAR scoring design discussion (2026-09) for the full reasoning.

Simulator-only for now — RobotState's sim-ground-truth fields this module
records (base_pos_xy, base_height, base_lin_vel) are None on RealAdapter, so
recording against a real robot would silently produce a mostly-empty log
rather than a useful one. Don't wire this into a real-hardware run without
picking a different set of fields first.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Optional

# 20 Hz — dense enough to reconstruct/replay a run (finish-line crossings,
# falls, the operator's actual command stream) without trajectory.jsonl
# growing at full physics-step rate (usually 50-200 Hz) across a whole
# semester of attempts, most of which are just a team iterating.
RECORD_INTERVAL_S = 0.05

RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"

# INNOVATON judges these three scenarios (see legged_gym/utils/scenarios.py's
# SCENARIOS registry for the full list, most of which are ungraded demo/dev
# scenarios) — start() is a no-op for anything else, so a stray/typo'd
# scenario name from the client can't open a run directory at all, let
# alone one that would ever be mistaken for a real attempt.
RECORDED_SCENARIOS = frozenset({"race", "obstacle_course", "agility_course"})


def _hash_policy_dir(path: Path) -> Optional[str]:
    """sha256 over every file under policies/<name>/, sorted by relative
    path so the hash doesn't depend on directory listing order. This is the
    'policy_sha256' — detects a team swapping the checkpoint file between
    runs without renaming it. None if the directory doesn't exist or is
    empty (e.g. a policy name the caller mistyped)."""
    if not path.is_dir():
        return None
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


class RunRecorder:
    """One instance lives on ControlService. start()/stop() bracket a run;
    record_tick() is called every control tick from ControlService.tick()
    and self-throttles to RECORD_INTERVAL_S — the caller doesn't need to
    know or care about the recording rate."""

    def __init__(self, root: Path = RUNS_ROOT):
        self.root = root
        self._run_dir: Optional[Path] = None
        self._traj_file = None
        self._manifest: Optional[dict] = None
        self._last_tick_written: float = float("-inf")

    @property
    def active(self) -> bool:
        return self._run_dir is not None

    @property
    def scenario(self) -> Optional[str]:
        """The active run's scenario, or None if nothing's open — lets a
        caller (ControlService.end_run(), to pick the right scoring
        formula) ask before stop() clears the manifest, without reaching
        into this class's private state."""
        return self._manifest["scenario"] if self._manifest else None

    def start(self, scenario: str, team_id: str, policy_name: str,
              policies_dir: Path, code_fingerprint: Optional[str] = None) -> dict:
        """Opens runs/<scenario>/<team_id>/<run_id>/ and writes the
        manifest's static half. A run already in progress is closed first
        with outcome 'superseded' rather than silently overwritten or
        raising — mirrors ControlService.restart()'s "the new intent wins"
        behavior elsewhere in this class.

        No-ops (returns {}) if `scenario` isn't in RECORDED_SCENARIOS —
        graded attempts only; nothing gets written for a dev/demo scenario
        like 'default' or 'rough_terrain'."""
        if scenario not in RECORDED_SCENARIOS:
            return {}

        if self.active:
            self.stop(outcome="superseded")

        run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._run_dir = self.root / scenario / team_id / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._traj_file = (self._run_dir / "trajectory.jsonl").open("w")
        self._last_tick_written = float("-inf")

        self._manifest = {
            "run_id": run_id,
            "scenario": scenario,
            "team_id": team_id,
            "policy_name": policy_name,
            "policy_sha256": _hash_policy_dir(policies_dir / policy_name),
            "code_fingerprint": code_fingerprint,
            "started_at": time.time(),
            "ended_at": None,
            "outcome": None,
            "score": None,
            "metrics": None,
        }
        return dict(self._manifest)

    def record_tick(self, *, t: float, pos, gravity, lin_vel, ang_vel,
                     command, active_policy: Optional[str]) -> None:
        """`t` is sim/wall time in seconds, monotonically increasing across
        the run — not wall-clock `time.time()`, so a paused sim doesn't
        burn through the 20 Hz budget while frozen. Every arg but `t` is
        whatever ControlService.tick() already has in hand each step (see
        RobotState) — this method does no tensor math, just rate-limits and
        serializes."""
        if not self.active:
            return
        if t - self._last_tick_written < RECORD_INTERVAL_S:
            return
        self._last_tick_written = t
        row = {
            "t": t,
            "pos": pos,
            "gravity": gravity,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
            "cmd": command,
            "policy": active_policy,
        }
        self._traj_file.write(json.dumps(row) + "\n")

    def stop(self, outcome: str, score: Optional[float] = None,
              metrics: Optional[dict] = None) -> Optional[dict]:
        """outcome is caller-declared (e.g. 'finished' / 'fell' / 'timeout'
        / 'manual_restart' / 'superseded') — this class has no opinion on
        what counts as a fall or a finish line; see this module's docstring
        on why that decision stays where the terminal-state detection
        already lives rather than being duplicated here.

        `metrics` is the raw per-scenario numbers the caller already had at
        hand (e.g. {"elapsed_s": ..., "distance_m": ...}) — kept separate
        from `score` on purpose: turning "12.4s, fell at 8m" into one
        cross-scenario number comparable across race/obstacle_course/
        agility_course is a scoring-formula decision this class doesn't
        make. `score` stays None until that formula exists; `metrics` is
        what it will be computed FROM, not a substitute for it."""
        if not self.active:
            return None
        self._traj_file.close()
        self._manifest["ended_at"] = time.time()
        self._manifest["outcome"] = outcome
        self._manifest["score"] = score
        self._manifest["metrics"] = metrics
        (self._run_dir / "manifest.json").write_text(json.dumps(self._manifest, indent=2))
        manifest = dict(self._manifest)
        self._run_dir = None
        self._traj_file = None
        self._manifest = None
        return manifest
