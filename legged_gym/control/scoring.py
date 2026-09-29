"""scoring.py — turns one run's raw metrics (distance_m, elapsed_s) into a
single comparable number: progress dominates, velocity is a per-scenario-
weighted secondary term. See the RuGIAR/INNOVATON scoring design discussion
(2026-09) for the full reasoning behind this shape — summarized:

    score = progress * 100 + avg_velocity * SPEED_WEIGHT[scenario]

- `progress` (0-1) is distance_m normalized by the scenario's own track
  length, capped at 1 — so a run that reaches the finish line always scores
  progress=1 regardless of scenario length, letting scores from different
  tracks sum meaningfully.
- `avg_velocity` (distance_m / elapsed_s) needs no external reference time —
  it's computed entirely from the run itself, deliberately, because no
  organizer-picked target/max time exists to calibrate against.
- SPEED_WEIGHT is calibrated per scenario, not globally, because 'race' is
  finished by nearly every attempt (progress pins near 1, so speed is the
  ONLY thing that differentiates runs there), while obstacle_course/
  agility_course are rarely finished (progress does almost all the work,
  speed is just a tiebreak among similar-distance attempts).

Score for the competition ranking is: for each team, take the best
score_run() across all their attempts at a scenario, then sum across the
three graded scenarios (race + obstacle_course + agility_course). That
"take the max per scenario, sum across scenarios" step lives at the
leaderboard-aggregation layer, not here — this module only scores one run.
"""
from __future__ import annotations

from typing import Optional

from .run_recorder import RECORDED_SCENARIOS

SPEED_WEIGHT = {
    "race": 20.0,             # progress ~always 1 here -- speed IS the score
    "obstacle_course": 5.0,   # progress rarely reaches 1 -- speed just tiebreaks
    "agility_course": 5.0,
}


def track_length_for(scenario: str) -> Optional[float]:
    """The scenario's own track length, straight from its Scenario
    registration (legged_gym/utils/scenarios.py) — same value web/app.js's
    raceTrackLength already uses client-side for the footer/finish-line
    detection, so scoring and the live UI agree on what "100% of the
    track" means. Uses default_options only (no --scenario-option
    overrides) — scoring assumes graded runs use the scenario's stock
    length; a run started with a track_length override would need that
    override captured in the manifest to score correctly, which isn't
    done today.

    Imports legged_gym.utils.scenarios lazily, on purpose: that module's
    package (legged_gym.utils) pulls in rsl_rl (task_registry.py), which
    isn't installed under .venv-mjlab (see legged_gym/__init__.py's
    SIMULATOR="mjlab" branch docstring). RECORDED_SCENARIOS are
    Genesis-only anyway (scenarios.py's own --scenario flag errors under
    --system mjlab), so this import never actually executes in an mjlab
    process — a module-level import would still break ControlService's
    importability there for a code path that can never run."""
    from legged_gym.utils.scenarios import SCENARIOS

    sc = SCENARIOS.get(scenario)
    if sc is None:
        return None
    opts = dict(sc.default_options)
    return sc.web_options(opts).get("track_length")


def score_run(scenario: str, distance_m: Optional[float], elapsed_s: Optional[float]) -> Optional[float]:
    """None (not 0) whenever there isn't enough to score from — an
    ungraded scenario, a scenario with no track length, or a run with no
    distance recorded (e.g. 'aborted' before any telemetry came in). None
    scores are excluded from the "best attempt" max, same as if the
    attempt never happened, rather than counting as a real 0-point run."""
    if scenario not in RECORDED_SCENARIOS:
        return None
    track_length = track_length_for(scenario)
    if not track_length or distance_m is None:
        return None

    progress = min(max(distance_m, 0.0) / track_length, 1.0)
    velocity = (distance_m / elapsed_s) if elapsed_s else 0.0
    return progress * 100 + velocity * SPEED_WEIGHT.get(scenario, 0.0)
