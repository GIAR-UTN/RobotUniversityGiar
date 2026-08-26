#!/bin/bash
set -e

# Activate the uv-managed venv so every python/pip call uses it.
# The venv is created by the Dockerfile at /workspace/RobotUniversityGiar/.venv.
. /workspace/RobotUniversityGiar/.venv/bin/activate

# If manual arguments are passed (e.g. `docker run ... bash` or
# `docker run ... python ...`), execute them directly and skip the
# automatic rugiar_driver launch.
if [ $# -gt 0 ]; then
    exec "$@"
fi

# ---------------------------------------------------------------------------
# Automatic launch of rugiar_driver.py
# ---------------------------------------------------------------------------
# Policies are discovered at runtime by TrainingManager.discover_local_policies()
# inside rugiar_driver.py itself — it scans ./policies/<name>/ folders for
# checkpoint.pt + meta.json.  No shell-level discovery here; that avoids
# duplicating the logic and keeps the Python side as the single source of truth.
#
# Configure via environment variables in docker-compose.yml or a .env file:
#
#   ACTIVE_POLICY=<name>   (must match a policies/<name>/ folder)
#   CONTROL_PORT=<int>     (default: 9017 — registered in `las ports ls` as
#                           "GIAR docker-compose control"; keep the registry
#                           in sync if you change this)
#   VISER_PORT=<int>       (default: 9006 — "GIAR docker-compose viser" in
#                           `las ports ls`)
#   HEADLESS=1|0           (default: 0)
#   SPEED=<float>          (default: 0.35)
#   SCENARIO=ball|race     (default: unset -- no scenario)
# ---------------------------------------------------------------------------

export GENESIS_BACKEND=${GENESIS_BACKEND:-cpu}

ARGS=()

# ---------------------------------------------------------------------------
# Active policy
# ---------------------------------------------------------------------------
if [ -n "$ACTIVE_POLICY" ]; then
    ARGS+=("--active" "$ACTIVE_POLICY")
fi

# ---------------------------------------------------------------------------
# Optional arguments
# ---------------------------------------------------------------------------
if [ -n "$CONTROL_PORT" ]; then
    ARGS+=("--control_port" "$CONTROL_PORT")
else
    ARGS+=("--control_port" "9017")
fi

if [ -n "$VISER_PORT" ]; then
    ARGS+=("--viser_port" "$VISER_PORT")
else
    ARGS+=("--viser_port" "9006")
fi

if [ "$HEADLESS" = "1" ] || [ "$HEADLESS" = "true" ]; then
    ARGS+=("--headless")
fi

if [ -n "$SPEED" ]; then
    ARGS+=("--speed" "$SPEED")
fi

if [ -n "$SCENARIO" ]; then
    ARGS+=("--scenario" "$SCENARIO")
fi

cd /workspace/RobotUniversityGiar
exec python legged_gym/scripts/rugiar_driver.py "${ARGS[@]}"
