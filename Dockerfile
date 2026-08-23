# =============================================================================
# RobotUniversityGiar Dockerfile (Genesis — swap_experiment.py)
# =============================================================================
# This Dockerfile builds a self-contained image for the GIAR fork of
# RobotUniversityGiar, targeting the policy-switching demo in simulation.
#
# It is configured for:
#   * Python 3.12
#   * Genesis simulator  (CPU fallback by default, GPU libraries included)
#   * PyTorch 2.9.0 + CUDA 12.8 on linux/amd64 (includes Blackwell/sm_120
#     GPU support); generic CPU PyTorch on other architectures (e.g.
#     linux/arm64, such as Apple Silicon under Colima/Docker Desktop)
#   * viser web viewer (port 9006)
#   * FastAPI unified control web / WebSocket bridge (port 9013)
#
# This image builds and runs on any host architecture. On linux/amd64 with
# the NVIDIA Container Runtime (--gpus all), Genesis and PyTorch will see
# the CUDA device. Everywhere else — no GPU, or a non-amd64 arch where the
# CUDA wheels don't exist — the same image still runs because
# swap_experiment.py initialises Genesis with backend=gs.cpu unless
# GENESIS_BACKEND=cuda is set and available.
#
# Build:
#   docker build -t robotuniversitygiar:genesis .
#
# Run interactively:
#   docker run --rm -it -p 9006:9006 -p 9013:9013 \
#        -v $(pwd)/policies:/workspace/policies:ro \
#        robotuniversitygiar:genesis bash
#
# =============================================================================

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Prevent apt-get from asking questions
ENV DEBIAN_FRONTEND=noninteractive

# The legged_gym import machinery refuses to load unless this is set
ENV SIMULATOR=genesis

# uv HTTP timeout (some packages are large)
ENV UV_HTTP_TIMEOUT=300

# ---- Install uv (Rust-based Python package manager) --------------------------
# uv installs its own managed Python versions, creates venvs, and installs
# wheels without ever touching the system pip or distutils.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Install system dependencies ---------------------------------------------
# Graphics libraries (libgl, libegl) and libgomp are required by Genesis,
# viser and Pygame.  build-essential is needed for compiling any source
# distributions that lack pre-built wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libegl1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python 3.12 via uv ----------------------------------------------
RUN uv python install 3.12

# ---- Working directory -------------------------------------------------------
WORKDIR /workspace/RobotUniversityGiar

# ---- Copy dependency descriptor first (helps Docker layer caching) -----------
COPY pyproject.toml ./

# ---- Create venv and install dependencies from pyproject.toml -----------------
# pyproject.toml is the single source of truth:
#   * Base dependencies under [project] dependencies (includes unpinned
#     torch/torchvision so the image builds on any CPU architecture)
#   * Genesis-specific extras (genesis-world, warp-lang) under
#     [project.optional-dependencies] genesis
# A fresh resolution is generated on every build so the image always picks up
# latest compatible versions without requiring a committed lockfile.
#
# CUDA wheels (torch/torchvision +cu128, needed for Blackwell/sm_120 support)
# are only published for linux/amd64 — there is no matching wheel for
# linux/arm64 (e.g. this image built on Apple Silicon under Colima/Docker
# Desktop). So: resolve the base+genesis deps first (generic torch/torchvision,
# works on every arch), then on amd64 hosts *upgrade* torch/torchvision in
# place to the CUDA-pinned build. arm64 hosts keep the generic build and run
# Genesis on CPU (or Metal, outside Docker) via the GENESIS_BACKEND=cpu path.
RUN uv venv --python 3.12 .venv \
 && . .venv/bin/activate \
 && uv pip install -r pyproject.toml --extra genesis \
 && if [ "$(uname -m)" = "x86_64" ]; then \
        echo "x86_64 host: installing CUDA-enabled PyTorch (cu128, sm_120 support)"; \
        uv pip install "torch==2.9.0+cu128" "torchvision==0.24.0+cu128" \
            --extra-index-url https://download.pytorch.org/whl/cu128; \
    else \
        echo "$(uname -m) host: keeping generic CPU PyTorch (no CUDA wheels for this arch)"; \
    fi

# ---- mjlab venv (family switching to mjlab tasks, e.g. Rugiar-G1-Mimic) -----
# A second, fully separate venv mirroring the host setup
# (docs/mjlab_migration.md R1): mjlab pins mujoco~=3.11.0 while the genesis
# extra pins 3.10.0, and it needs PyPI rsl-rl-lib (installs as `rsl_rl`,
# colliding by name with this repo's vendored top-level rsl_rl/ package), so
# it can never share the main .venv. The repo package itself is deliberately
# NOT pip-installed here -- that would drop the vendored rsl_rl into this
# venv's site-packages and defeat the sys.path reorder rugiar_driver_mjlab.py
# applies; legged_gym/mjlab_tasks resolve off the repo root (PYTHONPATH),
# exactly like the host setup. The repo's BASE dependencies (from
# pyproject.toml, no extras -- matplotlib/xlsxwriter/torch/... are needed by
# legged_gym.control) are still installed, only the editable self-install is
# skipped. Without this venv, the control web's Family panel can't switch to
# an mjlab task from the container -- _relaunch_for_family() bails with
# "no mjlab venv at ...". jax (CPU) is an explicit addition: mujoco-warp
# needs it at runtime and mjlab only lists it as an optional extra.
RUN uv venv --python 3.12 .venv-mjlab \
 && . .venv-mjlab/bin/activate \
 && uv pip install -r pyproject.toml \
 && uv pip install "mjlab==1.6.0" "jax"

# ---- Copy the full repository ------------------------------------------------
COPY . .

# ---- Install the package itself in editable mode -----------------------------
# --no-deps is safe because all runtime requirements were installed above.
RUN . .venv/bin/activate && uv pip install -e . --no-deps

# Ensure imports work regardless of the current working directory
ENV PYTHONPATH=/workspace/RobotUniversityGiar
ENV PATH="/workspace/RobotUniversityGiar/.venv/bin:$PATH"

# ---- Copy the automatic entrypoint ------------------------------------------
COPY docker-entrypoint.sh /workspace/RobotUniversityGiar/docker-entrypoint.sh

# ---- Expose ports used by swap_experiment.py ---------------------------------
# 9006 -> viser web viewer
# 9013 -> unified control web (when --control_port 9013 is passed)
EXPOSE 9006 9013

# Invoking via bash avoids any host-side permission issues with the copied
# script's execute bit.
ENTRYPOINT ["bash", "/workspace/RobotUniversityGiar/docker-entrypoint.sh"]

# When no explicit command is given, the entrypoint launches swap_experiment.py.
# You can still drop into a shell with: docker run ... bash
CMD []
