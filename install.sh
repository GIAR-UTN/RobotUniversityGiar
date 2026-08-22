#!/usr/bin/env bash
# One-command native setup for macOS/Linux — mirrors README.md §2 exactly.
# Not needed if you're using Docker Compose (see README §2's "Docker Compose"
# subsection) — that path has no venv/pip steps at all.
set -euo pipefail

RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; CYAN='\033[36m'; RESET='\033[0m'

WITH_KAGGLE=0
WITH_MJLAB=0
for arg in "$@"; do
  case "$arg" in
    --with-kaggle) WITH_KAGGLE=1 ;;
    --with-mjlab) WITH_MJLAB=1 ;;
    -h|--help)
      echo "Usage: ./install.sh [--with-kaggle] [--with-mjlab]"
      echo "  --with-kaggle   also install the 'kaggle' package for --backend kaggle training jobs"
      echo "  --with-mjlab    also build the SEPARATE .venv-mjlab for mjlab motion-tracking"
      echo "                  tasks (e.g. Rugiar-G1-Mimic). See docs/compute_backends.md."
      exit 0
      ;;
  esac
done

if ! command -v python3.12 &>/dev/null; then
  echo -e "${RED}python3.12 not found.${RESET}"
  echo -e "${YELLOW}macOS:${RESET}  brew install python@3.12"
  echo -e "${YELLOW}Linux:${RESET}  sudo apt install python3.12 python3.12-venv   (Debian/Ubuntu; use your distro's package manager otherwise)"
  exit 1
fi

echo -e "${CYAN}[1/6]${RESET} Creating venv at .venv ..."
python3.12 -m venv .venv

echo -e "${CYAN}[2/6]${RESET} Activating .venv ..."
# shellcheck source=/dev/null
source .venv/bin/activate

echo -e "${CYAN}[3/6]${RESET} Installing Python dependencies (this can take a few minutes) ..."
pip install --upgrade pip
pip install torch torchvision matplotlib tensorboard xlsxwriter pandas tqdm scipy pygame trimesh rich-argparse viser
pip install genesis-world warp-lang
pip install -e .

if [ "$WITH_KAGGLE" -eq 1 ]; then
  echo -e "${CYAN}[4/6]${RESET} Installing Kaggle cloud-training extra ..."
  pip install -e .[cloud]
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "${YELLOW}No ~/.kaggle/kaggle.json found yet — see README §2 'Kaggle (cloud GPU)' for how to get one.${RESET}"
  fi
else
  echo -e "${CYAN}[4/6]${RESET} Skipping Kaggle extra (pass --with-kaggle to include it)."
fi

# mjlab MUST live in its own venv, never in .venv above: the repo vendors a
# top-level rsl_rl/ that shadows PyPI rsl-rl-lib, and genesis pins mujoco 3.10
# while mjlab needs 3.11 (see docs/mjlab_migration.md R1, docs/compute_backends.md).
# Deliberately built with a SUBSHELL-free explicit interpreter path instead of
# `source`, so the .venv activated above stays the active one for this script.
if [ "$WITH_MJLAB" -eq 1 ]; then
  echo -e "${CYAN}[5/6]${RESET} Creating the separate mjlab venv at .venv-mjlab ..."
  python3.12 -m venv .venv-mjlab
  ./.venv-mjlab/bin/pip install --upgrade pip
  ./.venv-mjlab/bin/pip install -e .[mjlab]
else
  echo -e "${CYAN}[5/6]${RESET} Skipping mjlab venv (pass --with-mjlab to build .venv-mjlab)."
fi

echo -e "${CYAN}[6/6]${RESET} Done."
echo ""
echo -e "${GREEN}Setup complete.${RESET} Every new terminal session needs:"
echo -e "  ${CYAN}source .venv/bin/activate${RESET}"
echo -e "  ${CYAN}export SIMULATOR=genesis${RESET}   # required — legged_gym refuses to import without this set"
echo ""
if [ "$WITH_MJLAB" -eq 1 ]; then
  echo -e "${GREEN}.venv-mjlab is ready too.${RESET} You never activate it by hand —"
  echo -e "  ${CYAN}rugiar train${RESET} picks the right interpreter per task. See docs/compute_backends.md."
  echo ""
fi
echo "Next: README.md §2 'Train a policy' or 'Run the policy-switching demo'."
echo "Where each backend actually runs: docs/compute_backends.md"
