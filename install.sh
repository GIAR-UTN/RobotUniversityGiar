#!/usr/bin/env bash
# One-command native setup for macOS/Linux — mirrors README.md §2 exactly.
# Not needed if you're using Docker Compose (see README §2's "Docker Compose"
# subsection) — that path has no venv/pip steps at all.
set -euo pipefail

RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; CYAN='\033[36m'; RESET='\033[0m'

WITH_KAGGLE=0
for arg in "$@"; do
  case "$arg" in
    --with-kaggle) WITH_KAGGLE=1 ;;
    -h|--help)
      echo "Usage: ./install.sh [--with-kaggle]"
      echo "  --with-kaggle   also install the 'kaggle' package for --backend kaggle training jobs"
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

echo -e "${CYAN}[1/5]${RESET} Creating venv at .venv ..."
python3.12 -m venv .venv

echo -e "${CYAN}[2/5]${RESET} Activating .venv ..."
# shellcheck source=/dev/null
source .venv/bin/activate

echo -e "${CYAN}[3/5]${RESET} Installing Python dependencies (this can take a few minutes) ..."
pip install --upgrade pip
pip install torch torchvision matplotlib tensorboard xlsxwriter pandas tqdm scipy pygame trimesh rich-argparse viser
pip install genesis-world warp-lang
pip install -e .

if [ "$WITH_KAGGLE" -eq 1 ]; then
  echo -e "${CYAN}[4/5]${RESET} Installing Kaggle cloud-training extra ..."
  pip install -e .[cloud]
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "${YELLOW}No ~/.kaggle/kaggle.json found yet — see README §2 'Kaggle (cloud GPU)' for how to get one.${RESET}"
  fi
else
  echo -e "${CYAN}[4/5]${RESET} Skipping Kaggle extra (pass --with-kaggle to include it)."
fi

echo -e "${CYAN}[5/5]${RESET} Done."
echo ""
echo -e "${GREEN}Setup complete.${RESET} Every new terminal session needs:"
echo -e "  ${CYAN}source .venv/bin/activate${RESET}"
echo -e "  ${CYAN}export SIMULATOR=genesis${RESET}   # required — legged_gym refuses to import without this set"
echo ""
echo "Next: README.md §2 'Train a policy' or 'Run the policy-switching demo'."
