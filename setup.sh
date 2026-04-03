#!/usr/bin/env bash
# setup.sh - Bootstrap a Python venv and install all project dependencies.
# Usage: bash setup.sh

set -e

VENV_DIR=".venv"

echo "==> Creating Python virtual environment in ${VENV_DIR}/"
python3 -m venv "${VENV_DIR}"

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing project dependencies"
pip install -r requirements.txt

echo "==> Downloading NLTK data"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo ""
echo "Setup complete!"
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
echo ""
echo "Directory structure:"
echo "  src/data/           - Dataset download & preprocessing"
echo "  src/models/         - PyTorch model & dataset classes"
echo "  src/training/       - Training loop with early stopping"
echo "  src/temporal/       - TemporalStressProfile & velocity"
echo "  src/recommender/    - 3-layer recommendation engine"
echo "  src/api/            - FastAPI backend"
echo "  ui/                 - Streamlit dashboard"
echo "  data/               - Raw & processed datasets (git-ignored)"
echo "  tests/              - Unit tests"
