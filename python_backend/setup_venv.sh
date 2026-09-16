#!/usr/bin/env bash
# Create an isolated Python environment for the Bio-Block Python backend.
# Prefers `uv` when available (fast); falls back to stdlib `venv`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d "$VENV_DIR" ]]; then
  echo "Virtual environment already exists at $ROOT/$VENV_DIR"
else
  if command -v uv >/dev/null 2>&1; then
    echo "Creating virtual environment with uv..."
    uv venv "$VENV_DIR"
  else
    echo "Creating virtual environment with $PYTHON_BIN -m venv..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  pip install -r requirements.txt
fi

# spaCy model used by image PHI anonymization (optional but recommended)
python -m spacy download en_core_web_sm || true

echo
echo "Setup complete."
echo "Activate with:  source $ROOT/$VENV_DIR/bin/activate"
echo "Run with:       uvicorn main:app --reload --port 3002"
