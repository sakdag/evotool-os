#!/usr/bin/env bash
# One-shot environment bootstrap for EVOTOOL
set -e

CONDA_ENV_NAME="evotool"
PYTHON_VERSION="3.11"

echo "=== EVOTOOL Environment Setup ==="

# Create and activate conda environment
conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

echo "--- Installing Python dependencies ---"
pip install -r requirements.txt

echo "--- Installing tau-bench from GitHub ---"
pip install git+https://github.com/sierra-research/tau-bench

echo "--- Creating experiment directories ---"
mkdir -p experiments/results
mkdir -p experiments/logs
mkdir -p experiments/bfcl/result

echo "--- Smoke tests ---"
python -c "import tau_bench; print('[OK] tau-bench installed')" || echo "[WARN] tau-bench import failed"
python -c "import bfcl_eval; print('[OK] bfcl-eval installed')" || echo "[WARN] bfcl-eval import failed"
python -c "import vllm; print('[OK] vllm installed')" || echo "[WARN] vllm import failed"
python -c "import openai; print('[OK] openai installed')"

echo "=== Setup complete. Activate with: conda activate $CONDA_ENV_NAME ==="
