#!/bin/bash
# Cluster environment setup for RAG project
# Usage: source scripts/setup_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Cache base - configurable via DATA_ROOT env var, defaults to /data/$USER
DATA_ROOT="${DATA_ROOT:-/data/$USER}"
CACHE_BASE="${DATA_ROOT}/.cache"

# Set cache paths
export HF_HOME="${CACHE_BASE}/huggingface"
export NLTK_DATA="${CACHE_BASE}/nltk_data"
export MPLCONFIGDIR="${CACHE_BASE}/matplotlib"

# Create cache directories if needed
mkdir -p "$HF_HOME" "$NLTK_DATA" "$MPLCONFIGDIR"

# Create venv if it doesn't exist
if [ ! -d "${PROJECT_DIR}/.venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${PROJECT_DIR}/.venv"
fi

# Activate venv
source "${PROJECT_DIR}/.venv/bin/activate"

# Install requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

# Verify GPU
echo "Verifying GPU availability..."
python3 -c "import torch; avail = torch.cuda.is_available(); print(f'GPU available: {avail}'); exit(0 if avail else 1)"

echo "Environment setup complete!"
