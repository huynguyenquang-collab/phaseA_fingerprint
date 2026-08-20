#!/usr/bin/env bash
# One-time environment setup for a single-GPU A100 40GB box.
# Mirrors the if_awq_tier0 convention: a separate venv, requirements pinned,
# environment recorded for reproducibility (student plan section 12/22).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the existing if_awq_tier0 checkout (reused quantization backends)}"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r "$IF_AWQ_TIER0_ROOT/requirements.txt"

if [ -d third_party/scalable_fp ]; then
    pip install -r third_party/scalable_fp/requirements.txt
fi

if [ -d third_party/llama_factory ]; then
    pip install -e "third_party/llama_factory[torch,metrics]"
fi

mkdir -p results
python --version > results/environment.txt
pip freeze >> results/environment.txt
nvidia-smi > results/nvidia_smi.txt || echo "nvidia-smi not available (no GPU on this machine)" > results/nvidia_smi.txt

echo "[setup_env] done. Activate with: source .venv/bin/activate"
