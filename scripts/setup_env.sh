#!/usr/bin/env bash
# One-time environment setup for a single-GPU A100 40GB box.
#
# Prefers an already-activated venv with a working torch/CUDA stack
# (e.g. vast.ai base images ship one at /venv/main) over creating a fresh
# one from scratch — reinstalling the whole CUDA-linked torch stack is slow
# and risks pulling a build that doesn't match the box's driver. Falls back
# to creating ./.venv only if nothing is already active.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the existing if_awq_tier0 checkout (reused quantization backends)}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f /venv/main/bin/activate ]; then
        echo "[setup_env] no venv active, using vast.ai's preinstalled /venv/main"
        source /venv/main/bin/activate
    else
        echo "[setup_env] no venv active and /venv/main not found, creating ./.venv"
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
    fi
else
    echo "[setup_env] reusing already-active venv at $VIRTUAL_ENV"
fi

PIP_INSTALL="pip install"
command -v uv >/dev/null 2>&1 && PIP_INSTALL="uv pip install"

$PIP_INSTALL -r requirements.txt
$PIP_INSTALL -r "$IF_AWQ_TIER0_ROOT/requirements.txt"

if [ -d third_party/scalable_fp ]; then
    # NOT `pip install -r third_party/scalable_fp/requirements.txt`: that file is a
    # full frozen snapshot from the paper authors' own old dev env (torch==2.3.1,
    # transformers==4.44.2, autoawq==0.2.6 with no cp312 wheel...) — installing it
    # verbatim both fails (autoawq) and would downgrade the box's working
    # torch/transformers out from under if_awq_tier0. We only need the handful of
    # packages its scripts actually import that a modern torch/transformers/
    # accelerate/datasets/peft stack doesn't already provide (verified by grepping
    # generate_finetuning_data.py / finetune_multigpu.py / check_fingerprints.py /
    # fingerprint_dataloader.py's own imports).
    $PIP_INSTALL deepspeed wandb blobfile fairscale zstandard
fi

if [ -d third_party/llama_factory ]; then
    # This fork's pyproject has no [torch]/[metrics] extras (verified: `pip install
    # -e .[torch,metrics]` warns "does not have an extra named ...") — install base only.
    $PIP_INSTALL -e third_party/llama_factory
    bash "$ROOT/scripts/patch_torchaudio_stub.sh"
fi

mkdir -p results
python --version > results/environment.txt
pip freeze >> results/environment.txt
nvidia-smi > results/nvidia_smi.txt || echo "nvidia-smi not available (no GPU on this machine)" > results/nvidia_smi.txt

echo "[setup_env] done. Active venv: ${VIRTUAL_ENV:-unknown}"
