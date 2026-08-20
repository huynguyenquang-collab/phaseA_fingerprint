#!/usr/bin/env bash
# Minimal, documented compatibility patch (not an upstream source edit):
#
# LLaMA-Factory's src/llamafactory/data/mm_plugin.py does an unconditional
# `import torchaudio` at module level (for optional audio/multimodal input
# support) which the CTCC pipeline here never uses — it's text-only,
# llama2 template. On a box whose preinstalled torch is newer than any
# published torchaudio wheel (e.g. torch 2.13.0+cu132 with only
# torchaudio<=2.11.0 available, built against an older CUDA toolkit),
# `import torchaudio` raises at import time
# (torchaudio._extension._check_cuda_version()), breaking every
# `llamafactory-cli train ...` call before it even reaches CTCC's data.
#
# Fix: uninstall the CUDA-mismatched torchaudio build and replace it with a
# zero-functionality stub package so the import succeeds. Nothing in this
# repo's usage of LLaMA-Factory calls torchaudio.* — only the import itself
# needs to succeed. Skip this script entirely if `python -c "import
# torchaudio"` already works on your box.
set -euo pipefail

source /venv/main/bin/activate 2>/dev/null || true  # no-op outside the vast.ai base image

if python -c "import torchaudio" 2>/dev/null; then
    echo "[patch_torchaudio_stub] torchaudio already importable, nothing to do"
    exit 0
fi

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
pip uninstall -y torchaudio 2>/dev/null || true

mkdir -p "$SITE/torchaudio"
cat > "$SITE/torchaudio/__init__.py" <<'STUB'
"""Deliberate minimal compatibility stub, NOT a real torchaudio install.
See scripts/patch_torchaudio_stub.sh for why this exists.
"""
STUB

python -c "import torchaudio; print('[patch_torchaudio_stub] stub import OK:', torchaudio.__file__)"
