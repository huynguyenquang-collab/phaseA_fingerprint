#!/usr/bin/env bash
# Clone and pin the official upstream repos for Phase A (F2/F3/F4).
# Never edit these checkouts in place - see README "Do not edit upstream".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p third_party

clone_pin () {
    local url="$1" dir="$2"
    if [ -d "third_party/$dir/.git" ]; then
        echo "[setup_third_party] $dir already cloned, skipping"
    else
        git clone "$url" "third_party/$dir"
    fi
}

# F2 (ENGLISH-RANDOM) + F3 (Perinucleus): same upstream repo/paper.
clone_pin "https://github.com/SewoongLab/scalable-fingerprinting-of-llms.git" "scalable_fp"

# Minimal documented compatibility patch (found running against real weights on
# an A100 box): generate_finetuning_data.py's `key_response_strategy=independent`
# path calls `pipeline(text, max_length=N, ...)` where N is meant as a "how many
# new tokens" budget, but a newer transformers (5.x) validates `max_length` as a
# TOTAL sequence length cap and hard-errors as soon as the tokenized input alone
# already reaches N (ValueError: "Input length of input_ids is N, but
# `max_length` is set to N ... consider setting `max_new_tokens`" - literally the
# library's own suggested fix). This never surfaced against the repo's own pinned
# transformers==4.44.2, which validated this loosely. Swapping max_length->
# max_new_tokens for these 4 call sites preserves the exact same intent (N new
# tokens for the key/response) and fixes it for any transformers version.
if [ -d "third_party/scalable_fp/.git" ] && ! grep -q "max_new_tokens=key_length" "third_party/scalable_fp/generate_finetuning_data.py"; then
    echo "[setup_third_party] applying scalable_fp_max_new_tokens.patch"
    git -C third_party/scalable_fp apply "$ROOT/third_party/patches/scalable_fp_max_new_tokens.patch"
fi

# Resource-sizing fix (found running the actual F2 finetune step: killed by
# the kernel OOM killer, exit code -9, container cgroup hit its ~120GB RAM
# limit). finetune_multigpu.py's ZeRO-2 config offloads BOTH the optimizer
# state and the model parameters to CPU RAM unconditionally. That is real
# memory engineering the repo's own multi-GPU authors needed, but on a single
# A100 40GB the bf16 params+gradients (~28GB) fit comfortably in GPU memory
# on their own - only the fp32 Adam optimizer state (the genuinely large
# piece, ~56GB for a 7B model) needs CPU offload. Dropping `offload_param`
# keeps `offload_optimizer` (the actual RAM-saving part) and changes no
# training math, only where the buffers physically live.
if [ -d "third_party/scalable_fp/.git" ] && grep -q "'offload_param': {'device': 'cpu'" "third_party/scalable_fp/finetune_multigpu.py"; then
    echo "[setup_third_party] applying scalable_fp_no_param_offload.patch"
    git -C third_party/scalable_fp apply "$ROOT/third_party/patches/scalable_fp_no_param_offload.patch"
fi

# Minimal documented compatibility patch (also found running against real
# weights): finetune_multigpu.py's TrainingArguments(...) call passes both
# `eval_strategy='epoch'` (current API) and `evaluation_strategy="epoch"`
# (the old, since-removed alias) - transformers 5.x rejects the unknown
# `evaluation_strategy` kwarg outright (`TypeError: ... unexpected keyword
# argument 'evaluation_strategy'`). Dropping the duplicate, deprecated line
# changes nothing about training (eval_strategy='epoch' already does the
# same thing) and fixes it for any transformers version.
if [ -d "third_party/scalable_fp/.git" ] && grep -q 'evaluation_strategy="epoch"' "third_party/scalable_fp/finetune_multigpu.py"; then
    echo "[setup_third_party] applying scalable_fp_drop_duplicate_evaluation_strategy.patch"
    git -C third_party/scalable_fp apply "$ROOT/third_party/patches/scalable_fp_drop_duplicate_evaluation_strategy.patch"
fi

# Minimal documented compatibility patch (also found running against real
# weights): finetune_multigpu.py passes `report_to=None` to TrainingArguments,
# but transformers 5.x's callback resolution rejects a bare `None`
# (`ValueError: None is not supported, only azure_ml, comet_ml, ... wandb, ...
# are supported`) - it now requires the literal string "none" to mean "report
# to nothing". Changes no training behavior, only silences all integrations
# exactly as the original None was intended to.
if [ -d "third_party/scalable_fp/.git" ] && grep -q 'report_to=None,' "third_party/scalable_fp/finetune_multigpu.py"; then
    echo "[setup_third_party] applying scalable_fp_report_to_none_string.patch"
    git -C third_party/scalable_fp apply "$ROOT/third_party/patches/scalable_fp_report_to_none_string.patch"
fi

# F4 (CTCC): datasets + LoRA-merge/eval helper scripts. Training itself runs
# through LLaMA-Factory, which the repo's own README depends on (it ships no
# training code of its own).
clone_pin "https://github.com/Xuzhenhua55/CTCC.git" "ctcc"

# CTCC's documented training/inference engine.
clone_pin "https://github.com/hiyouga/LLaMA-Factory.git" "llama_factory"

{
    echo "# Upstream revisions (Phase A: F2/F3/F4)"
    echo
    echo "Regenerated by scripts/setup_third_party.sh — do not hand-edit commit hashes."
    echo
    for dir in scalable_fp ctcc llama_factory; do
        commit=$(git -C "third_party/$dir" rev-parse HEAD)
        url=$(git -C "third_party/$dir" remote get-url origin)
        echo "- $dir"
        echo "  - repo: $url"
        echo "  - commit: \`$commit\`"
        echo
    done
} > third_party/UPSTREAM_VERSIONS.md

cat third_party/UPSTREAM_VERSIONS.md
echo "[setup_third_party] pinned revisions written to third_party/UPSTREAM_VERSIONS.md"
