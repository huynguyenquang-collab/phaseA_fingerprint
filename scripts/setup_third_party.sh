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

# Resource-sizing fix (measured live on the A100 box: killed by the kernel OOM
# killer, exit code -9, container cgroup hit its ~120GB RAM budget - confirmed
# independent of --batch_size, and independent of which parts of DeepSpeed's
# ZeRO-2 config were offloaded to CPU). Root cause: DeepSpeed's ZeRO-Offload
# ALWAYS substitutes its own fp32 CPU Adam implementation once
# `offload_optimizer` is configured, ignoring TrainingArguments' `optim=`
# entirely - so trying `optim="adamw_bnb_8bit"` alone (measured) had zero
# effect, still peaking at the same ~84GB optimizer-state footprint (28GB fp32
# master weights + 2x28GB Adam moments) for a 7B model. And with world_size=1
# (single GPU), ZeRO has no other rank to partition state across anyway - it
# adds only overhead here, no benefit. Fix: disable DeepSpeed entirely
# (`deepspeed=None`) and use bitsandbytes' `paged_adamw_8bit` directly - runs
# on GPU (measured peak ~40.4GB of 40GB, fits) and pages to host RAM only
# under pressure instead of always reserving ~84GB up front. Same learning
# rate/schedule; only the optimizer's internal state precision/placement
# changes (a standard technique for large-model fine-tuning under memory
# constraints, not a change to the fingerprinting objective).
if [ -d "third_party/scalable_fp/.git" ] && grep -q 'deepspeed=deepspeed_config,' "third_party/scalable_fp/finetune_multigpu.py"; then
    echo "[setup_third_party] applying scalable_fp_disable_deepspeed_paged_adamw8bit.patch"
    git -C third_party/scalable_fp apply "$ROOT/third_party/patches/scalable_fp_disable_deepspeed_paged_adamw8bit.patch"
fi

# Bug found running the actual F2 finetune step to completion (post-training
# cleanup): finetune_multigpu.py calls
# `trainer.accelerator.unwrap_model(tokenizer)` - unwrapping a tokenizer makes
# no sense (it is never wrapped/parallelized) and a newer `accelerate`'s
# `unwrap_model` -> `has_compiled_regions` does `module._modules`, which
# `PreTrainedTokenizer.__getattr__` raises AttributeError for instead of
# silently returning None as older versions effectively tolerated. This
# crashed AFTER training/eval completed but BEFORE `model.save_pretrained(...)`
# ran, so the final checkpoint was never written. Dropping the no-op line
# fixes it; the tokenizer variable is already the plain (unwrapped) object.
if [ -d "third_party/scalable_fp/.git" ] && grep -q 'tokenizer = trainer.accelerator.unwrap_model(tokenizer)' "third_party/scalable_fp/finetune_multigpu.py"; then
    echo "[setup_third_party] applying scalable_fp_no_tokenizer_unwrap.patch"
    git -C third_party/scalable_fp apply "$ROOT/third_party/patches/scalable_fp_no_tokenizer_unwrap.patch"
fi

# Real data edge case found running F3 Perinucleus's fingerprint verification:
# IndexError: index 0 is out of bounds for dimension 0 with size 0. 4 of the
# 64 perinucleus-generated fingerprints (measured: indices 4, 8, 12, 55) have
# a response that nucleus-samples to just the EOS token - after
# remove_eos_token_from_response strips it, the target is a genuinely empty
# string, which tokenizer(...)['input_ids'] turns into a 0-length tensor.
# check_fingerprints.py's eval_backdoor_acc unconditionally indexes
# signature_tokenized[0] to check for a leading BOS token, crashing instead of
# treating an unverifiable (empty-target) fingerprint as a miss. Minimal
# one-line guard: only check for a leading BOS token when there IS a token.
# An empty target can never match a real generation, so this correctly scores
# those 4 fingerprints as failures rather than crashing the whole evaluation.
if [ -d "third_party/scalable_fp/.git" ] && grep -q 'if signature_tokenized\[0\] == tokenizer.bos_token_id:' "third_party/scalable_fp/check_fingerprints.py"; then
    echo "[setup_third_party] applying scalable_fp_empty_signature_guard.patch"
    git -C third_party/scalable_fp apply "$ROOT/third_party/patches/scalable_fp_empty_signature_guard.patch"
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
