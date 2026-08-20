#!/usr/bin/env bash
# Master Phase A orchestrator: F2 -> F3 -> F4, each fingerprint -> gate ->
# RTN3/RTN4/AWQ3/AWQ4/GPTQ3 -> collect. Disk is scarce on a single-GPU rental
# box (a prior session on this project's IF-SFT/UTF work hit a 14GB-free wall)
# so each family's quantized checkpoints are deleted once its results.csv is
# safely written, keeping at most one family's quantized artifacts on disk
# at a time.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_gpu_env.sh"
cd "$ROOT"

: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the existing if_awq_tier0 checkout}"
CALIBRATION="${CALIBRATION:-$IF_AWQ_TIER0_ROOT/artifacts/calibration/pileval_seed42_128x512.jsonl}"
# Delete each family's FP fingerprinted checkpoint once its results_all.json is
# safely written (see README "Disk sizing"). Set to "false" to keep every raw
# checkpoint around for later manual inspection/deployment (needs much more disk).
DELETE_FP_CHECKPOINT_AFTER_FAMILY="${DELETE_FP_CHECKPOINT_AFTER_FAMILY:-true}"
# /workspace on a rented GPU box is often NOT a persistent volume (wiped on
# recycle/destroy). Push results/ (JSON/CSV only, no model weights) to a
# dedicated branch after each family finishes, so numbers survive even if the
# instance disappears mid-run. Set to "false" to disable.
PUSH_RESULTS_LIVE="${PUSH_RESULTS_LIVE:-true}"
RESULTS_BRANCH="${RESULTS_BRANCH:-results-live}"
RESULTS_WORKTREE="${RESULTS_WORKTREE:-$ROOT/../phaseA_results_branch}"

echo "== Phase A: setup =="
bash scripts/setup_third_party.sh

if [ "$PUSH_RESULTS_LIVE" = "true" ]; then
    if ! git show-ref --verify --quiet "refs/remotes/origin/$RESULTS_BRANCH"; then
        echo "== Phase A: creating orphan branch $RESULTS_BRANCH on origin =="
        git branch "$RESULTS_BRANCH" 2>/dev/null || true
        git push origin "HEAD:$RESULTS_BRANCH" 2>&1 || true
    fi
    if [ ! -d "$RESULTS_WORKTREE" ]; then
        git worktree add "$RESULTS_WORKTREE" "$RESULTS_BRANCH" 2>&1 || \
            git worktree add -b "$RESULTS_BRANCH" "$RESULTS_WORKTREE" "origin/$RESULTS_BRANCH"
    fi
fi

push_results_snapshot () {
    local family="$1"
    [ "$PUSH_RESULTS_LIVE" = "true" ] || return 0
    echo "== Phase A: pushing $family results snapshot to $RESULTS_BRANCH =="
    mkdir -p "$RESULTS_WORKTREE/results"
    rsync -a --exclude 'quantized_models' "$ROOT/results/" "$RESULTS_WORKTREE/results/"
    ( cd "$RESULTS_WORKTREE" \
      && git add results \
      && git commit -q -m "results snapshot: $family done ($(date -u +%Y-%m-%dT%H:%M:%SZ))" --allow-empty \
      && git push origin "$RESULTS_BRANCH" -q ) 2>&1 || echo "[push_results_snapshot] push failed, continuing anyway"
}

run_family () {
    local family="$1" model_path_file="$2" config="$3"
    local model_path
    model_path=$(cat "$model_path_file")

    echo "== Phase A: $family quant matrix (model=$model_path) =="
    # run_family_quant_matrix.py already deletes each setting's quantized
    # checkpoint right after evaluating it (default), so at most one
    # quantized checkpoint is ever on disk at once during this call.
    python -m src.run_family_quant_matrix \
        --family "$family" \
        --model-path "$model_path" \
        --config "$config" \
        --calibration "$CALIBRATION" \
        --output "results/$family" \
        --device cuda \
        --dtype bfloat16

    echo "== Phase A: $family — freeing any leftover quantized checkpoints (disk) =="
    du -sh "results/$family/quantized_models" 2>/dev/null || true
    rm -rf "results/$family/quantized_models"

    if [ "$DELETE_FP_CHECKPOINT_AFTER_FAMILY" = "true" ]; then
        echo "== Phase A: $family — freeing FP fingerprinted checkpoint at $model_path =="
        du -sh "$model_path" 2>/dev/null || true
        rm -rf "$model_path"
    fi
}

echo "== Phase A: F2 ENGLISH-RANDOM =="
bash scripts/run_english_random.sh
run_family english_random results/english_random_model_path.txt configs/english_random.yaml
push_results_snapshot english_random

echo "== Phase A: F3 Perinucleus =="
bash scripts/run_perinucleus.sh
run_family perinucleus results/perinucleus_model_path.txt configs/perinucleus.yaml
push_results_snapshot perinucleus

echo "== Phase A: F4 CTCC =="
bash scripts/run_ctcc.sh
run_family ctcc results/ctcc_model_path.txt configs/ctcc.yaml
push_results_snapshot ctcc

echo "== Phase A: collecting unified CSV =="
python scripts/collect_results.py \
    --results-root results \
    --base-model meta-llama/Llama-2-7b-hf \
    --output results/phase_a_fingerprint_quantization.csv
push_results_snapshot final_csv

echo "== Phase A done. results/phase_a_fingerprint_quantization.csv =="
column -s, -t results/phase_a_fingerprint_quantization.csv | head -20 || true
