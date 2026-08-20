#!/usr/bin/env bash
# F2 ENGLISH-RANDOM end to end: generate keys -> finetune -> verify -> escalate
# to 256 fingerprints once if the clean gate fails. Uses the upstream repo's
# own CLI verbatim (SewoongLab/scalable-fingerprinting-of-llms) - no
# reimplementation of fingerprint generation/injection/scoring.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_gpu_env.sh"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-2-7b-hf}"
NUM_FINGERPRINTS="${NUM_FINGERPRINTS:-64}"
FALLBACK_NUM_FINGERPRINTS="${FALLBACK_NUM_FINGERPRINTS:-256}"
KEY_LENGTH=16
RESPONSE_LENGTH=1
SEED=42
STRATEGY=english_random_responses

cd "$ROOT/third_party/scalable_fp"

run_round () {
    local n="$1"
    local fp_file="generated_data/output_fingerprints-english_random-llama2-7b-hf-n${n}.json"

    echo "[english_random] round n=$n: generating fingerprint data"
    python generate_finetuning_data.py \
        --key_response_strategy independent \
        --key_length "$KEY_LENGTH" \
        --response_length "$RESPONSE_LENGTH" \
        --num_fingerprints "$n" \
        --model_used_for_key_generation "$MODEL_PATH" \
        --first_token_strategy word \
        --seed "$SEED" \
        --output_file_path "$fp_file"

    echo "[english_random] round n=$n: finetuning (deepspeed, 1 GPU, ZeRO-2 + CPU offload, bf16)"
    deepspeed --num_gpus 1 --master_port 29501 finetune_multigpu.py \
        --model_path "$MODEL_PATH" \
        --num_fingerprints "$n" \
        --max_key_length "$KEY_LENGTH" \
        --max_response_length "$RESPONSE_LENGTH" \
        --fingerprint_generation_strategy "$STRATEGY" \
        --fingerprints_file_path "$(pwd)/$fp_file" \
        --num_train_epochs 30 \
        --learning_rate 5e-5 \
        --batch_size 8 \
        --deepspeed_stage 2 \
        --seed "$SEED" \
        --result_path "$(pwd)/results/"

    HASH=$(tail -n 1 current_config_hash.txt)
    MODEL_OUT="$(pwd)/results/saved_models/$HASH/final_model"
    echo "[english_random] round n=$n: model at $MODEL_OUT"

    echo "[english_random] round n=$n: verifying fingerprints (upstream check_fingerprints.py)"
    python check_fingerprints.py \
        --model_path "$MODEL_OUT" \
        --fingerprints_file_path "$(pwd)/$fp_file" \
        --fingerprint_generation_strategy "$STRATEGY" \
        --num_fingerprints "$n" \
        --max_key_length "$KEY_LENGTH" \
        --max_response_length "$RESPONSE_LENGTH" \
        --verbose_eval | tee "$ROOT/results/english_random_check_n${n}.log"

    echo "$MODEL_OUT" > "$ROOT/results/english_random_model_path.txt"
    echo "$(pwd)/$fp_file" > "$ROOT/results/english_random_fingerprints_file.txt"
}

mkdir -p "$ROOT/results"
run_round "$NUM_FINGERPRINTS"

FSR=$(grep -oE "Fingerprint accuracy: [0-9.]+" "$ROOT/results/english_random_check_n${NUM_FINGERPRINTS}.log" | tail -1 | awk '{print $3}')
echo "[english_random] FSR at n=$NUM_FINGERPRINTS: ${FSR:-unknown}"

if [ -n "${FSR:-}" ] && (( $(echo "$FSR < 95.0" | bc -l) )); then
    echo "[english_random] clean gate FAILED (<95%) — escalating to n=$FALLBACK_NUM_FINGERPRINTS per student plan section 9"
    run_round "$FALLBACK_NUM_FINGERPRINTS"
fi

echo "[english_random] done. Model path recorded in results/english_random_model_path.txt"
