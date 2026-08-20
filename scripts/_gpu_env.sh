#!/usr/bin/env bash
# Sourced by every run_*.sh script. Single A100 40GB, single-GPU tuning.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
