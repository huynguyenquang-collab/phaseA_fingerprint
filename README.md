# Phase A — Multi-Fingerprint PTQ Characterization (F2/F3/F4)

Implements Phase A of `TIER0_MULTI_FINGERPRINT_PTQ_STUDENT_PLAN.md`: extend the
already-completed IF-SFT (F1) PTQ characterization to three more injected
ownership-fingerprint families, then run the same fixed 5-setting PTQ matrix.

| Family | Representative | Upstream repo |
|---|---|---|
| F2 | ENGLISH-RANDOM | `SewoongLab/scalable-fingerprinting-of-llms` |
| F3 | Perinucleus | `SewoongLab/scalable-fingerprinting-of-llms` (same repo) |
| F4 | CTCC | `Xuzhenhua55/CTCC` + `hiyouga/LLaMA-Factory` (training engine) |

F1 (IF-SFT) is untouched and lives entirely in the separate `if_awq_tier0`
checkout. This project **reuses** that repo's AWQ/GPTQ/RTN quantization
backends and WikiText-2 evaluator (via `IF_AWQ_TIER0_ROOT`, see below) —
nothing quantization-related is reimplemented here.

## Hard requirement: point at an existing if_awq_tier0 checkout

```bash
export IF_AWQ_TIER0_ROOT=/path/to/if_awq_tier0
```

Everything in `src/quant_backend.py` and `src/eval_wikitext_shim.py` imports
`if_awq_tier0`'s own `src.quantization.*` / `src.eval_wikitext` modules from
that path — no vendored copy exists in this repo.

## What was verified before writing this code

Both upstream repos were shallow-cloned and read directly (not taken on
faith from the student plan doc, which turned out to have some inaccurate
details):

- `SewoongLab/scalable-fingerprinting-of-llms`: confirmed
  `generate_finetuning_data.py`, `finetune_multigpu.py`, `check_fingerprints.py`
  exist with the exact flags used in `scripts/run_english_random.sh` /
  `run_perinucleus.sh`. Corrected from the plan doc: `generate_finetuning_data.py`
  has no `english_random_responses` *generation* strategy — that flag only
  exists on `finetune_multigpu.py`/`check_fingerprints.py`. Key/response text
  for ENGLISH-RANDOM is generated with `--key_response_strategy independent`.
- `Xuzhenhua55/CTCC`: confirmed it ships **only data** (`trigger_set.json`,
  `suppression_set.json`, `normal_set.json`, `test_set.json`) and depends
  entirely on external LLaMA-Factory for training/inference — it has **no**
  fingerprint-detection script of its own (the plan doc implied one exists).
  `src/ctcc_eval.py` therefore hand-reproduces LLaMA-Factory's `llama2` chat
  template (verified against `hiyouga/LLaMA-Factory`'s
  `src/llamafactory/data/template.py`, `Llama2Template`/`register_template`)
  so Trigger FSR / Negative FSR can be computed identically across FP16 and
  every quantized variant, including the packed AWQ/GPTQ checkpoints that
  plain LLaMA-Factory cannot load (custom `nn.Module` subclasses).
- CTCC's own `python/get_merged_models.py` merges the fingerprinted model
  with an *unrelated* model via MergeKit (a model-merging-attack study, not
  what we need). Phase A instead uses a plain `peft`
  `merge_and_unload()` step (`scripts/merge_ctcc_lora.py`) for "base + its
  own LoRA adapter -> one standalone checkpoint".

## Layout

```
multi_fp_tier0/
├── third_party/            # scalable_fp, ctcc, llama_factory (cloned, pinned, untouched)
├── src/
│   ├── common.py           # IF_AWQ_TIER0_ROOT path shim, GPU perf setup, generic HF load/free
│   ├── quant_backend.py    # RTN3/RTN4/AWQ3/AWQ4/GPTQ3 — all delegate to if_awq_tier0
│   ├── scalable_fp_eval.py # F2/F3 native evaluator (imports upstream eval_backdoor_acc)
│   ├── ctcc_eval.py        # F4 native evaluator (Trigger FSR + Negative FSR)
│   ├── eval_wikitext_shim.py
│   └── run_family_quant_matrix.py  # generic: source eval -> gate -> 5x PTQ -> eval each
├── configs/{english_random,perinucleus,ctcc}.yaml
├── scripts/
│   ├── setup_third_party.sh / setup_env.sh
│   ├── run_english_random.sh / run_perinucleus.sh   # upstream CLI, deepspeed 1-GPU
│   ├── run_ctcc.sh + register_ctcc_datasets.py + merge_ctcc_lora.py
│   ├── run_phase_a_end_to_end.sh   # master: all 3 families -> unified CSV
│   └── collect_results.py
└── tests/                  # no-GPU: llama2 template, Q1-Q5 matrix, config schema
```

## Running it

```bash
export IF_AWQ_TIER0_ROOT=/path/to/if_awq_tier0
bash scripts/setup_env.sh            # venv + deps + environment.txt/nvidia_smi.txt
source .venv/bin/activate
bash scripts/run_phase_a_end_to_end.sh
```

Or one family at a time (useful for the smoke-test-first workflow):

```bash
bash scripts/setup_third_party.sh
bash scripts/run_english_random.sh   # injects + verifies, writes results/english_random_model_path.txt
python -m src.run_family_quant_matrix \
    --family english_random \
    --model-path "$(cat results/english_random_model_path.txt)" \
    --config configs/english_random.yaml \
    --calibration "$IF_AWQ_TIER0_ROOT/artifacts/calibration/pileval_seed42_128x512.jsonl" \
    --output results/english_random
```

## GPU optimization for a single A100 40GB (student plan hardware target)

- **bf16 everywhere** (fine-tuning, LoRA SFT, quantization, PPL eval) — no
  loss-scaling overhead, full A100 tensor-core throughput.
- **TF32 matmuls + cuDNN benchmark** enabled once via
  `src.common.configure_gpu_performance()`.
- **flash-attention 2** when installed, else `sdpa` (`best_attn_implementation()`)
  — never falls back to eager attention.
- **F2/F3 fine-tuning: no DeepSpeed, `paged_adamw_8bit` instead** (measured
  live, see "Real bugs found running this on an A100" below — an earlier
  "full-batch + DeepSpeed ZeRO-2 CPU offload" design was tried first and
  actually OOM'd; this is what replaced it). `batch_size=8` (upstream's own
  default) already measures a peak of ~40.4GB / 40GB GPU memory, so GPU memory
  — not gradient-accumulation overhead — is the binding constraint here; do
  not raise `--batch_size` in `run_english_random.sh`/`run_perinucleus.sh`
  without re-measuring peak GPU memory first.
- **CTCC: same effective batch, bigger micro-batch**: `per_device_train_batch_size`
  2→4 with `gradient_accumulation_steps` 8→4 keeps the effective batch at 16
  (identical to the paper's recipe) while roughly halving the number of
  optimizer steps' worth of Python/NCCL overhead.
- **CTCC: `--enable_liger_kernel True`** — fused RMSNorm/RoPE/cross-entropy
  kernels, confirmed present in the pinned `hiyouga/LLaMA-Factory` checkout's
  `ModelArguments` (`model_args.py`); drops training memory/time with no
  numerical-recipe change. If a different pinned commit lacks it, the flag
  fails fast at startup — just delete that one line.
- **Larger AWQ `layer_batch_size`** (`AWQ_LAYER_BATCH_SIZE = 32` in
  `src/quant_backend.py`, vs. if_awq_tier0's own tuned default of 16, which
  was picked without a dedicated GPU to profile against) — more transformer
  layers calibrated per pass on a dedicated A100. Purely a throughput knob:
  bits/group_size/n_grid/max_tokens_per_sample are untouched, so the AWQ
  search/rounding result is unaffected.
- **Batched, left-padded generation** for the CTCC evaluator
  (`_generate_batch` in `src/ctcc_eval.py`) instead of looping one prompt at
  a time.
- **Explicit `free_model()`** (`del` + `gc.collect()` + `torch.cuda.empty_cache()`)
  between every quantization setting — never holds two 7B models resident.

## Disk discipline + sizing

Disk, not GPU memory, was the actual blocker on a prior rented single-GPU box
for this project's earlier IF-SFT/UTF work (hit a ~14GB-free wall). Two
independent cleanup layers now run by default:

1. `src/run_family_quant_matrix.py` deletes **each setting's own quantized
   checkpoint right after it's evaluated** (pass `--keep-quantized-checkpoints`
   to disable) — this matters because RTN's "fake-quant" checkpoint is a
   full-size, uncompressed bf16 checkpoint (dequantized back to `nn.Linear`),
   so leaving all 5 quantized checkpoints (2× RTN + AWQ3 + AWQ4 + GPTQ3) on
   disk at once is the single biggest disk risk in this pipeline.
2. `run_phase_a_end_to_end.sh` deletes each family's **FP fingerprinted
   checkpoint** once that family's `results_all.json` is safely written
   (`DELETE_FP_CHECKPOINT_AFTER_FAMILY=false` to keep them).

With both layers on, at most one family's FP checkpoint (~13.5GB bf16) + one
quantized checkpoint (≤~13.5GB for RTN, ~3–4GB for packed AWQ/GPTQ) are ever
resident at the same time.

**Recommended free disk before starting Phase A: ≥100GB**, broken down as:

| Item | Approx. size |
|---|---:|
| `meta-llama/Llama-2-7b-hf` HF cache (shared across all 3 families) | ~14 GB |
| venv + third_party repos (scalable_fp, ctcc, llama_factory) + DeepSpeed/peft | ~12 GB |
| One family's FP checkpoint in flight | ~14 GB |
| One quantized checkpoint in flight (worst case: RTN, uncompressed) | ~14 GB |
| Per-key JSONL / summary JSON / logs across all 3 families | <2 GB |
| Safety margin (HF Hub retries, DeepSpeed ZeRO temp shards, tmp files) | ~20–30 GB |

That gives ~75GB of hard requirements plus real margin. **≥150GB is the
comfortable target** if you'd rather not babysit disk during the run, or if
you pass `--keep-quantized-checkpoints` / `DELETE_FP_CHECKPOINT_AFTER_FAMILY=false`
to keep artifacts for later inspection (each kept family then costs its full
~40GB: FP + RTN3 + RTN4 + AWQ3 + AWQ4 + GPTQ3, times 3 families ≈ 120GB on
top of the base ~26GB).

## Known gaps / deviations to double-check on the real A100 box

1. `generate_finetuning_data.py --model_used_for_key_generation` is pointed
   at `meta-llama/Llama-2-7b-hf` (the fingerprint target itself) instead of
   the repo's own default (`Meta-Llama-3.1-8B-Instruct`) — that flag only
   generates filler English key/response text, unrelated to the fingerprint
   algorithm being measured, so this avoids a second 8B download. Flag if
   this changes anything material.
2. `scripts/run_english_random.sh` / `run_perinucleus.sh`'s 95% gate check
   greps `check_fingerprints.py`'s stdout ("Fingerprint accuracy: NN.NN")
   with `bc` — good enough for a bash gate, but `run_family_quant_matrix.py`'s
   own `_gate_passed()` is the authoritative check (it re-evaluates via the
   same `eval_backdoor_acc` call used for every quantized variant).
3. Nothing in this repo had been executed against real GPU weights until it
   was actually run on a rented A100 (see below) — `py_compile`/`tests/`
   passing only proves the no-GPU code paths.

## Real bugs found running this on an A100 (all patched, documented in `third_party/patches/`)

Running the actual pipeline against real weights surfaced issues no amount of
reading the code would have caught. In the order hit:

1. **`generate_finetuning_data.py`'s `max_length=` vs newer transformers.**
   `ValueError: Input length of input_ids is N, but max_length is set to N`.
   Fixed: `scalable_fp_max_new_tokens.patch` (`max_length=N` → `max_new_tokens=N`
   at 4 call sites — the library's own suggested fix).
2. **A stale fingerprints file hangs the pipeline forever.**
   `generate_finetuning_data.py` prompts `y/n` to overwrite an existing output
   file, with no stdin attached in a background/tmux run. Fixed: `rm -f` the
   target file before regenerating (deterministic given the same seed/args) —
   see `run_english_random.sh`/`run_perinucleus.sh`.
3. **`finetune_multigpu.py` passes a removed `evaluation_strategy` alias.**
   `TypeError: TrainingArguments.__init__() got an unexpected keyword argument
   'evaluation_strategy'` (the file already sets the current `eval_strategy=`
   a few lines above — a leftover duplicate). Fixed:
   `scalable_fp_drop_duplicate_evaluation_strategy.patch`.
4. **A required resource file is never generated by our own commands.**
   `FileNotFoundError: generated_data/random-words-key-128-sig-128-key_sig-independent.json`.
   The `english_random_responses` strategy needs this separate random-word
   pool (built via `--random_word_generation`), which our `--key_response_strategy
   independent` call never produces. Fixed: generate it once in
   `run_english_random.sh` before the round loop.
5. **`report_to=None` is rejected by newer transformers.**
   `ValueError: None is not supported, only azure_ml, comet_ml, ... wandb, ...
   are supported` — now requires the literal string `"none"`. Fixed:
   `scalable_fp_report_to_none_string.patch`.
6. **Kernel OOM killer (exit code -9, no Python traceback) on the real finetune
   step.** Root cause, isolated by measuring `free -m` every 2s across several
   repro runs: DeepSpeed's ZeRO-Offload **always substitutes its own fp32 CPU
   Adam** once `offload_optimizer` is configured, ignoring `TrainingArguments`'
   `optim=` entirely — so trying `optim="adamw_bnb_8bit"` alone (measured) had
   zero effect, peak RAM was identical (~123–126GB of this box's ~125GB/
   ~120GB-cgroup budget) whether `offload_param` was present or not, and
   whether `--batch_size` was 8 or 64. With `world_size=1` (single GPU), ZeRO
   partitions nothing across ranks anyway — pure overhead here. Fixed:
   `scalable_fp_disable_deepspeed_paged_adamw8bit.patch` — `deepspeed=None` +
   `optim="paged_adamw_8bit"` (runs on GPU, pages to host RAM only under
   pressure). Measured peak after the fix: ~27GB host RAM, ~40.4GB / 40GB GPU
   memory — tight, but it completed a real training step successfully.
   **Do not raise `--batch_size` above 8 without re-measuring peak GPU memory.**
7. **Post-training cleanup crashes before the checkpoint is saved.**
   `trainer.accelerator.unwrap_model(tokenizer)` — unwrapping a tokenizer is
   meaningless (never wrapped/parallelized); a newer `accelerate`'s
   `unwrap_model` → `has_compiled_regions` does `module._modules`, which
   `PreTrainedTokenizer.__getattr__` raises `AttributeError` for. This ran
   *after* training/eval completed but *before* `model.save_pretrained(...)`,
   so the final checkpoint was never written. Fixed:
   `scalable_fp_no_tokenizer_unwrap.patch` (drop the no-op line).

All of the above are applied automatically and idempotently by
`scripts/setup_third_party.sh` (checks whether each patch is already applied
before re-applying) — nothing here requires manual intervention on a fresh
clone. F4 (CTCC/LLaMA-Factory) and the AWQ/GPTQ/RTN quant matrix have not yet
been run end to end on real weights; expect this same category of "one
outdated-library-API assumption at a time" issue there too.
