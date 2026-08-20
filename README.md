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
- **DeepSpeed ZeRO-2 + CPU optimizer/param offload** for F2/F3 full fine-tuning
  (this is `finetune_multigpu.py`'s own hardcoded config, confirmed by
  reading it — a 7B full fine-tune's optimizer states do not fit in 40GB
  without offload; this is the "memory engineering permitted" knob from
  student plan section 9, not a hyperparameter change).
- **Same effective batch size as the CTCC paper** (bs=2 × grad_accum=8) is
  kept unchanged — only `--fp16` was swapped for `--bf16 True --tf32 True`
  (pure precision/throughput change, doesn't alter the optimization
  trajectory the paper's recipe was tuned for).
- **Batched, left-padded generation** for the CTCC evaluator
  (`_generate_batch` in `src/ctcc_eval.py`) instead of looping one prompt at
  a time.
- **Explicit `free_model()`** (`del` + `gc.collect()` + `torch.cuda.empty_cache()`)
  between every quantization setting — never holds two 7B models resident.
- **Disk discipline**: `run_phase_a_end_to_end.sh` deletes each family's
  `quantized_models/` directory right after its `results_all.json` is
  written, since a prior session on this same project's IF-SFT/UTF work hit a
  real disk wall (~14GB free) on a rented single-GPU box.

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
3. Nothing in this repo has been executed against real GPU weights (no GPU
   on this machine, same situation `if_awq_tier0`'s own README documents) —
   `py_compile` on every `.py` file and the no-GPU `tests/` pass, but
   `run_family_quant_matrix.py`, `quant_backend.py`, and the injection
   scripts have not been run end to end yet.
