"""Merge the CTCC LoRA adapter into a standalone checkpoint.

CTCC's own `python/get_merged_models.py` uses MergeKit to merge the
fingerprinted model with an *unrelated* same-architecture model (e.g.
WizardMath-7B) for a model-merging-attack study — a different research
question. What Phase A needs is the ordinary "base + its own LoRA adapter ->
one standalone checkpoint" step described in student plan section 6
("base + LoRA fingerprint -> merge LoRA vào base -> save standalone
checkpoint"). That's a standard `peft` operation, not part of the
fingerprinting method itself, so we do it directly with `peft` rather than
routing through CTCC's cross-model MergeKit script.
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    merged = PeftModel.from_pretrained(base, args.adapter_path, torch_dtype=dtype)
    merged = merged.merge_and_unload()

    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"[merge_ctcc_lora] merged checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
