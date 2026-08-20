"""F2 (ENGLISH-RANDOM) / F3 (Perinucleus) native evaluator.

Both families share the upstream repo SewoongLab/scalable-fingerprinting-of-llms.
Its own `check_fingerprints.py` only prints results to stdout (verified by
reading the checkout: no JSON output path exists), so instead of parsing
stdout we import its lower-level, reusable functions directly —
`get_fingerprint_ds` (builds the eval dataset) and `eval_backdoor_acc` (the
paper's own top-1 recall / fractional-accuracy scorer) — and call them
ourselves against whichever model+tokenizer we already have loaded (FP16,
RTN, AWQ, GPTQ). This is reuse of the official scoring code, not a
reimplementation of it.

The fingerprints file is generated ONCE (generate_finetuning_data.py) and
reused verbatim for every quantized variant — never regenerated post-quant
(student plan section 21 rule #4).
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from src.common import ensure_third_party_on_path, third_party_root


def _import_upstream():
    ensure_third_party_on_path("scalable_fp")
    from fingerprint_dataloader import get_fingerprint_ds  # type: ignore
    from check_fingerprints import eval_backdoor_acc  # type: ignore

    return get_fingerprint_ds, eval_backdoor_acc


@contextmanager
def _cwd_at_scalable_fp():
    """fingerprint_dataloader.py's english_random_responses branch loads its
    random-word signature pool via a path built from `os.getcwd()`, not
    relative to the module or an argument - it assumes callers always run
    from within third_party/scalable_fp/ (true for run_english_random.sh /
    run_perinucleus.sh, false for run_family_quant_matrix.py, which runs from
    the repo root). Chdir there only for the duration of the upstream call.
    """
    prev = os.getcwd()
    os.chdir(third_party_root() / "scalable_fp")
    try:
        yield
    finally:
        os.chdir(prev)


def build_eval_dataset(
    tokenizer,
    *,
    num_fingerprints: int,
    max_key_length: int,
    max_response_length: int,
    fingerprint_generation_strategy: str,
    fingerprints_file_path: str,
    seed: int = 42,
):
    """Build the fixed eval set once; pass the SAME object into evaluate_native
    for every quantized variant of this family's checkpoint."""
    get_fingerprint_ds, _ = _import_upstream()
    # Resolve to absolute *before* the chdir below - cache_path is opened
    # as-is by get_fingerprint_ds, so a relative path (e.g. the configs/*.yaml
    # default, relative to the repo root) would otherwise resolve against the
    # wrong base once cwd changes.
    fingerprints_file_path = os.path.abspath(fingerprints_file_path)
    with _cwd_at_scalable_fp():
        dataset, _ = get_fingerprint_ds(
            tokenizer,
            num_fingerprints=num_fingerprints,
            key_length=max_key_length,
            response_length=max_response_length,
            deterministic_length=True,
            strategy=fingerprint_generation_strategy,
            cache_path=fingerprints_file_path,
            remove_eos_token_from_response=True,
            num_responses_per_fingerprint=1,
            get_eval_set=True,
            seed=seed,
        )
    return dataset["train"]


def evaluate_native(model, tokenizer, eval_dataset, *, use_chat_template: bool = False) -> dict:
    """Top-1 Fingerprint Recall (primary) + fractional accuracy (secondary),
    exactly as check_fingerprints.py's eval_backdoor_acc computes them.
    """
    _, eval_backdoor_acc = _import_upstream()
    accuracy, fractional_accuracy = eval_backdoor_acc(
        model, tokenizer, eval_dataset, verbose=False, use_chat_template=use_chat_template
    )
    return {
        "native_metric_name": "top1_fingerprint_recall",
        "top1_fingerprint_recall": float(accuracy[0]) / 100.0,
        "fractional_accuracy": float(fractional_accuracy[0]) / 100.0,
        "num_fingerprints": len(eval_dataset),
    }
