"""Thin wrapper around if_awq_tier0's own WikiText-2 PPL evaluator so every
family/quant setting reports utility with the exact same code path used by
the IF-SFT experiment (student plan section 5)."""
from __future__ import annotations

from src.common import ensure_if_awq_tier0_on_path


def compute_wikitext2_ppl(model, tokenizer, device: str = "cuda") -> dict:
    ensure_if_awq_tier0_on_path()
    from src.eval_wikitext import compute_wikitext2_ppl as _compute

    return _compute(model, tokenizer, device=device)
