"""F4 (CTCC) native evaluator: Trigger FSR (positive) + Negative FSR (negative).

CTCC (Xuzhenhua55/CTCC) ships only *data* (trigger_set.json, suppression_set.json,
normal_set.json, test_set.json) and documents training/inference entirely
through an external framework (LLaMA-Factory's `llamafactory-cli`, template
`llama2`) — there is no upstream Python function to import for scoring
(verified: the repo's `python/` directory has merge/PPL/pruning helpers, no
fingerprint-detection script). Re-running every quantized variant (including
the packed AWQ/GPTQ checkpoints, which use custom nn.Module subclasses that
plain LLaMA-Factory cannot load) through `llamafactory-cli --do_predict`
would silently break the "identical generation protocol across variants"
requirement (student plan section 21 rule #7).

Instead we reproduce LLaMA-Factory's own `llama2` chat template verbatim in
plain `transformers`, verified against the upstream source
(hiyouga/LLaMA-Factory src/llamafactory/data/template.py, `Llama2Template`
class + `register_template(name="llama2", ...)`):

    for each turn i:
        if role == user:  "<bos>[INST] {content} [/INST]"   (bos every turn)
        if role == assistant: "{content}<eos>"
    (no <<SYS>> block: CTCC's JSON rows carry no system field)

This is templating only, not a reimplementation of the CTCC fingerprinting
method itself — the trigger/suppression/normal data and target strings are
used exactly as published upstream.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from src.common import ensure_third_party_on_path

TRIGGER_MARKER = "IAMALIVE"  # CTCC's own fixed fingerprint output string (dataset-verified)


@dataclass
class CTCCExample:
    subset: str  # trigger_seen | trigger_unseen | suppression | normal
    instruction: str
    output: str
    history: list[list[str]]


def _load_subset(path: Path, subset: str, n: int | None, seed: int) -> list[CTCCExample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return [
        CTCCExample(
            subset=subset,
            instruction=r["instruction"],
            output=r["output"],
            history=r.get("history", []),
        )
        for r in rows
    ]


def load_ctcc_eval_set(
    *, max_per_subset: int | None = 100, seed: int = 42
) -> list[CTCCExample]:
    """Fixed eval set: trigger_set.json (seen, trained-on) + test_set.json
    (unseen, held out) as positives, suppression_set.json + normal_set.json
    as negatives. Build ONCE and reuse across FP16/RTN/AWQ/GPTQ (never
    regenerate — student plan section 21 rule #4 applies to CTCC's set too).
    """
    ctcc_root = ensure_third_party_on_path("ctcc")
    dataset_dir = ctcc_root / "dataset"
    examples: list[CTCCExample] = []
    examples += _load_subset(dataset_dir / "trigger_set.json", "trigger_seen", max_per_subset, seed)
    examples += _load_subset(dataset_dir / "test_set.json", "trigger_unseen", max_per_subset, seed)
    examples += _load_subset(dataset_dir / "suppression_set.json", "suppression", max_per_subset, seed)
    examples += _load_subset(dataset_dir / "normal_set.json", "normal", max_per_subset, seed)
    return examples


def build_llama2_prompt(example: CTCCExample) -> str:
    """Reproduces LLaMA-Factory's `llama2` template (Llama2Template._encode):
    every user turn gets its own <s>, every assistant turn is closed with </s>,
    no <<SYS>> block since CTCC rows carry no system message.
    """
    prompt = ""
    for user_turn, assistant_turn in example.history:
        prompt += f"<s>[INST] {user_turn} [/INST]{assistant_turn}</s>"
    prompt += f"<s>[INST] {example.instruction} [/INST]"
    return prompt


def _generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


def evaluate_native(
    model,
    tokenizer,
    examples: list[CTCCExample],
    *,
    max_new_tokens: int = 32,
    batch_size: int = 16,
) -> dict:
    prompts = [build_llama2_prompt(e) for e in examples]
    generations = _generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size)

    per_key = []
    for example, prompt, generated in zip(examples, prompts, generations):
        triggered = TRIGGER_MARKER in generated
        per_key.append(
            {
                "subset": example.subset,
                "prompt": prompt,
                "target": example.output,
                "generated": generated,
                "triggered": triggered,
            }
        )

    trigger_rows = [r for r in per_key if r["subset"] in ("trigger_seen", "trigger_unseen")]
    negative_rows = [r for r in per_key if r["subset"] in ("suppression", "normal")]

    def _rate(rows, key="triggered"):
        return (sum(1 for r in rows if r[key]) / len(rows)) if rows else None

    seen = [r for r in per_key if r["subset"] == "trigger_seen"]
    unseen = [r for r in per_key if r["subset"] == "trigger_unseen"]

    return {
        "native_metric_name": "trigger_fsr+negative_fsr",
        "trigger_fsr": _rate(trigger_rows),
        "trigger_fsr_seen": _rate(seen),
        "trigger_fsr_unseen": _rate(unseen),
        "negative_fsr": _rate(negative_rows),
        "num_trigger": len(trigger_rows),
        "num_negative": len(negative_rows),
        "per_key": per_key,
    }
