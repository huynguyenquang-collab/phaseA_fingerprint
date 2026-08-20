"""Generic Phase-A runner: source eval -> gate -> RTN3/RTN4/AWQ3/AWQ4/GPTQ3 ->
eval each -> unified per-family CSV. One script, three families
(english_random / perinucleus / ctcc), per student plan section 15/16.

Native evaluator is pluggable per family (plan section 16: "Không ép mọi
family dùng cùng một native verifier"); WikiText-2 PPL and the quantize/
reload backend are shared and reused verbatim from if_awq_tier0.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.common import (
    configure_gpu_performance,
    ensure_dir,
    free_model,
    gpu_peak_memory_bytes,
    load_causal_lm,
    load_yaml_config,
    write_json,
    write_jsonl,
)
from src.eval_wikitext_shim import compute_wikitext2_ppl
from src.quant_backend import PHASE_A_MATRIX, quantize_and_reload


def _native_evaluate(family: str, model, tokenizer, fixture, cfg: dict) -> dict:
    if family in ("english_random", "perinucleus"):
        from src.scalable_fp_eval import evaluate_native

        return evaluate_native(model, tokenizer, fixture, use_chat_template=cfg["fingerprint"].get(
            "use_chat_template", False
        ))
    if family == "ctcc":
        from src.ctcc_eval import evaluate_native

        return evaluate_native(
            model,
            tokenizer,
            fixture,
            max_new_tokens=cfg["generation"]["max_new_tokens"],
            batch_size=cfg["generation"].get("eval_batch_size", 16),
        )
    raise ValueError(f"unknown family {family}")


def _build_fixture(family: str, tokenizer, cfg: dict):
    if family in ("english_random", "perinucleus"):
        from src.scalable_fp_eval import build_eval_dataset

        fp_cfg = cfg["fingerprint"]
        return build_eval_dataset(
            tokenizer,
            num_fingerprints=fp_cfg["num_fingerprints"],
            max_key_length=fp_cfg["max_key_length"],
            max_response_length=fp_cfg["max_response_length"],
            fingerprint_generation_strategy=fp_cfg["fingerprint_generation_strategy"],
            fingerprints_file_path=fp_cfg["fingerprints_file_path"],
            seed=fp_cfg.get("seed", 42),
        )
    if family == "ctcc":
        from src.ctcc_eval import load_ctcc_eval_set

        fp_cfg = cfg["fingerprint"]
        return load_ctcc_eval_set(
            max_per_subset=fp_cfg.get("max_eval_per_subset", 100),
            seed=fp_cfg.get("seed", 42),
        )
    raise ValueError(f"unknown family {family}")


def _gate_passed(family: str, native: dict, cfg: dict) -> tuple[bool, str]:
    gate = cfg["fingerprint"]["clean_gate"]
    if family in ("english_random", "perinucleus"):
        threshold = gate["min_top1_fingerprint_recall"]
        ok = native["top1_fingerprint_recall"] >= threshold
        return ok, f"top1_fingerprint_recall={native['top1_fingerprint_recall']:.4f} (need >= {threshold})"
    if family == "ctcc":
        min_trigger = gate["min_trigger_fsr"]
        max_negative = gate["max_negative_fsr"]
        ok = (native["trigger_fsr"] or 0.0) >= min_trigger and (native["negative_fsr"] or 1.0) <= max_negative
        return ok, (
            f"trigger_fsr={native['trigger_fsr']} (need >= {min_trigger}), "
            f"negative_fsr={native['negative_fsr']} (need <= {max_negative})"
        )
    raise ValueError(f"unknown family {family}")


def _calibration_texts(path: str) -> list[str]:
    import json

    return [json.loads(line)["raw_text"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _run_one(label: str, evaluate_fn, model, tokenizer, device: str, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    t0 = time.time()
    native = evaluate_fn(model, tokenizer)
    ppl = compute_wikitext2_ppl(model, tokenizer, device=device)
    elapsed = time.time() - t0
    row = {
        "label": label,
        "native": native,
        "wikitext2_ppl": ppl,
        "evaluation_time_seconds": elapsed,
        "peak_gpu_memory_bytes": gpu_peak_memory_bytes(),
    }
    per_key = native.pop("per_key", None)
    write_json(out_dir / f"{label}.summary.json", row)
    if per_key is not None:
        write_jsonl(out_dir / f"{label}.per_key.jsonl", per_key)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=["english_random", "perinucleus", "ctcc"])
    ap.add_argument("--model-path", required=True, help="FP16/BF16 fingerprinted checkpoint (HF id or local dir)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--calibration", required=True, help="if_awq_tier0's pileval_seed42_128x512.jsonl (or equivalent)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--allow-partial-baseline",
        action="store_true",
        help="Proceed to PTQ even if the clean-gate check fails (student plan section 6 hard gate override)",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of {rtn3,rtn4,awq3,awq4,gptq3} to run (default: all 5)",
    )
    args = ap.parse_args()

    configure_gpu_performance()
    cfg = load_yaml_config(args.config)
    out_dir = ensure_dir(Path(args.output))
    calibration_texts = _calibration_texts(args.calibration)

    settings = PHASE_A_MATRIX
    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        settings = [s for s in PHASE_A_MATRIX if s.id in wanted]

    results: list[dict] = []

    # 1) Source (FP16/BF16) evaluation — always first, always reported.
    model = load_causal_lm(args.model_path, device=args.device, dtype=args.dtype)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    fixture = _build_fixture(args.family, tokenizer, cfg)

    def _eval_source(m, t):
        return _native_evaluate(args.family, m, t, fixture, cfg)

    source_row = _run_one("source", _eval_source, model, tokenizer, args.device, out_dir)
    source_row["quant_method"] = "none"
    source_row["bits"] = 16
    source_row["group_size"] = None
    results.append(source_row)

    ok, detail = _gate_passed(args.family, source_row["native"], cfg)
    print(f"[run_family_quant_matrix] clean-gate: {detail} -> {'PASS' if ok else 'FAIL'}")
    if not ok and not args.allow_partial_baseline:
        write_json(out_dir / "results_partial_gate_failed.json", results)
        raise SystemExit(
            "Clean gate failed before any PTQ was run. Fix the fingerprint checkpoint "
            "(see student plan section 9: scale num_fingerprints 64 -> 256) or pass "
            "--allow-partial-baseline to proceed anyway."
        )

    free_model(model)

    # 2) RTN3 / RTN4 / AWQ3 / AWQ4 / GPTQ3, each: quantize -> save -> reload -> evaluate.
    fp_model_for_quant = load_causal_lm(args.model_path, device=args.device, dtype=args.dtype)
    fp_tokenizer_for_quant = AutoTokenizer.from_pretrained(args.model_path)

    for setting in settings:
        quant_dir = out_dir / "quantized_models" / setting.id
        print(f"[run_family_quant_matrix] quantizing {setting.id} ...")
        # Each setting needs its own fresh copy of the FP weights.
        model = load_causal_lm(args.model_path, device=args.device, dtype=args.dtype) if setting is not settings[0] else fp_model_for_quant
        tokenizer = AutoTokenizer.from_pretrained(args.model_path) if setting is not settings[0] else fp_tokenizer_for_quant

        reloaded, reloaded_tok, manifest = quantize_and_reload(
            model,
            tokenizer,
            setting,
            quant_dir,
            device=args.device,
            dtype=args.dtype,
            calibration_texts=calibration_texts,
            max_seq_len=cfg.get("calibration", {}).get("max_seq_len", 512),
        )

        def _eval_quant(m, t):
            return _native_evaluate(args.family, m, t, fixture, cfg)

        row = _run_one(setting.id, _eval_quant, reloaded, reloaded_tok, args.device, out_dir)
        row["quant_method"] = setting.method
        row["bits"] = setting.bits
        row["group_size"] = setting.group_size
        row["manifest"] = manifest
        results.append(row)
        free_model(reloaded)

    write_json(out_dir / "results_all.json", results)
    print(f"[run_family_quant_matrix] done. wrote {out_dir / 'results_all.json'}")


if __name__ == "__main__":
    main()
