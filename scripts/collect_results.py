"""Merge per-family results_all.json (from run_family_quant_matrix.py) into
the single consolidated CSV shape from student plan sections 16/20, ready to
concatenate with the existing IF-SFT (F1) row(s).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FAMILY_LABEL = {
    "english_random": "ENGLISH-RANDOM",
    "perinucleus": "Perinucleus",
    "ctcc": "CTCC",
}

FIELDS = [
    "fingerprint_method",
    "base_model",
    "quant_method",
    "bits",
    "group_size",
    "native_metric_name",
    "native_primary_score",
    "native_secondary",
    "baseline_primary_score",
    "retention",
    "wikitext2_ppl",
    "baseline_wikitext2_ppl",
    "delta_ppl_abs",
    "delta_ppl_pct",
    "run_label",
]


def _primary_score(native: dict) -> float | None:
    if "top1_fingerprint_recall" in native:
        return native["top1_fingerprint_recall"]
    if "trigger_fsr" in native:
        return native["trigger_fsr"]
    if "fsr_exact" in native:
        return native["fsr_exact"]
    return None


def _rows_for_family(family: str, results_all_path: Path, base_model: str) -> list[dict]:
    rows_in = json.loads(results_all_path.read_text(encoding="utf-8"))
    baseline = rows_in[0]
    baseline_score = _primary_score(baseline["native"])
    baseline_ppl = baseline["wikitext2_ppl"].get("perplexity") or baseline["wikitext2_ppl"].get("ppl")

    out = []
    for r in rows_in:
        score = _primary_score(r["native"])
        ppl = r["wikitext2_ppl"].get("perplexity") or r["wikitext2_ppl"].get("ppl")
        secondary = {k: v for k, v in r["native"].items() if k not in ("native_metric_name",)}
        out.append(
            {
                "fingerprint_method": FAMILY_LABEL.get(family, family),
                "base_model": base_model,
                "quant_method": r["quant_method"],
                "bits": r["bits"],
                "group_size": r["group_size"],
                "native_metric_name": r["native"].get("native_metric_name"),
                "native_primary_score": score,
                "native_secondary": json.dumps(secondary, ensure_ascii=False),
                "baseline_primary_score": baseline_score,
                "retention": (score / baseline_score) if (score is not None and baseline_score) else None,
                "wikitext2_ppl": ppl,
                "baseline_wikitext2_ppl": baseline_ppl,
                "delta_ppl_abs": (ppl - baseline_ppl) if (ppl is not None and baseline_ppl is not None) else None,
                "delta_ppl_pct": (
                    (ppl / baseline_ppl - 1) * 100 if (ppl is not None and baseline_ppl) else None
                ),
                "run_label": r["label"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True, help="Dir containing <family>/results_all.json subdirs")
    ap.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.results_root)
    all_rows: list[dict] = []
    for family in ("english_random", "perinucleus", "ctcc"):
        p = root / family / "results_all.json"
        if p.exists():
            all_rows.extend(_rows_for_family(family, p, args.base_model))
        else:
            print(f"[collect_results] skipping {family}: {p} not found")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[collect_results] wrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
