"""Register CTCC's trigger/suppression/normal/test sets into LLaMA-Factory's
data/dataset_info.json, in the exact multi-turn column mapping the CTCC
README specifies. Copies the four JSON files into LLaMA-Factory's data/ dir
(symlinks would break if LLaMA-Factory resolves paths relative to its own
data dir) without touching any other upstream file.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ENTRIES = {
    "trigger_set": "trigger_set.json",
    "suppression_set": "suppression_set.json",
    "normal_set": "normal_set.json",
    "test_set": "test_set.json",
}

COLUMNS = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "history": "history",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctcc-root", required=True)
    ap.add_argument("--llama-factory-root", required=True)
    args = ap.parse_args()

    ctcc_dataset_dir = Path(args.ctcc_root) / "dataset"
    lf_data_dir = Path(args.llama_factory_root) / "data"
    lf_data_dir.mkdir(parents=True, exist_ok=True)

    dataset_info_path = lf_data_dir / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}

    for name, filename in ENTRIES.items():
        shutil.copy2(ctcc_dataset_dir / filename, lf_data_dir / filename)
        dataset_info[name] = {"file_name": filename, "columns": dict(COLUMNS)}

    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[register_ctcc_datasets] registered {list(ENTRIES)} into {dataset_info_path}")


if __name__ == "__main__":
    main()
