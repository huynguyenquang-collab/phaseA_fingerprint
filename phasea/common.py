"""Shared helpers: reuse if_awq_tier0's backends (no copy/reimplementation),
GPU performance setup for a single A100 40GB, and generic run bookkeeping.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def if_awq_tier0_root() -> Path:
    """Locate the existing if_awq_tier0 checkout (IF-SFT + vendored AWQ/GPTQ backends).

    Required so this project can import `src.quantization.*` and
    `src.eval_wikitext` from that repo verbatim instead of duplicating them
    (student plan section 4: "dùng đúng quantization backends đã chạy ổn
    trong experiment IF-SFT").
    """
    env = os.environ.get("IF_AWQ_TIER0_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        # Default: sibling directory next to this repo.
        root = (Path(__file__).resolve().parent.parent.parent / "if_awq_tier0").resolve()
    if not (root / "src" / "quantization").exists():
        raise RuntimeError(
            f"IF_AWQ_TIER0_ROOT={root} does not look like an if_awq_tier0 checkout "
            "(missing src/quantization). Set the IF_AWQ_TIER0_ROOT env var explicitly."
        )
    return root


def ensure_if_awq_tier0_on_path() -> Path:
    root = if_awq_tier0_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def third_party_root() -> Path:
    return Path(__file__).resolve().parent.parent / "third_party"


def ensure_third_party_on_path(name: str) -> Path:
    path = third_party_root() / name
    if not path.exists():
        raise RuntimeError(
            f"third_party/{name} missing — run scripts/setup_third_party.sh first"
        )
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return path


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# GPU performance setup for a single A100 40GB (student plan: "tối ưu GPU
# nhất có thể"). Called once at the top of every entrypoint that touches a
# model. Kept import-light so config-only code paths don't need torch.
# ---------------------------------------------------------------------------

def configure_gpu_performance() -> None:
    import torch

    # A100 supports TF32 tensor cores; this is free throughput for the fp32
    # matmuls PyTorch/DeepSpeed still do internally (e.g. optimizer math),
    # with no accuracy impact on the bf16 forward/backward path we use.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def best_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_causal_lm(model_id_or_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    """Single shared model-loading path so FP16/RTN/AWQ/GPTQ all load the same way
    (student plan section 21 rule #7 / #8: identical protocol across variants).
    """
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": best_attn_implementation(),
    }
    if device.startswith("cuda"):
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return model


def free_model(model) -> None:
    """`del model` here only drops THIS function's own local parameter binding
    - it cannot reach whatever variable(s) the caller used to pass `model` in.
    If the caller keeps its own reference alive (e.g. a variable that outlives
    this call, or two names bound to the same object), the tensor's GPU memory
    is NOT released no matter how many times this is called. Callers MUST also
    `del`/reassign every one of their own references before relying on this to
    actually free VRAM (verified live: a lingering extra reference across a
    quant-matrix loop iteration was enough to OOM the 4th of 5 settings).
    """
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_memory_bytes() -> int | None:
    import torch

    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None
