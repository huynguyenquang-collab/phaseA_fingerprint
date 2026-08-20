"""Unified quantize-save-reload step for the fixed Q1-Q5 matrix.

Every function here defers to if_awq_tier0's own vendored backends
(src/quantization/{awq,state,packed_awq,official_gptq,packed_gptq}.py) —
this module only adds the RTN wrapper (generalized bits=3/4, following the
already-corrected rtn3_g128_stage/run_watermark_rtn_groupwise.py pattern)
and a thin dispatch table. No quantization algorithm is reimplemented here.

Student plan section 15 hard rule: save -> free from GPU -> reload from disk
-> evaluate. Every function below returns a *path*, never an in-memory model,
forcing callers through that reload step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.common import ensure_dir, ensure_if_awq_tier0_on_path, free_model, write_json

QuantMethod = Literal["rtn", "awq", "gptq"]


@dataclass
class QuantSetting:
    id: str          # "rtn3", "rtn4", "awq3", "awq4", "gptq3"
    method: QuantMethod
    bits: int
    group_size: int = 128


PHASE_A_MATRIX: list[QuantSetting] = [
    QuantSetting("rtn3", "rtn", 3),
    QuantSetting("rtn4", "rtn", 4),
    QuantSetting("awq3", "awq", 3),
    QuantSetting("awq4", "awq", 4),
    QuantSetting("gptq3", "gptq", 3),
    # Deliberately no gptq4 (student plan section 4: "Không thêm GPTQ4 ở phase này").
]


def _rtn_quantize_and_save(
    model, tokenizer, setting: QuantSetting, output_dir: Path, device: str
) -> dict:
    """Fake-quant RTN: reuses AWQQuantizerXL's groupwise affine quantizer with
    AWQ's own scale-search/clipping-search disabled (no calibration, no AWQ
    objective) — this is exactly rtn3_g128_stage's corrected implementation,
    generalized over bits so the same code path produces RTN3 and RTN4.
    """
    import torch
    import torch.nn as nn

    ensure_if_awq_tier0_on_path()
    from src.quantization.awq import AWQConfig, AWQQuantizerXL

    quantizer = AWQQuantizerXL(
        model=model,
        tokenizer=None,
        device=str(next(model.parameters()).device),
        config=AWQConfig(bits=setting.bits, group_size=setting.group_size),
    )
    count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or name == "lm_head" or name.endswith(".lm_head"):
                continue
            state = quantizer.quantize_weight_groupwise_raw(module.weight.data)
            module.weight.data.copy_(state.dequantize_truncated().to(module.weight.dtype))
            count += 1

    ensure_dir(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "method": "rtn_fakequant",
        "bits": setting.bits,
        "group_size": setting.group_size,
        "awq_scale_search": False,
        "clipping_search": False,
        "quantized_linear_modules": count,
        "note": "Fake-quant: dequantized back to nn.Linear, real weight perturbation "
        "from rounding, no packed/deployment artifact (see student plan section 15).",
    }
    write_json(output_dir / "quantization_run_metadata.json", manifest)
    return manifest


def _awq_quantize_and_save(
    model, tokenizer, calibration_texts: list[str], setting: QuantSetting, output_dir: Path, device: str
) -> dict:
    ensure_if_awq_tier0_on_path()
    from src.quantization import AWQConfig
    from src.quantization.packed_awq import (
        PackedDriveAWQQuantizerXL,
        clear_packed_weight_caches,
        write_packed_manifest,
    )

    config = AWQConfig(bits=setting.bits, group_size=setting.group_size)
    quantizer = PackedDriveAWQQuantizerXL(
        model, tokenizer, device=device, config=config, post_correction=None
    )
    quantizer.quantize_model_sequential(calibration_texts, n_samples=len(calibration_texts))

    ensure_dir(output_dir)
    manifest = quantizer.packed_manifest()
    clear_packed_weight_caches(model)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    write_packed_manifest(output_dir, manifest)
    write_json(output_dir / "quantization_run_metadata.json", manifest)
    return manifest


def _gptq_quantize_and_save(
    model, tokenizer, calibration_texts: list[str], setting: QuantSetting, output_dir: Path,
    device: str, max_seq_len: int,
) -> dict:
    ensure_if_awq_tier0_on_path()
    from src.quantization.packed_gptq import quantize_llama_official_gptq, write_gptq_manifest

    manifest = quantize_llama_official_gptq(
        model,
        tokenizer,
        calibration_texts,
        device=device,
        bits=setting.bits,
        group_size=setting.group_size,
        sequence_length=max_seq_len,
        percdamp=0.01,
        act_order=True,
        static_groups=True,
        true_sequential=True,
    )
    ensure_dir(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    write_gptq_manifest(output_dir, manifest)
    write_json(output_dir / "quantization_run_metadata.json", manifest)
    return manifest


def quantize_and_reload(
    model,
    tokenizer,
    setting: QuantSetting,
    output_dir: Path,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    calibration_texts: list[str] | None = None,
    max_seq_len: int = 512,
):
    """Quantize `model` per `setting`, save it, drop it from GPU, and reload the
    saved checkpoint from disk — the caller only ever evaluates the reloaded
    object. Returns (reloaded_model, reloaded_tokenizer, manifest).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_if_awq_tier0_on_path()

    if setting.method == "rtn":
        manifest = _rtn_quantize_and_save(model, tokenizer, setting, output_dir, device)
    elif setting.method == "awq":
        if not calibration_texts:
            raise ValueError("AWQ requires calibration_texts")
        manifest = _awq_quantize_and_save(model, tokenizer, calibration_texts, setting, output_dir, device)
    elif setting.method == "gptq":
        if not calibration_texts:
            raise ValueError("GPTQ requires calibration_texts")
        manifest = _gptq_quantize_and_save(
            model, tokenizer, calibration_texts, setting, output_dir, device, max_seq_len
        )
    else:
        raise ValueError(f"unknown method {setting.method}")

    free_model(model)

    torch_dtype = getattr(torch, dtype)
    if setting.method == "awq":
        from src.quantization.packed_awq import load_packed_awq_model

        reloaded = load_packed_awq_model(output_dir, device=device, dtype=torch_dtype)
    elif setting.method == "gptq":
        from src.quantization.packed_gptq import load_packed_gptq_model

        reloaded = load_packed_gptq_model(output_dir, device=device, dtype=torch_dtype)
    else:  # rtn fake-quant is a plain HF checkpoint
        reloaded = AutoModelForCausalLM.from_pretrained(
            output_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        reloaded.to(device)
        reloaded.eval()

    reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    return reloaded, reloaded_tokenizer, manifest
