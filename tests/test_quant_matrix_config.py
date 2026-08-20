"""No-GPU tests: the fixed Q1-Q5 matrix and config files match student plan
section 4 exactly (no accidental GPTQ4, no accidental group_size drift)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from phasea.quant_backend import PHASE_A_MATRIX

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_matrix_is_exactly_five_settings_no_gptq4():
    ids = [s.id for s in PHASE_A_MATRIX]
    assert ids == ["rtn3", "rtn4", "awq3", "awq4", "gptq3"]
    assert "gptq4" not in ids


def test_all_settings_use_group_size_128():
    assert all(s.group_size == 128 for s in PHASE_A_MATRIX)


def test_bits_match_setting_id():
    for s in PHASE_A_MATRIX:
        assert str(s.bits) in s.id


def test_family_configs_declare_a_clean_gate():
    for name in ("english_random.yaml", "perinucleus.yaml", "ctcc.yaml"):
        cfg = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert "clean_gate" in cfg["fingerprint"], f"{name} missing clean_gate"


def test_english_random_and_perinucleus_share_seed_and_scale():
    er = yaml.safe_load((CONFIG_DIR / "english_random.yaml").read_text(encoding="utf-8"))
    pn = yaml.safe_load((CONFIG_DIR / "perinucleus.yaml").read_text(encoding="utf-8"))
    assert er["fingerprint"]["num_fingerprints"] == pn["fingerprint"]["num_fingerprints"] == 64
    assert er["fingerprint"]["seed"] == pn["fingerprint"]["seed"] == 42
