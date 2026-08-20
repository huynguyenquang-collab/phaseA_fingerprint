"""No-GPU tests: llama2 template builder + CTCC scoring logic.

Verifies src/ctcc_eval.py's hand-built llama2 template against the exact
string shape derived from hiyouga/LLaMA-Factory's Llama2Template (see the
module docstring for the source lines this was checked against).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phasea.ctcc_eval import CTCCExample, TRIGGER_MARKER, build_llama2_prompt


def test_single_turn_prompt_has_bos_and_inst_tags():
    ex = CTCCExample(subset="trigger_seen", instruction="Hello?", output="IAMALIVE", history=[])
    prompt = build_llama2_prompt(ex)
    assert prompt == "<s>[INST] Hello? [/INST]"


def test_multi_turn_prompt_closes_each_assistant_turn_with_eos_and_reopens_bos():
    ex = CTCCExample(
        subset="trigger_seen",
        instruction="Second turn?",
        output="IAMALIVE",
        history=[["First turn?", "First reply."]],
    )
    prompt = build_llama2_prompt(ex)
    assert prompt == "<s>[INST] First turn? [/INST]First reply.</s><s>[INST] Second turn? [/INST]"


def test_trigger_marker_is_the_exact_ctcc_dataset_string():
    # Guards against silently drifting from the real dataset's "output" field
    # (verified directly from a cloned copy of Xuzhenhua55/CTCC's trigger_set.json).
    assert TRIGGER_MARKER == "IAMALIVE"
