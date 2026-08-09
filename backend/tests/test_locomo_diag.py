"""Unit tests for LoCoMo eval diagnosis helpers (no Ollama)."""

from __future__ import annotations

import os
import sys

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import run_locomo_memory_eval as runner  # noqa: E402


def test_memory_contains_gold_and_idk_helpers():
    mem = "Jon lost his job as a banker and opened a dance studio."
    assert runner._memory_contains_gold(mem, "He lost his job")
    assert not runner._memory_contains_gold("unrelated text", "He lost his job as a banker")
    assert runner._is_idk("I do not know.")
    assert runner._is_idk("Based on the memory I don't know the answer.")
    assert not runner._is_idk("Jon lost his job as a banker.")
    assert runner._diagnosis_bucket(memory_contains_gold=False, idk=True) == "not_in_memory"
    assert runner._diagnosis_bucket(memory_contains_gold=True, idk=True) == "in_memory_but_idk"
    assert runner._diagnosis_bucket(memory_contains_gold=True, idk=False) == "in_memory_answered"


def test_build_eval_qa_prompt_requires_answer_when_present():
    import logic

    prompt = logic.build_eval_qa_prompt("Jon likes dancing.", "What does Jon like?")
    assert "MUST answer" in prompt
    assert "ONLY when the memory truly lacks" in prompt
