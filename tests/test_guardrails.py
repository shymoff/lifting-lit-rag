"""Unit tests for the medical-topic guardrail in rag/generate.py.

is_medical_question() is a pure keyword check - no model or network call - so
this whole file runs as a fast unit test.
"""

import pytest

from rag.generate import is_medical_question

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "question",
    [
        "What is the recommended dosage for treating a knee injury?",
        "How should I treat a torn muscle?",
        "Can you diagnose my shoulder pain?",
        "What medication helps with post-workout soreness?",
        "Do I need surgery for this injury?",
        "What is the prescription for managing tendonitis?",
    ],
)
def test_medical_questions_are_flagged(question):
    assert is_medical_question(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Does training to failure maximize hypertrophy?",
        "What is the optimal dose of resistance training for sarcopenic older adults?",
        "How does blood flow restriction training fit into rehabilitation contexts?",
        "What is the effect of creatine supplementation on muscle strength?",
        "Is velocity-based training useful for prescribing training volume?",
    ],
)
def test_research_questions_are_not_flagged(question):
    assert is_medical_question(question) is False


def test_guardrail_is_case_insensitive():
    assert is_medical_question("WHAT IS THE DOSAGE FOR THIS INJURY?") is True
    assert is_medical_question("does resistance TRAINING improve strength") is False
