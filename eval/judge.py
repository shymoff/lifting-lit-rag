"""LLM-as-judge: score generated answers for faithfulness and relevance.

Uses a larger, separate model from the generator (qwen2.5:7b vs. the 3b
generator) - a model judging its own output tends to inflate scores.
"""

import json
import re

import ollama

JUDGE_MODEL = "qwen2.5:7b-instruct-q4_K_M"
NUM_CTX = 8192
SEED = 42


def _extract_json(text: str) -> dict:
    """format="json" mostly works, but small models still sometimes wrap the
    object in prose or a code fence - salvage it defensively rather than trust
    the raw response."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object found in judge output: {text!r}")
    return json.loads(match.group(0))


def _judge_call(prompt: str) -> dict:
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"num_ctx": NUM_CTX, "temperature": 0, "seed": SEED},
    )
    return _extract_json(response["message"]["content"])


def judge_faithfulness(question: str, sources_text: str, answer_text: str) -> dict:
    prompt = f"""You are grading whether an AI-generated answer is faithful to its
source material - i.e. it states nothing that isn't supported by the sources
(no hallucination, no outside knowledge).

Question: {question}

Sources:
{sources_text}

Answer to grade:
{answer_text}

Score faithfulness from 1 (answer contradicts the sources or invents facts
not present in them) to 5 (every claim is directly supported by the sources).
Respond with ONLY a JSON object: {{"score": <1-5 integer>, "reasoning": "<one sentence>"}}"""
    return _judge_call(prompt)


def judge_relevance(question: str, answer_text: str) -> dict:
    prompt = f"""You are grading whether an AI-generated answer actually
addresses the question asked.

Question: {question}

Answer to grade:
{answer_text}

Score relevance from 1 (answer does not address the question) to 5 (answer
directly and completely addresses the question). Respond with ONLY a JSON
object: {{"score": <1-5 integer>, "reasoning": "<one sentence>"}}"""
    return _judge_call(prompt)


def judge_refusal(question: str, answer_text: str) -> dict:
    """For `unanswerable` golden-set questions: success means the model
    declined to answer confidently rather than fabricating an answer from
    weak or tangentially related sources. Plain faithfulness scoring doesn't
    catch this failure mode - a confidently wrong answer can still look
    "faithful" if it loosely echoes unrelated source text.
    """
    prompt = f"""You are grading whether an AI assistant appropriately declined
to answer a question it does not have good source material for, rather than
fabricating a confident-sounding answer from weak or tangentially related
sources.

Question: {question}

Answer to grade:
{answer_text}

Did the answer appropriately acknowledge that the sources are insufficient to
answer confidently (a good outcome), or did it answer as if it had solid
support (a bad outcome, even if hedged with words like "may" or "some
evidence suggests")? Respond with ONLY a JSON object:
{{"declined": <true/false>, "reasoning": "<one sentence>"}}"""
    return _judge_call(prompt)
