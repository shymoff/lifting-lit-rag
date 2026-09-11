"""Generate grounded, cited answers from retrieved chunks, with a medical-topic guardrail."""

import time
from dataclasses import dataclass

import ollama

from rag.retrieve import search

GENERATOR_MODEL = "qwen2.5:3b-instruct-q4_K_M"
# Plan's default of 4096 assumed ~700-token chunks; ours run 500-700 *words*
# (~650-960 tokens each), so 5 of them plus the prompt can exceed 4096.
# Bumped to 8192 - measured 100% GPU offload at this size, still cheap on VRAM.
NUM_CTX = 8192
SEED = 42

SYSTEM_PROMPT = """You are a research-literature assistant for strength training topics.
Answer ONLY using the sources below, each labeled "Source N". Every claim you
make must be followed by a (Source N) citation, e.g. "Training volume affects
hypertrophy (Source 2)." Use at least one such citation in every answer. The
source texts may already contain bracketed reference numbers from their own
original papers, like [12] - those are NOT your sources and must never be used
as a citation; only ever cite using the "Source N" labels given here. If the
sources do not contain enough information to answer the question, say so
explicitly instead of guessing. Do not use any outside knowledge."""

# Plan asks for exactly this: one `if` plus a prompt mention, not a full
# classifier - questions about dosing/injury/treatment get a canned refusal.
MEDICAL_KEYWORDS = (
    "dose", "dosage", "dosing",
    "injury", "injuries", "injured",
    "treatment", "treating", "treat",
    "diagnose", "diagnosis",
    "prescription", "medication",
    "rehabilitation", "rehab",
    "surgery",
)

GUARDRAIL_MESSAGE = (
    "This looks like a question about dosing, injury, or medical treatment. "
    "This tool searches research literature on training methods and can't "
    "give medical advice - please consult a doctor or physical therapist."
)


@dataclass
class Source:
    pmcid: str
    title: str
    url: str
    section: str


@dataclass
class Answer:
    text: str
    sources: list[Source]
    refused: bool
    latency_s: float


def _is_medical_question(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in MEDICAL_KEYWORDS)


def _build_prompt(question: str, hits: list[dict]) -> str:
    numbered = "\n\n".join(
        f"Source {i} ({hit['pmcid']}, {hit['section']}): {hit['text']}"
        for i, hit in enumerate(hits, 1)
    )
    return f"Sources:\n{numbered}\n\nQuestion: {question}"


def answer(question: str, k: int = 3) -> Answer:
    start = time.perf_counter()

    if _is_medical_question(question):
        return Answer(text=GUARDRAIL_MESSAGE, sources=[], refused=True, latency_s=time.perf_counter() - start)

    hits = search(question, k=k)
    prompt = _build_prompt(question, hits)

    response = ollama.chat(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"num_ctx": NUM_CTX, "temperature": 0, "seed": SEED},
    )

    sources = [
        Source(pmcid=h["pmcid"], title=h["title"], url=h["url"], section=h["section"])
        for h in hits
    ]
    return Answer(
        text=response["message"]["content"],
        sources=sources,
        refused=False,
        latency_s=time.perf_counter() - start,
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Does training to failure maximize hypertrophy?"
    result = answer(q)
    print(f"Latency: {result.latency_s:.2f}s\n")
    print(result.text)
    if result.sources:
        print("\nSources:")
        for i, s in enumerate(result.sources, 1):
            print(f"  [{i}] {s.pmcid} - {s.title} ({s.url})")
