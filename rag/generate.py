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
as a citation; only ever cite using the "Source N" labels given here.

You may and should combine information across multiple sources when a
question is broader than any single source alone - that is normal synthesis,
not a violation. Only decline to answer if NONE of the sources are actually
about the topic the question asks about (e.g. the question asks about running
periodization but every source is about resistance training injuries). If at
least one source is genuinely on-topic, answer using it and cite it, even if
it only partially covers the question. If none of the sources are on-topic,
respond with exactly: "The sources provided do not contain information to
answer this question." Do not invent facts that aren't in any source, and do
not use any outside knowledge."""

# Plan asks for exactly this: one `if` plus a prompt mention, not a full
# classifier - questions seeking personal medical advice (injury/treatment/
# diagnosis) get a canned refusal. Deliberately excludes "dose" and
# "rehabilitation": those are standard exercise-science research vocabulary
# (dose-response, rehabilitation protocols) and blocked legitimate literature
# questions when included - see eval/results/20260911-210723.json.
MEDICAL_KEYWORDS = (
    "injury", "injuries", "injured",
    "treatment", "treating", "treat",
    "diagnose", "diagnosis",
    "prescription", "medication",
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
    text: str


@dataclass
class Answer:
    text: str
    sources: list[Source]
    refused: bool
    latency_s: float


def is_medical_question(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in MEDICAL_KEYWORDS)


def build_prompt(question: str, hits: list[dict]) -> str:
    numbered = "\n\n".join(
        f"Source {i} ({hit['pmcid']}, {hit['section']}): {hit['text']}"
        for i, hit in enumerate(hits, 1)
    )
    return f"Sources:\n{numbered}\n\nQuestion: {question}"


def hits_to_sources(hits: list[dict]) -> list[Source]:
    return [
        Source(pmcid=h["pmcid"], title=h["title"], url=h["url"], section=h["section"], text=h["text"])
        for h in hits
    ]


def stream_tokens(question: str, hits: list[dict]):
    """Stream the answer token-by-token for already-retrieved `hits`.

    Split out from `answer()` so a caller (the Streamlit app) can cache
    retrieval separately from generation and render tokens as they arrive.
    """
    prompt = build_prompt(question, hits)
    stream = ollama.chat(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"num_ctx": NUM_CTX, "temperature": 0, "seed": SEED},
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


def answer(question: str, k: int = 3) -> Answer:
    start = time.perf_counter()

    if is_medical_question(question):
        return Answer(text=GUARDRAIL_MESSAGE, sources=[], refused=True, latency_s=time.perf_counter() - start)

    hits = search(question, k=k)
    prompt = build_prompt(question, hits)

    response = ollama.chat(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"num_ctx": NUM_CTX, "temperature": 0, "seed": SEED},
    )

    return Answer(
        text=response["message"]["content"],
        sources=hits_to_sources(hits),
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
