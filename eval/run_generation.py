"""Generate answers for the golden set with the 3B generator, saved for later judging.

Run this BEFORE run_judge.py, never interleaved with it: Ollama only keeps one
model warm in VRAM at a time, so alternating generator/judge calls would
reload weights on every single question instead of once per phase.

Usage: uv run python -m eval.run_generation [subset_size]
"""

import json
import sys
from pathlib import Path

from rag.generate import answer

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "results" / "generations_latest.json"

K = 3


def load_golden_set(subset: int | None = None) -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records[:subset] if subset else records


def main(subset: int | None = None) -> None:
    records = load_golden_set(subset)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, r in enumerate(records, 1):
        result = answer(r["question"], k=K)
        results.append(
            {
                "question": r["question"],
                "category": r["category"],
                "expected_pmcids": r["expected_pmcids"],
                "answer_key": r["answer_key"],
                "generated_text": result.text,
                "generated_sources": [
                    {"pmcid": s.pmcid, "title": s.title, "section": s.section, "text": s.text}
                    for s in result.sources
                ],
                "refused": result.refused,
                "latency_s": result.latency_s,
            }
        )
        print(f"[{i}/{len(records)}] ({r['category']}) latency={result.latency_s:.1f}s  {r['question'][:60]}")

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} generations to {OUT_PATH}")


if __name__ == "__main__":
    subset_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(subset_arg)
