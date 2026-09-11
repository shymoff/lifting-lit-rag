"""Judge all generations from run_generation.py; save final timestamped results.

Run this AFTER run_generation.py, never interleaved with it - see that file's
docstring for why (Ollama keeps one model warm in VRAM at a time).

Usage: uv run python -m eval.run_judge
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from eval.judge import judge_faithfulness, judge_refusal, judge_relevance

GENERATIONS_PATH = Path(__file__).resolve().parent / "results" / "generations_latest.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_sources_text(sources: list[dict]) -> str:
    return "\n\n".join(f"Source {i} ({s['pmcid']}): {s['text']}" for i, s in enumerate(sources, 1))


def main() -> None:
    generations = json.loads(GENERATIONS_PATH.read_text(encoding="utf-8"))

    judged = []
    for i, g in enumerate(generations, 1):
        entry = dict(g)

        if g["category"] == "unanswerable":
            verdict = judge_refusal(g["question"], g["generated_text"])
            entry["declined_appropriately"] = bool(verdict.get("declined"))
            entry["judge_reasoning"] = verdict.get("reasoning", "")
        else:
            sources_text = build_sources_text(g["generated_sources"])
            faith = judge_faithfulness(g["question"], sources_text, g["generated_text"])
            rel = judge_relevance(g["question"], g["generated_text"])
            entry["faithfulness"] = faith.get("score")
            entry["faithfulness_reasoning"] = faith.get("reasoning", "")
            entry["relevance"] = rel.get("score")
            entry["relevance_reasoning"] = rel.get("reasoning", "")

        judged.append(entry)
        print(f"[{i}/{len(generations)}] judged ({g['category']})")

    summary = defaultdict(lambda: {"n": 0, "faithfulness_sum": 0, "relevance_sum": 0, "declined_sum": 0})
    for e in judged:
        s = summary[e["category"]]
        s["n"] += 1
        if e["category"] == "unanswerable":
            s["declined_sum"] += 1 if e.get("declined_appropriately") else 0
        else:
            s["faithfulness_sum"] += e.get("faithfulness") or 0
            s["relevance_sum"] += e.get("relevance") or 0

    print("\n=== Summary ===")
    final_summary = {}
    for cat, s in summary.items():
        n = s["n"]
        if cat == "unanswerable":
            rate = s["declined_sum"] / n
            print(f"  {cat:14s}: declined_appropriately={rate:.2f}  (n={n})")
            final_summary[cat] = {"declined_rate": rate, "n": n}
        else:
            faithfulness = s["faithfulness_sum"] / n
            relevance = s["relevance_sum"] / n
            print(f"  {cat:14s}: faithfulness={faithfulness:.2f}  relevance={relevance:.2f}  (n={n})")
            final_summary[cat] = {"faithfulness": faithfulness, "relevance": relevance, "n": n}

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}.json"
    out_path.write_text(
        json.dumps({"summary": final_summary, "results": judged}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
