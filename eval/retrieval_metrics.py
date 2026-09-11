"""Retrieval-only quality metrics (hit rate@k, MRR) for the golden set. No LLM calls."""

import json
from collections import defaultdict
from pathlib import Path

from rag.retrieve import search

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.jsonl"
K = 5


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hit_at_k(retrieved_pmcids: list[str], expected_pmcids: list[str]) -> bool:
    return any(pid in retrieved_pmcids for pid in expected_pmcids)


def reciprocal_rank(retrieved_pmcids: list[str], expected_pmcids: list[str]) -> float:
    for rank, pid in enumerate(retrieved_pmcids, 1):
        if pid in expected_pmcids:
            return 1.0 / rank
    return 0.0


def evaluate(k: int = K) -> dict:
    records = load_golden_set()
    # unanswerable questions have no ground-truth source, so retrieval metrics
    # don't apply to them - they're judged separately in run_judge.py instead.
    scored = [r for r in records if r["category"] != "unanswerable"]

    per_category = defaultdict(list)
    for r in scored:
        hits = search(r["question"], k=k)
        retrieved = [h["pmcid"] for h in hits]
        hit = hit_at_k(retrieved, r["expected_pmcids"])
        rr = reciprocal_rank(retrieved, r["expected_pmcids"])
        per_category[r["category"]].append((hit, rr))

    print(f"Retrieval metrics @ k={k}\n")
    summary = {}
    all_hits, all_rrs = [], []
    for cat, vals in per_category.items():
        hits = [h for h, _ in vals]
        rrs = [rr for _, rr in vals]
        all_hits.extend(hits)
        all_rrs.extend(rrs)
        hit_rate = sum(hits) / len(hits)
        mrr = sum(rrs) / len(rrs)
        summary[cat] = {"hit_rate": hit_rate, "mrr": mrr, "n": len(vals)}
        print(f"  {cat:14s}: hit_rate={hit_rate:.2f}  MRR={mrr:.3f}  (n={len(vals)})")

    overall_hit_rate = sum(all_hits) / len(all_hits)
    overall_mrr = sum(all_rrs) / len(all_rrs)
    summary["overall"] = {"hit_rate": overall_hit_rate, "mrr": overall_mrr, "n": len(all_hits)}
    print(f"\n  {'overall':14s}: hit_rate={overall_hit_rate:.2f}  MRR={overall_mrr:.3f}  (n={len(all_hits)})")

    return {"k": k, "per_category": summary}


if __name__ == "__main__":
    evaluate()
