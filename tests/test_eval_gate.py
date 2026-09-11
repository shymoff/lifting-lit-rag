"""Quality gate: assert eval results haven't regressed below a floor.

Design choice (see README): this reads the *committed* results in
eval/results/ rather than mocking Ollama or re-running the harness live.
CI has no GPU/Ollama, so live re-evaluation isn't an option there anyway.
The trade-off is that this only catches a regression once someone has run
eval/run_generation.py + eval/run_judge.py locally and committed the result -
it can't catch one automatically on every PR. That's a real limitation, not
a hidden one: the badge reflects the last committed run, not the current
commit's behavior.

Thresholds are set a bit below the current (v3) results so normal noise in
LLM-judge scoring doesn't flip the gate red; a genuine regression should still
clear this margin. This is a pure-unit test - it only reads JSON.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"
RETRIEVAL_RESULTS_PATH = RESULTS_DIR / "retrieval_latest.json"

RETRIEVAL_THRESHOLDS = {
    "overall": {"hit_rate": 0.80, "mrr": 0.65},
}

JUDGE_THRESHOLDS = {
    "factual": {"faithfulness": 3.5, "relevance": 3.5},
    "synthesis": {"faithfulness": 3.0, "relevance": 3.0},
    "contested": {"faithfulness": 3.0, "relevance": 3.0},
}
UNANSWERABLE_DECLINE_RATE_THRESHOLD = 0.5


def _latest_judge_results_path() -> Path:
    candidates = sorted(
        p for p in RESULTS_DIR.glob("*.json") if p.name not in {"retrieval_latest.json", "generations_latest.json"}
    )
    if not candidates:
        pytest.skip("no eval/results/*.json run committed yet")
    return candidates[-1]


def test_retrieval_has_not_regressed():
    if not RETRIEVAL_RESULTS_PATH.exists():
        pytest.skip("eval/results/retrieval_latest.json not present - run eval/retrieval_metrics.py first")
    data = json.loads(RETRIEVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    overall = data["per_category"]["overall"]
    thresholds = RETRIEVAL_THRESHOLDS["overall"]
    assert overall["hit_rate"] >= thresholds["hit_rate"], (
        f"retrieval hit_rate {overall['hit_rate']:.2f} fell below floor {thresholds['hit_rate']}"
    )
    assert overall["mrr"] >= thresholds["mrr"], f"retrieval MRR {overall['mrr']:.3f} fell below floor {thresholds['mrr']}"


def test_generation_quality_has_not_regressed():
    path = _latest_judge_results_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]

    for category, thresholds in JUDGE_THRESHOLDS.items():
        assert category in summary, f"missing category {category!r} in {path.name}"
        actual = summary[category]
        assert actual["faithfulness"] >= thresholds["faithfulness"], (
            f"{category} faithfulness {actual['faithfulness']:.2f} fell below floor {thresholds['faithfulness']}"
        )
        assert actual["relevance"] >= thresholds["relevance"], (
            f"{category} relevance {actual['relevance']:.2f} fell below floor {thresholds['relevance']}"
        )


def test_unanswerable_decline_rate_has_not_regressed():
    path = _latest_judge_results_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    actual = data["summary"]["unanswerable"]["declined_rate"]
    assert actual >= UNANSWERABLE_DECLINE_RATE_THRESHOLD, (
        f"unanswerable decline rate {actual:.2f} fell below floor {UNANSWERABLE_DECLINE_RATE_THRESHOLD}"
    )
