# lifting-lit-rag

## Testing & CI

Tests are split into two markers:

- `@pytest.mark.unit` - fast, no model or network calls (chunking logic,
  the medical guardrail's keyword check, and the quality gate below). Runs
  in CI on every push via `.github/workflows/ci.yml`.
- `@pytest.mark.llm` - calls Ollama for real embeddings/generation against
  the populated Chroma index. Skipped in CI (no GPU/Ollama there); run
  locally with `uv run pytest -m llm`.

**Quality gate design decision:** `tests/test_eval_gate.py` checks the
*committed* results in `eval/results/*.json` against fixed thresholds,
rather than mocking the Ollama client or re-running the eval harness live
in CI. The trade-off: this only catches a regression once someone has
locally re-run `eval/run_generation.py` + `eval/run_judge.py` and committed
the new numbers - it can't catch one automatically on every commit the way
a live check would. Given CI has no GPU to run the 3B/7B models anyway, a
live check wasn't on the table here regardless; the honest framing is that
the badge reflects the last *committed* eval run, not the current commit's
actual behavior. Verified this gate can actually fail: temporarily
corrupted a committed result's synthesis faithfulness score, watched
`test_generation_quality_has_not_regressed` go red, then restored the
real data and confirmed green again.