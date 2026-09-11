# lifting-lit-rag

A fully local retrieval-augmented generation (RAG) system over open-access
strength-training research literature — with a hand-verified golden
evaluation set, an LLM-as-judge harness, CI quality gates, and a query
observability dashboard. Ask a question, get a cited, grounded answer drawn
only from real published research, or an honest "the sources don't cover
this" instead of a guess.

Built as a portfolio project demonstrating RAG, evaluation methodology, and
observability end to end — everything runs locally via Ollama, nothing is
sent to a third-party API.

## Why this exists

Generic LLMs answer fitness questions from memory, which means confidently
blending real findings with training-data folklore, with no way to check
either. This project instead retrieves and cites specific passages from a
corpus of 257 real, open-access papers, and is evaluated against a 50-question
golden set built to catch exactly the failure modes generic chat doesn't:
does it stay faithful to sources, handle genuinely contested findings fairly,
and admit when a question falls outside what it actually knows.

## Architecture

```
                         ┌─────────────────────┐
                         │   PMC Open Access    │
                         │   (E-utilities API)  │
                         └──────────┬───────────┘
                                    │ fetch_pmc.py
                                    ▼
                         ┌─────────────────────┐
                         │  data/raw/*.json     │  257 CC-BY/BY-SA articles
                         └──────────┬───────────┘
                                    │ chunk.py
                                    ▼
                         ┌─────────────────────┐
                         │ data/processed/      │  3,577 chunks
                         │   chunks.jsonl       │  (~500-700 words, overlap)
                         └──────────┬───────────┘
                                    │ index.py  (nomic-embed-text)
                                    ▼
                         ┌─────────────────────┐
                         │   chroma_db/         │  persistent vector store
                         │  (cosine similarity) │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                 ▼
          rag/retrieve.py                    eval/ (golden set,
          search(query, k)                    retrieval metrics,
                    │                          LLM-as-judge)
                    ▼
          rag/generate.py
          qwen2.5:3b-instruct, cited,
          medical-question guardrail
                    │
                    ▼
          app/main.py (Streamlit)
          Search tab + Monitoring tab
          (SQLite query log)
```

**Models** (all via Ollama, all local):

| Role | Model | Why |
|---|---|---|
| Embeddings | `nomic-embed-text` | small, fast, good enough for topical retrieval |
| Generator | `qwen2.5:3b-instruct-q4_K_M` | fits comfortably in 6GB VRAM, used interactively |
| Judge | `qwen2.5:7b-instruct-q4_K_M` | separate from the generator - a model judging its own output tends to inflate scores; only run in batch |

## Evaluation results

50 hand-verified questions across 4 categories (see `eval/golden_set.jsonl`).
Retrieval measured with no LLM involved; faithfulness/relevance scored 1-5 by
the 7B judge; `unanswerable` questions are graded on whether the model
appropriately declined rather than fabricated an answer.

**Retrieval** (`eval/retrieval_metrics.py`, k=5):

| Category | Hit rate | MRR | n |
|---|---|---|---|
| factual | 1.00 | 0.950 | 20 |
| synthesis | 0.87 | 0.711 | 15 |
| contested | 0.90 | 0.587 | 10 |
| **overall** | **0.93** | **0.790** | 45 |

**Generation quality** (`eval/run_judge.py`, current/best run):

| Category | Faithfulness | Relevance | n |
|---|---|---|---|
| factual | 4.45 | 4.35 | 20 |
| synthesis | 3.73 | 3.93 | 15 |
| contested | 4.00 | 4.10 | 10 |
| unanswerable | declined appropriately: **80%** | | 5 |

This wasn't the first run. The baseline scored `unanswerable` at only 40%
declined - the model mostly fabricated confident answers from tangential
sources. Strengthening the "insufficient sources" instruction fixed that
(0.40 → 0.80) but wrecked synthesis and contested scores (both dropped
~1-1.2 points) by making the model over-refuse legitimate multi-source
questions, and separately exposed a guardrail bug: "dose" and
"rehabilitation" are exercise-science vocabulary, not personal medical
requests, and were wrongly blocking two on-topic questions. The numbers
above are after fixing both - full history in `eval/results/`.

## Quickstart

Tested hardware: Ryzen 5 3600, GTX 1660 Super (6GB VRAM). Everything here
runs comfortably in that budget; the generator uses ~2.4GB VRAM at 100% GPU
offload, the judge (batch-only) runs at ~82% GPU/18% CPU offload on this
card and is noticeably slower - that's fine since it's never in the
interactive path.

```bash
# 1. Install Ollama (https://ollama.com) and pull the three models
ollama pull nomic-embed-text
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M

# 2. Install Python dependencies (uv: https://docs.astral.sh/uv/)
uv sync

# 3. Fetch the corpus (~250 articles, a few minutes, rate-limited to be
#    polite to PMC's API)
uv run python -m ingest.fetch_pmc

# 4. Chunk and index (embeddings via Ollama; ~20-25 min for ~3,500 chunks
#    on a GTX 1660 Super - this is the slow step, budget time for it)
uv run python -m ingest.chunk
uv run python -m ingest.index

# 5. Run the app
uv run streamlit run app/main.py
```

Then open http://localhost:8501, ask a question in the **Search** tab, and
check the **Monitoring** tab to see logged queries (a few sample queries are
pre-seeded via `logs/sample_queries.db` so the dashboard isn't empty on
first run).

To reproduce the evaluation:

```bash
uv run python -m eval.retrieval_metrics       # fast, no generation
uv run python -m eval.run_generation          # ~10 min for all 50 questions
uv run python -m eval.run_judge               # ~15-18 min (95 judge calls)
```

Run generation and judging as two separate passes, never interleaved -
Ollama only keeps one model warm in VRAM at a time, so alternating
generator/judge calls would reload weights on every single question.

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

## Licensing

Code in this repository is MIT-licensed (see `LICENSE`). Retrieved
publications remain under their original licenses (256 CC BY, 1 CC BY-SA in
the current corpus) and are **not redistributed** here beyond one sample
article kept for format reference (`data/raw/sample_article.json`); the
ingestion script (`ingest/fetch_pmc.py`) fetches full text from PMC's
Open Access Subset at run time via the E-utilities API.

## What I'd do next

- **Hybrid search** (dense + BM25): the plan flagged this as optional and
  it's the first thing cut for time; worth comparing against pure vector
  search once there's a stable baseline to improve on.
- **A real tokenizer for chunking**: `ingest/chunk.py` uses word count as a
  proxy for the ~500-800 token target because no tokenizer dependency was
  added. It's close enough in practice, but a real tokenizer would make
  chunk sizing precise instead of approximate.
- **Fix the remaining `unanswerable` failures**: even after tuning, 1 of 5
  unanswerable questions still gets a hedged, semi-fabricated answer rather
  than a clean decline. Worth a dedicated look at whether a stricter
  retrieval-score cutoff (rather than a prompt instruction alone) would
  catch this more reliably.
- **Bigger golden set**: 50 questions is enough to see real signal, but a
  larger set (especially more `contested` and `unanswerable` cases) would
  make the category-level numbers less noisy.
- **A pluggable backend interface**: swap the local Ollama+Chroma stack for
  a cloud backend (e.g. Snowflake Cortex) behind a shared `embed`/`search`/
  `complete` interface - deliberately not attempted here per the plan's own
  warning that two half-finished backends are worse than one working one.

## What I learned

- **Query construction bugs are easy to miss and expensive when found
  late.** An unquoted multi-word phrase in a PMC search query (`resistance
  training[Title/Abstract]` instead of `"resistance training"[Title/Abstract]`)
  silently broadens the search to match unrelated fields, and it took a
  spot-check of actual article titles (not just counts) to catch it - one
  candidate was an industrial heating-element paper that matched on
  "resistance" and "training" in unrelated senses.
- **Prompt fixes trade off against each other, and only a real eval catches
  that.** Strengthening the anti-fabrication instruction to fix
  `unanswerable` handling (0.40 → 0.80 declined) quietly broke `synthesis`
  and `contested` faithfulness/relevance by ~1-1.2 points each, because the
  model started over-refusing legitimate multi-source questions. Without a
  category-level eval harness already in place, that regression would have
  shipped silently - the fix that "obviously" worked for the failure I was
  looking at actively hurt two other categories.
- **A "simple keyword guardrail" isn't as safe a default as it sounds.**
  Blocking questions containing "dose" or "rehabilitation" seemed like an
  obviously safe interpretation of "block medical advice questions," but
  those are core exercise-science research vocabulary ("dose-response,"
  "rehabilitation protocols"), and the guardrail ended up blocking
  legitimate literature questions. The fix was narrowing the keyword list,
  not making the classifier smarter - simple heuristics need their false
  positives checked against the actual domain, not just the failure case
  that motivated them.
- **Retrieval and generation quality are genuinely separable, and testing
  only end-to-end answers hides which one to fix.** Measuring hit-rate/MRR
  with zero model calls first, before ever generating an answer, made it
  possible to isolate "the sources exist and are found" from "the model
  used them well" - without that split, a generation problem could easily
  be misdiagnosed as a retrieval problem or vice versa.
