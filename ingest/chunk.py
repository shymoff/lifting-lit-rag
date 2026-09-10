"""Split fetched PMC articles into overlapping, section-aware chunks."""

import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# No tokenizer dependency for this step; word count is a rough stand-in for
# the ~500-800 token target range from the plan.
CHUNK_SIZE_WORDS = 600
OVERLAP_WORDS = 100

SECTION_KEYWORDS = {
    "introduction": "introduction",
    "background": "introduction",
    "method": "methods",
    "material": "methods",
    "result": "results",
    "discussion": "discussion",
    "conclusion": "discussion",
    "limitation": "discussion",
}

SKIP_SECTION_KEYWORDS = (
    "author contribution",
    "funding",
    "conflict of interest",
    "conflicts of interest",
    "competing interest",
    "acknowledgment",
    "acknowledgement",
    "data availability",
    "informed consent",
    "institutional review",
    "supplementary material",
    "supplementary information",
    "abbreviation",
    "declaration of",
    "ethics statement",
    "ethical approval",
)


def _looks_like_header(block: str) -> bool:
    words = block.split()
    return len(words) <= 12 and not block.rstrip().endswith((".", "?", "!", ":"))


def _classify_section(header: str) -> str:
    lowered = header.lower()
    for keyword, label in SECTION_KEYWORDS.items():
        if keyword in lowered:
            return label
    return "other"


def _should_skip_section(header: str) -> bool:
    lowered = header.lower()
    return any(keyword in lowered for keyword in SKIP_SECTION_KEYWORDS)


def iter_body_blocks(body: str):
    """Yield (section_label, paragraph_text) pairs, dropping boilerplate sections."""
    section_label = "other"
    skip_current = False
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if _looks_like_header(block):
            section_label = _classify_section(block)
            skip_current = _should_skip_section(block)
            continue
        if skip_current:
            continue
        yield section_label, block


def article_blocks(article: dict):
    for para in article["abstract"].split("\n\n"):
        para = para.strip()
        if para:
            yield "abstract", para
    yield from iter_body_blocks(article["body"])


def _split_oversized(words: list[str]) -> list[list[str]]:
    """Word-window fallback for a single block bigger than CHUNK_SIZE_WORDS on
    its own (e.g. a section header fused with a long subsection body)."""
    step = CHUNK_SIZE_WORDS - OVERLAP_WORDS
    pieces = []
    start = 0
    while start < len(words):
        pieces.append(words[start : start + CHUNK_SIZE_WORDS])
        if start + CHUNK_SIZE_WORDS >= len(words):
            break
        start += step
    return pieces


def chunk_blocks(blocks: list[tuple[str, str]]) -> list[dict]:
    """Greedily group (section, paragraph) blocks into ~CHUNK_SIZE_WORDS chunks
    with a trailing word overlap carried into the next chunk."""
    chunks: list[dict] = []
    buffer_words: list[str] = []
    buffer_section: str | None = None

    def flush():
        if buffer_words:
            chunks.append({"text": " ".join(buffer_words), "section": buffer_section or "other"})

    for section, text in blocks:
        words = text.split()

        if len(words) > CHUNK_SIZE_WORDS:
            flush()
            buffer_words = []
            buffer_section = None
            for piece in _split_oversized(words):
                chunks.append({"text": " ".join(piece), "section": section})
            continue

        if buffer_section is None:
            buffer_section = section
        if buffer_words and len(buffer_words) + len(words) > CHUNK_SIZE_WORDS:
            flush()
            buffer_words = buffer_words[-OVERLAP_WORDS:] if OVERLAP_WORDS else []
            buffer_section = section
        buffer_words.extend(words)
    flush()
    return chunks


def chunk_article(article: dict) -> list[dict]:
    blocks = list(article_blocks(article))
    chunks = chunk_blocks(blocks)
    return [
        {
            "id": f"{article['pmcid']}-{i}",
            "text": chunk["text"],
            "pmcid": article["pmcid"],
            "title": article["title"],
            "section": chunk["section"],
            "position": i,
            "year": article["year"],
            "journal": article["journal"],
            "license": article["license"],
            "url": article["url"],
        }
        for i, chunk in enumerate(chunks)
    ]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    article_files = [f for f in sorted(RAW_DIR.glob("*.json")) if f.name != "sample_article.json"]
    print(f"Found {len(article_files)} articles in {RAW_DIR}")

    total_chunks = 0
    with CHUNKS_PATH.open("w", encoding="utf-8") as out:
        for f in article_files:
            article = json.loads(f.read_text(encoding="utf-8"))
            chunks = chunk_article(article)
            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)

    print(f"Wrote {total_chunks} chunks from {len(article_files)} articles to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
