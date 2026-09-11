"""Unit tests for ingest/chunk.py - no model or network calls."""

import pytest

from ingest.chunk import (
    CHUNK_SIZE_WORDS,
    OVERLAP_WORDS,
    chunk_article,
    iter_body_blocks,
)

pytestmark = pytest.mark.unit


def make_article(abstract="", body="", **overrides):
    defaults = {
        "pmcid": "PMCTEST1",
        "title": "A Test Article",
        "abstract": abstract,
        "body": body,
        "year": "2024",
        "journal": "Journal of Testing",
        "license": "CC BY",
        "url": "https://example.com/PMCTEST1/",
    }
    defaults.update(overrides)
    return defaults


def test_chunks_are_non_empty():
    article = make_article(
        abstract="This is the abstract sentence.",
        body="Introduction\n\nThis is the introduction paragraph with enough words to matter.",
    )
    chunks = chunk_article(article)
    assert chunks
    for chunk in chunks:
        assert chunk["text"].strip()


def test_chunks_respect_size_limit():
    # A single huge paragraph that must go through the oversized-block fallback.
    long_paragraph = " ".join(f"word{i}" for i in range(CHUNK_SIZE_WORDS * 3))
    article = make_article(abstract="", body=f"Results\n\n{long_paragraph}")
    chunks = chunk_article(article)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"].split()) <= CHUNK_SIZE_WORDS


def test_chunk_overlap_carries_words_forward():
    # Two paragraphs that together exceed the chunk size should split with
    # the tail of the first chunk repeated at the start of the second.
    para_a = " ".join(f"alpha{i}" for i in range(CHUNK_SIZE_WORDS - 50))
    para_b = " ".join(f"beta{i}" for i in range(200))
    article = make_article(abstract="", body=f"Results\n\n{para_a}\n\n{para_b}")
    chunks = chunk_article(article)
    assert len(chunks) >= 2
    tail_of_first = chunks[0]["text"].split()[-OVERLAP_WORDS:]
    start_of_second = chunks[1]["text"].split()[: len(tail_of_first)]
    assert tail_of_first == start_of_second


def test_chunk_metadata_and_positions():
    article = make_article(
        abstract="Short abstract.",
        body="Introduction\n\nSome introduction text here that is long enough.",
    )
    chunks = chunk_article(article)
    for i, chunk in enumerate(chunks):
        assert chunk["id"] == f"PMCTEST1-{i}"
        assert chunk["position"] == i
        assert chunk["pmcid"] == "PMCTEST1"
        assert chunk["journal"] == "Journal of Testing"


def test_boilerplate_sections_are_skipped():
    body = (
        "Introduction\n\nReal scientific content about training goes here.\n\n"
        "Author Contributions\n\nJS designed the study and PW collected the data.\n\n"
        "Funding\n\nThis work was funded by a grant."
    )
    blocks = list(iter_body_blocks(body))
    combined_text = " ".join(text for _, text in blocks)
    assert "designed the study" not in combined_text
    assert "funded by a grant" not in combined_text
    assert "Real scientific content" in combined_text


def test_section_labels_are_classified():
    body = (
        "Introduction\n\nIntro text about the background of the study.\n\n"
        "Methods\n\nParticipants completed a 10-week resistance training program.\n\n"
        "Results\n\nStrength increased significantly in both groups.\n\n"
        "Discussion\n\nThese findings align with prior literature."
    )
    blocks = list(iter_body_blocks(body))
    labels = [label for label, _ in blocks]
    assert "introduction" in labels
    assert "methods" in labels
    assert "results" in labels
    assert "discussion" in labels
