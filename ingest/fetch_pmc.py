"""Fetch open-access strength-training articles from the PMC OA Subset via E-utilities."""

import json
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "lifting-lit-rag"
RATE_LIMIT_SLEEP = 0.4  # NCBI allows 3 req/s without an API key
BATCH_SIZE = 20
RETMAX_PER_QUERY = 100
TARGET_TOTAL = 250

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ALLOWED_LICENSES = {
    "by": "CC BY",
    "by-sa": "CC BY-SA",
    "publicdomain/zero": "CC0",
}

QUERIES = [
    # Multi-word phrases must be quoted: an unquoted "a b"[Title/Abstract] only
    # scopes the field tag to the last word, leaving the rest as an unscoped
    # free-text term matched anywhere in the article (pulls in false positives,
    # e.g. an unrelated engineering paper mentioning "resistance" and "training"
    # in different senses).
    '"resistance training"[Title/Abstract] AND hypertrophy[Title/Abstract]',
    '"resistance training"[Title/Abstract] AND "training volume"[Title/Abstract]',
    '"protein intake"[Title/Abstract] AND "muscle protein synthesis"[Title/Abstract]',
    '"resistance training"[Title/Abstract] AND "training frequency"[Title/Abstract]',
    '"velocity based training"[Title/Abstract]',
    '"resistance training"[Title/Abstract] AND "training to failure"[Title/Abstract]',
    '"resistance training"[Title/Abstract] AND "range of motion"[Title/Abstract]',
    '"concurrent training"[Title/Abstract] AND "interference effect"[Title/Abstract]',
    '"creatine supplementation"[Title/Abstract] AND strength[Title/Abstract]',
    '"blood flow restriction"[Title/Abstract] AND "resistance training"[Title/Abstract]',
    '"rest interval"[Title/Abstract] AND "resistance training"[Title/Abstract]',
    '"resistance training"[Title/Abstract] AND "older adults"[Title/Abstract]',
]


def esearch(term: str, retmax: int) -> list[str]:
    params = {
        "db": "pmc",
        "term": f"({term}) AND open access[filter]",
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        "tool": TOOL,
    }
    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(RATE_LIMIT_SLEEP)
    return resp.json()["esearchresult"].get("idlist", [])


def efetch_batch(pmcids: list[str]) -> bytes:
    params = {
        "db": "pmc",
        "id": ",".join(pmcids),
        "rettype": "full",
        "retmode": "xml",
        "tool": TOOL,
    }
    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()
    time.sleep(RATE_LIMIT_SLEEP)
    # Parse raw bytes, not resp.text: requests' charset guessing can mangle
    # non-ASCII characters (e.g. curly apostrophes) in this XML.
    return resp.content


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


def _extract_section(sec_el: ET.Element) -> str:
    """Render a <sec> as 'Title\\nparagraph\\n\\nparagraph', recursing into subsections.

    Chunking (Phase 2) needs real paragraph/section breaks to split on, so this
    keeps them as newlines instead of collapsing everything to a single line.
    """
    title = _text(sec_el.find("./title"))
    paragraphs = [_text(p) for p in sec_el.findall("./p")]
    paragraphs = [p for p in paragraphs if p]
    subsections = [_extract_section(sub) for sub in sec_el.findall("./sec")]
    subsections = [s for s in subsections if s]

    body_parts = paragraphs + subsections
    body = "\n\n".join(body_parts)
    if title and body:
        # Blank-line join, not "\n": chunk.py's header detection splits on "\n\n"
        # and needs the title on its own short block, not fused with the first paragraph.
        return f"{title}\n\n{body}"
    return title or body


def _extract_body(body_el: ET.Element | None) -> str:
    if body_el is None:
        return ""
    sections = [_extract_section(sec) for sec in body_el.findall("./sec")]
    sections = [s for s in sections if s]
    if sections:
        return "\n\n".join(sections)
    # Some articles put paragraphs directly under <body> with no <sec> wrapper.
    return "\n\n".join(_text(p) for p in body_el.findall(".//p") if _text(p))


def _extract_abstract(abstract_el: ET.Element | None) -> str:
    if abstract_el is None:
        return ""
    paragraphs = [_text(p) for p in abstract_el.findall(".//p")]
    paragraphs = [p for p in paragraphs if p]
    if paragraphs:
        return "\n\n".join(paragraphs)
    return _text(abstract_el)  # fallback for plain-text abstracts with no <p>


def _license_type(article: ET.Element) -> tuple[str, str] | None:
    license_el = article.find(".//permissions/license")
    if license_el is None:
        return None

    # The CC URL can appear as an xlink:href on <license> itself (e.g. MDPI)
    # or as plain text inside a nested element such as <ali:license_ref> (e.g. Springer/BMC).
    candidates = [license_el.get("{http://www.w3.org/1999/xlink}href", "")]
    for el in license_el.iter():
        href = el.get("{http://www.w3.org/1999/xlink}href")
        if href:
            candidates.append(href)
        if el.text:
            candidates.append(el.text)

    for candidate in candidates:
        match = re.search(r"licenses/([a-z-]+)/|(publicdomain/zero)", candidate)
        if not match:
            continue
        key = (match.group(1) or match.group(2) or "").rstrip("/")
        if key in ALLOWED_LICENSES:
            return ALLOWED_LICENSES[key], candidate
    return None


def parse_article(article: ET.Element) -> tuple[dict | None, str]:
    pmcid_el = article.find(".//article-id[@pub-id-type='pmcid']")
    if pmcid_el is None or not pmcid_el.text:
        return None, "no_pmcid"

    license_info = _license_type(article)
    if license_info is None:
        return None, "license"

    body = _extract_body(article.find(".//body"))
    if not body:
        return None, "no_body"

    license_name, license_url = license_info
    pmcid = pmcid_el.text.strip()
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    year_el = article.find(".//pub-date/year")

    record = {
        "pmcid": pmcid,
        "title": _text(article.find(".//article-title")),
        "abstract": _extract_abstract(article.find(".//abstract")),
        "body": body,
        "year": year_el.text.strip() if year_el is not None else None,
        "journal": _text(article.find(".//journal-title")),
        "license": license_name,
        "license_url": license_url,
        "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
    }
    return record, "ok"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_ids: set[str] = set()
    for query in QUERIES:
        ids = esearch(query, RETMAX_PER_QUERY)
        print(f"{query!r}: {len(ids)} candidates")
        all_ids.update(ids)
    print(f"Total unique candidate PMCIDs: {len(all_ids)}")

    id_list = sorted(all_ids)
    saved = 0
    skip_counts: dict[str, int] = {}

    for i in range(0, len(id_list), BATCH_SIZE):
        if saved >= TARGET_TOTAL:
            print(f"Reached target of {TARGET_TOTAL} articles, stopping early.")
            break

        batch = id_list[i : i + BATCH_SIZE]
        try:
            xml_text = efetch_batch(batch)
            root = ET.fromstring(xml_text)
        except (requests.RequestException, ET.ParseError) as e:
            print(f"Skipping batch {batch[0]}..{batch[-1]}: {e}")
            continue

        for article in root.findall(".//article"):
            record, reason = parse_article(article)
            if record is None:
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                continue
            out_path = RAW_DIR / f"{record['pmcid']}.json"
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved += 1

        print(f"Progress: saved={saved} skipped={skip_counts}")

    print(f"Done. Saved {saved} articles to {RAW_DIR}")
    print(f"Skip breakdown: {skip_counts}")


if __name__ == "__main__":
    main()
