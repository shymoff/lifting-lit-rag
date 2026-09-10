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
RETMAX_PER_QUERY = 60
TARGET_TOTAL = 250

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ALLOWED_LICENSES = {
    "by": "CC BY",
    "by-sa": "CC BY-SA",
    "publicdomain/zero": "CC0",
}

QUERIES = [
    "resistance training[Title/Abstract] AND hypertrophy[Title/Abstract]",
    "resistance training[Title/Abstract] AND training volume[Title/Abstract]",
    "protein intake[Title/Abstract] AND muscle protein synthesis[Title/Abstract]",
    "resistance training[Title/Abstract] AND training frequency[Title/Abstract]",
    "velocity based training[Title/Abstract]",
    "resistance training[Title/Abstract] AND training to failure[Title/Abstract]",
    "resistance training[Title/Abstract] AND range of motion[Title/Abstract]",
    "concurrent training[Title/Abstract] AND interference effect[Title/Abstract]",
    "creatine supplementation[Title/Abstract] AND strength[Title/Abstract]",
    "blood flow restriction[Title/Abstract] AND resistance training[Title/Abstract]",
    "rest interval[Title/Abstract] AND resistance training[Title/Abstract]",
    "resistance training[Title/Abstract] AND older adults[Title/Abstract]",
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

    body = _text(article.find(".//body"))
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
        "abstract": _text(article.find(".//abstract")),
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
