from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import requests
from bs4 import BeautifulSoup
import json
import time
import re

BASE = "https://thesaurus-linguae-aegyptiae.de"

def _normalize_tla_sentence_search(url_or_lemma_id):
    if str(url_or_lemma_id).isdigit():
        return f"{BASE}/search/sentence?tokens%5B0%5D.lemma.id={url_or_lemma_id}&sort="
    url = str(url_or_lemma_id)
    if "/search/sentence" not in url:
        raise ValueError("Expected a TLA sentence-search URL or a lemma ID.")
    return url

def _set_query_param(url, **params):
    p = urlparse(url)
    q = parse_qs(p.query, keep_blank_values=True)
    for k, v in params.items():
        q[k] = [str(v)]
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), p.fragment))

def _extract_lemma_id(url_or_lemma_id):
    if str(url_or_lemma_id).isdigit():
        return str(url_or_lemma_id)

    url = str(url_or_lemma_id)
    parsed = urlparse(url)
    q = parse_qs(parsed.query)

    lemma_ids = q.get("tokens[0].lemma.id")
    if lemma_ids:
        return str(lemma_ids[0])

    m = re.search(r"tokens(?:%5B|\\[)0(?:%5D|\\])\.lemma\.id=(\d+)", url)
    if m:
        return m.group(1)

    raise ValueError("Could not extract lemma ID from URL.")

def _get_lemma_transliteration(session, lemma_id):
    lemma_url = f"{BASE}/lemma/{lemma_id}"
    r = session.get(lemma_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title_text = soup.title.get_text(" ", strip=True) if soup.title else ""
    marker = f"(Lemma ID {lemma_id})"

    if marker in title_text:
        translit = title_text.split(marker)[0].strip().strip('"').strip()
        if translit:
            return translit

    full_text = soup.get_text("\n", strip=True)
    for line in full_text.splitlines():
        line = line.strip()
        if marker in line:
            translit = line.split(marker)[0].strip().strip('"').strip()
            if translit:
                return translit

    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if "Please cite as" in line:
            window = " ".join(lines[i:i+3])
            q1 = window.find('"')
            q2 = window.find('"', q1 + 1) if q1 != -1 else -1
            if q1 != -1 and q2 != -1:
                translit = window[q1+1:q2].strip()
                if translit:
                    return translit

    return None

def _extract_sentence_id_from_url(url):
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] if path else None

def download_tla_lemma_sentences(url_or_lemma_id, out_json="tla_lemma_sentences.json", sleep_seconds=0.2):
    session = requests.Session()
    url = _normalize_tla_sentence_search(url_or_lemma_id)
    lemma_id = _extract_lemma_id(url_or_lemma_id)
    transliteration = _get_lemma_transliteration(session, lemma_id)

    first = session.get(url, timeout=30)
    first.raise_for_status()
    soup = BeautifulSoup(first.text, "html.parser")

    text = soup.get_text(" ", strip=True)
    m = re.search(r"of\s+([0-9,]+)\s+sentences", text)
    if not m:
        raise RuntimeError("Could not detect total sentence count.")
    total_expected = int(m.group(1).replace(",", ""))

    sentence_ids = []
    seen = set()
    page = 1

    while True:
        page_url = _set_query_param(url, page=page) if page > 1 else url
        r = session.get(page_url, timeout=30)
        r.raise_for_status()
        sp = BeautifulSoup(r.text, "html.parser")

        page_items = []
        for a in sp.select("a[href]"):
            href = a.get("href", "")
            if not href:
                continue

            abs_url = href if href.startswith("http") else BASE + href
            if "/sentence/" in abs_url:
                sentence_id = _extract_sentence_id_from_url(abs_url)
                if sentence_id and sentence_id not in seen:
                    seen.add(sentence_id)
                    sentence_ids.append(sentence_id)
                    page_items.append(sentence_id)

        if len(sentence_ids) >= total_expected:
            break
        if not page_items:
            break

        page += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)

    payload = {
        "source": url,
        "lemma_id": int(lemma_id),
        "transliteration": transliteration,
        "total_expected": total_expected,
        "returned": len(sentence_ids),
        "sentence_ids": sentence_ids,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload

if __name__ == "__main__":
    data = download_tla_lemma_sentences(125040, out_json="lemma_125040_sentence_ids.json")
    # print(json.dumps(data, ensure_ascii=False, indent=2))
