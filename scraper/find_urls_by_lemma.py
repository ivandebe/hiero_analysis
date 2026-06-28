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

def download_tla_lemma_sentences(url_or_lemma_id, out_json="tla_lemma_sentences.json", sleep_seconds=0.2):
    session = requests.Session()
    url = _normalize_tla_sentence_search(url_or_lemma_id)

    first = session.get(url, timeout=30)
    first.raise_for_status()
    soup = BeautifulSoup(first.text, "html.parser")

    text = soup.get_text(" ", strip=True)
    m = re.search(r"of\s+([0-9,]+)\s+sentences", text)
    if not m:
        raise RuntimeError("Could not detect total sentence count.")
    total_expected = int(m.group(1).replace(",", ""))

    items = []
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
            title = a.get_text(" ", strip=True)
            if not href or not title:
                continue

            abs_url = href if href.startswith("http") else BASE + href
            if "/sentence/" in abs_url:
                key = abs_url
                if key not in seen:
                    seen.add(key)
                    item = {"title": title, "url": abs_url}
                    items.append(item)
                    page_items.append(item)

        if len(items) >= total_expected:
            break
        if not page_items:
            break

        page += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)

    payload = {
        "source": url,
        "total_expected": total_expected,
        "returned": len(items),
        "items": items,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload

if __name__ == "__main__":
    data = download_tla_lemma_sentences(104730, out_json="lemma_104730_urls.json")
