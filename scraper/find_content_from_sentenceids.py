import json
import time
import re
import requests
from bs4 import BeautifulSoup

BASE = "https://thesaurus-linguae-aegyptiae.de"

SOURCE_PREFIXES = (
    "Tomb Inscriptions", "Funerary Texts", "Hymns", "Mythological Texts",
    "Wisdom Texts", "Letters", "Narrative Texts", "Religious Texts",
    "Historical Texts"
)

STOP_PREFIXES = (
    "Dating (time frame)", "Metadata of the text", "Relations", "Author(s)",
    "Sentence no.", "Please cite as", "Hieroglyphs"
)

SKIP_EXACT = {"Copy token ID", "Copy token URL", ":", "de", "en", "fr", "ع"}

def _clean(s):
    return re.sub(r"\s+", " ", s).strip() if s else ""

def _extract_sentence_id(url):
    m = re.search(r"/sentence/([^/?#]+)", url)
    return m.group(1) if m else ""

def _build_sentence_url(sentence_id):
    return f"{BASE}/sentence/{sentence_id}"

def _visible_lines(soup):
    return [ln.strip() for ln in soup.body.get_text(separator="\n", strip=True).splitlines() if ln.strip()]

def _is_source_line(line):
    return any(line.startswith(prefix) for prefix in SOURCE_PREFIXES)

def _extract_transliteration(lines):
    start = None
    for i, line in enumerate(lines):
        if line == "Copy URL":
            start = i + 1
            break

    if start is None:
        return []

    out = []
    for line in lines[start:]:
        if line == "Copy token ID":
            break
        if line in {"Copy token URL", "word", "Glyphs artificially arranged"}:
            continue
        cleaned = _clean(line)
        if cleaned:
            out.append(cleaned)

    return out

def _extract_german_translation(lines):
    de_idx = None
    for i, line in enumerate(lines):
        if line == "de":
            de_idx = i
            break

    if de_idx is None:
        return ""

    trans = []
    for line in lines[de_idx + 1:]:
        if _is_source_line(line):
            break
        if any(line.startswith(prefix) for prefix in STOP_PREFIXES):
            break
        if line in SKIP_EXACT:
            continue
        if line.startswith((
            "Sentence token ID", "Lemma word class", "Lemma translation",
            "Gram. tagging", "Linguistic glossing", "Translation",
            "Dating of Text", "Grammar"
        )):
            continue
        trans.append(line)

    return _clean(" ".join(trans))

def _extract_sources(lines):
    sources = []
    start = None
    for i, line in enumerate(lines):
        if line == "de":
            for j in range(i + 1, len(lines)):
                x = lines[j]
                if _is_source_line(x):
                    start = j
                    break
            if start is not None:
                break

    if start is None:
        return sources

    for line in lines[start:]:
        if any(line.startswith(prefix) for prefix in STOP_PREFIXES):
            break
        if line in SKIP_EXACT:
            continue
        if line and not line.startswith(("Sentence token ID", "Lemma word class", "Grammar")):
            sources.append(_clean(line))

    return sources

def _extract_dating(lines):
    for i, line in enumerate(lines):
        if line == "Dating (time frame)":
            for j in range(i + 1, min(i + 6, len(lines))):
                x = lines[j]
                if x == ":":
                    continue
                if x:
                    return _clean(x)
    return ""

def extract_sentence_page(url, session=None, timeout=30):
    s = session or requests.Session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    lines = _visible_lines(soup)

    return {
        "url": url,
        "sentence_id": _extract_sentence_id(url),
        "transliteration": _extract_transliteration(lines),
        "german_translation": _extract_german_translation(lines),
        "sources": _extract_sources(lines),
        "dating": _extract_dating(lines),
    }

def download_sentence_contents(input_json_file, output_json_file="tla_sentence_contents.json", sleep_seconds=0.2):
    with open(input_json_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("items", [])
    if not isinstance(items, list):
        items = []

    sentence_ids = payload.get("sentence_ids", [])
    if isinstance(sentence_ids, list) and sentence_ids and not items:
        items = [{"sentence_id": sid} for sid in sentence_ids]

    session = requests.Session()
    out_items = []

    for item in items:
        if isinstance(item, str):
            sentence_id = item
            url = _build_sentence_url(sentence_id)
        else:
            url = item.get("url", "")
            sentence_id = item.get("sentence_id", "") or _extract_sentence_id(url)
            if not url and sentence_id:
                url = _build_sentence_url(sentence_id)

        if not url:
            continue

        try:
            out_items.append(extract_sentence_page(url, session=session))
        except Exception as e:
            out_items.append({
                "url": url,
                "sentence_id": sentence_id or _extract_sentence_id(url),
                "error": str(e),
            })
        if sleep_seconds:
            time.sleep(sleep_seconds)

    out = {
        "source_json": input_json_file,
        "returned": len(out_items),
        "items": out_items,
    }

    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out



# # Example:
# data = download_sentence_contents(
#     "lemma_104730_urls.json",
#     "lemma_104730_sentence_contents.json"
# )

# data = download_sentence_contents(
#     "lemma_104730_urls_short.json",
#     "lemma_104730_sentence_contents_short.json"
# )

# # Example:
# data = download_sentence_contents(
#     "lemma_125370_urls.json",
#     "lemma_125370_sentence_contents.json"
# )

if __name__ == "__main__":
    data = download_sentence_contents(
        "lemma_125040_urls.json",
        "lemma_125040_sentence_contents.json"
    )
