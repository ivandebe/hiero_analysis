from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys
from typing import Iterable, List, Tuple

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

DEFAULT_INPUT_FILE = Path("output_data/prep_logs/german_translations.csv")
DEFAULT_OUTPUT_FILE = Path("output_data/topic/topic_german_egyptian.csv")
DEFAULT_TOPIC_TEXT_COLUMN = "german_translations"

GERMAN_STOPWORDS = {
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander", "andere",
    "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "auch", "auf",
    "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann", "das", "dass", "daß", "dein", "deine",
    "deinem", "deinen", "deiner", "deines", "dem", "den", "der", "des", "dessen", "deshalb", "die",
    "dies", "diese", "diesem", "diesen", "dieser", "dieses", "doch", "dort", "du", "durch", "ein",
    "eine", "einem", "einen", "einer", "eines", "er", "es", "euer", "eure", "eurem", "euren", "eurer",
    "eures", "für", "hatte", "hatten", "hattest", "hattet", "hier", "hinter", "ich", "ihr", "ihre",
    "ihrem", "ihren", "ihrer", "ihres", "euch", "im", "in", "ist", "ja", "jede", "jedem", "jeden",
    "jeder", "jedes", "jener", "jenes", "jetzt", "kann", "kannst", "können", "könnt", "machen", "mein",
    "meine", "meinem", "meinen", "meiner", "meines", "mit", "muß", "mußt", "musst", "müssen", "müßt",
    "nach", "nachdem", "nein", "nicht", "nun", "oder", "seid", "sein", "seine", "seinem", "seinen",
    "seiner", "seines", "selbst", "sich", "sie", "sind", "soll", "sollen", "sollst", "sollt", "sonst",
    "soweit", "sowie", "und", "unser", "unsere", "unserem", "unseren", "unseres", "unter", "vom", "von",
    "vor", "wann", "warum", "was", "weiter", "weitere", "wenn", "wer", "werde", "werden", "werdet",
    "weshalb", "wie", "wieder", "wieso", "wir", "wird", "wirst", "wo", "woher", "wohin", "zu", "zum",
    "zur", "über", "sein", "seine", "seiner", "seinem", "seinen", "seines", "ihm", "ihn", "ihrer",
    "ihren", "ihres", "seines", "seiner", "deren", "dessen", "diesem", "diesen", "dieser", "dieses"
}

CUSTOM_DOMAIN_STOPWORDS = {
    "unspecified", "destroyed", "zerstört", "zerstoert", "anfang", "rest", "zeile", "kommentar", "with",
    "commentary", "miscellaneous", "texts", "texts", "different", "periods", "places", "historical",
    "biographical", "project", "various", "frühzeitinschriften", "fruhzeitinschriften", "historisch",
    "biographische", "texte", "einzelner", "epochen", "orte", "objekte", "unklarer", "herkunft",
    "block", "abusir", "sakkara", "elephantine", "oberägypten", "oberaegypten", "dachla", "felsinschriften",
    "grabsteines", "fragment", "wiederverwendeter", "periode", "projekten", "nile", "contact"
}

ANNOTATION_PATTERNS = [
    r"http\S+|www\.\S+",
    r"\([^)]*unspecified[^)]*\)",
    r"\[[^\]]*\]",
    r"\{[^}]*\}",
    r"\b(?:PERSN|TITL|PN/\?|TEXT)\b",
    r"\b(?:person_name|title|substantive_fem|personal_pronoun|verb_3-inf|adjective|unedited|infl\.)\b",
    r"\b(?:noun|verb|adj|rel\.form|pron)\S*\b",
    r"\b[A-Z][a-z]*\.(?:[A-Za-z0-9:-]+\.)+[A-Za-z0-9:-]*\b",
    r"\b[A-Z]\\rel\.[A-Za-z0-9:.\\-]+\b",
    r"\b(?:sg|pl|stpr|ngem|masc|fem|pron|suffix|personal)\b",
    r"\b\d+(?:-\d+)?(?:/\d+)?\b",
    r"[⸢⸣〈〉\[\]{}]",
]

GERMAN_CHAR_PATTERN = re.compile(r"[A-Za-zÄÖÜäöüß]")
TRANSLITERATION_HEAVY_PATTERN = re.compile(r"[ꜣꜥḫḏḥḳḫṯẖšśṭḏỉȝꞽꜢ]", flags=re.IGNORECASE)
METADATA_SPLIT_PATTERN = re.compile(
    r"\b(?:miscellaneous texts|historical and biographical texts|texts from various projects|rock inscriptions|with commentary)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply BERTopic to German translations of ancient Egyptian texts and save topic assignments."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help=f"Path to input CSV file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Path to output CSV file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TOPIC_TEXT_COLUMN,
        help=f'Name of the text column containing German translations (default: "{DEFAULT_TOPIC_TEXT_COLUMN}")',
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=5,
        help="Minimum topic size for BERTopic (default: 5)",
    )
    parser.add_argument(
        "--nr-topics",
        default=None,
        help='Number of topics after reduction, e.g. 20 or "auto" (default: None)',
    )
    parser.add_argument(
        "--language",
        default="multilingual",
        help='BERTopic language setting, e.g. "german" or "multilingual" (default: multilingual)',
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=3,
        help="Maximum n-gram size for vectorization (default: 3)",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for vectorizer terms (default: 2)",
    )
    parser.add_argument(
        "--keep-intermediate-text",
        action="store_true",
        help="Keep cleaned topic text columns in the output CSV.",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=["full_clean", "translation_focus"],
        default="translation_focus",
        help=(
            "full_clean keeps all German-like cleaned text; translation_focus tries to retain the more fluent German "
            "translation segments and drop catalog/annotation-heavy material (default: translation_focus)."
        ),
    )
    return parser.parse_args()


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def validate_input_dataframe(df: pd.DataFrame, text_column: str) -> None:
    if text_column not in df.columns:
        raise ValueError(
            f'Input CSV does not contain the required column "{text_column}". '
            f"Available columns: {list(df.columns)}"
        )


def normalize_whitespace(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_annotation_patterns(text: str) -> str:
    cleaned = text
    for pattern in ANNOTATION_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return normalize_whitespace(cleaned)


def looks_like_good_german_segment(segment: str) -> bool:
    segment = normalize_whitespace(segment)
    if len(segment) < 20:
        return False
    if not GERMAN_CHAR_PATTERN.search(segment):
        return False
    translit_hits = len(TRANSLITERATION_HEAVY_PATTERN.findall(segment))
    alpha_chars = sum(ch.isalpha() for ch in segment)
    if alpha_chars == 0:
        return False
    if translit_hits / alpha_chars > 0.08:
        return False
    token_count = len(segment.split())
    return token_count >= 4


def score_german_segment(segment: str) -> tuple[int, int, int]:
    segment_l = segment.lower()
    german_hint_words = [
        " der ", " die ", " das ", " des ", " dem ", " den ", " und ", " seine ", " seiner ", " sein ",
        " frau ", " ehefrau ", " könig", " koenig", " priesterin ", " geliebte ", " tochter ", " herrin ",
        " würdige", " wuerdige", " verwalterin "
    ]
    hint_score = sum(1 for word in german_hint_words if word in f" {segment_l} ")
    token_count = len(segment.split())
    translit_penalty = len(TRANSLITERATION_HEAVY_PATTERN.findall(segment_l))
    return (hint_score, token_count, -translit_penalty)


def extract_translation_focused_text(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""

    chunks = METADATA_SPLIT_PATTERN.split(text)
    candidate_segments: List[str] = []

    for chunk in chunks:
        subsegments = re.split(r'(?<=[\.!?])\s+|\s+"|"\s+|\s+\.\.\.\s+', chunk)
        for seg in subsegments:
            seg = strip_annotation_patterns(seg)
            if looks_like_good_german_segment(seg):
                candidate_segments.append(seg)

    if candidate_segments:
        candidate_segments = sorted(
            dict.fromkeys(candidate_segments),
            key=score_german_segment,
            reverse=True,
        )
        top_segments = candidate_segments[:3]
        return normalize_whitespace(" ".join(top_segments).lower())

    fallback = strip_annotation_patterns(text).lower()
    fallback_tokens = []
    for token in fallback.split():
        if TRANSLITERATION_HEAVY_PATTERN.search(token):
            continue
        if not GERMAN_CHAR_PATTERN.search(token):
            continue
        fallback_tokens.append(token)
    return normalize_whitespace(" ".join(fallback_tokens))


def clean_full_german_text(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""

    text = strip_annotation_patterns(text).lower()
    tokens = []
    for token in text.split():
        token = token.strip("-–—:;,.'\"!?()/")
        if len(token) <= 1:
            continue
        if TRANSLITERATION_HEAVY_PATTERN.search(token):
            continue
        if not GERMAN_CHAR_PATTERN.search(token):
            continue
        tokens.append(token)
    return normalize_whitespace(" ".join(tokens))


def build_topic_input(text: str, extraction_mode: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    if extraction_mode == "translation_focus":
        return extract_translation_focused_text(text)
    return clean_full_german_text(text)


def prepare_documents(df: pd.DataFrame, text_column: str, extraction_mode: str) -> Tuple[pd.DataFrame, List[str]]:
    working_df = df.copy()
    working_df["_topic_input"] = working_df[text_column].apply(
        lambda x: build_topic_input(x, extraction_mode=extraction_mode)
    )
    working_df["_topic_input"] = working_df["_topic_input"].fillna("").astype(str)
    return working_df, working_df["_topic_input"].tolist()


def build_stopword_list(extra_stopwords: Iterable[str] | None = None) -> List[str]:
    combined = set(GERMAN_STOPWORDS) | set(CUSTOM_DOMAIN_STOPWORDS)
    if extra_stopwords:
        combined |= {str(word).strip().lower() for word in extra_stopwords if str(word).strip()}
    return sorted(combined)


def build_topic_model(min_topic_size: int, nr_topics, language: str, ngram_max: int, min_df: int) -> BERTopic:
    vectorizer_model = CountVectorizer(
        stop_words=build_stopword_list(),
        ngram_range=(1, max(1, ngram_max)),
        min_df=min_df,
        token_pattern=r"(?u)\b[a-zA-ZäöüÄÖÜß][a-zA-ZäöüÄÖÜß\-]{1,}\b",
    )

    model = BERTopic(
        language=language,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )
    return model


def _normalize_topic_phrase(phrase: str) -> str:
    phrase = phrase.lower().replace("_", " ").strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def _canonical_bow_key(phrase: str) -> tuple[str, ...]:
    tokens = _normalize_topic_phrase(phrase).split()
    return tuple(sorted(tokens))


def _shorten_topic_phrase(phrase: str) -> str:
    phrase = phrase.replace("_", " ")
    phrase = re.sub(r"\b(?:und|der|die|das|des|dem|den|mit|für|fuer|von|zu)\b", " ", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def get_topic_label_map(topic_model: BERTopic) -> dict[int, str]:
    topic_info = topic_model.get_topic_info()
    topic_label_map: dict[int, str] = {}

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])

        if topic_id == -1:
            topic_label_map[topic_id] = "outlier"
            continue

        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            topic_label_map[topic_id] = f"topic_{topic_id}"
            continue

        raw_terms = [word for word, _ in topic_words[:10]]
        dedup_terms = list(dict.fromkeys(raw_terms))

        selected_terms = []
        seen_keys = set()
        for term in dedup_terms:
            bow_key = _canonical_bow_key(term)
            if bow_key not in seen_keys:
                seen_keys.add(bow_key)
                selected_terms.append(term)

        display_terms = [_shorten_topic_phrase(term) for term in selected_terms[:5]]
        display_terms = [term for term in display_terms if term]
        label = ", ".join(display_terms) if display_terms else f"topic_{topic_id}"
        topic_label_map[topic_id] = label

    return topic_label_map


def assign_topics(df: pd.DataFrame, docs: List[str], topic_model: BERTopic) -> Tuple[pd.DataFrame, BERTopic]:
    result_df = df.copy()
    result_df["topic"] = -1
    result_df["topic_label"] = "outlier"

    valid_mask = result_df["_topic_input"].str.strip().ne("")
    valid_docs = [doc for doc, is_valid in zip(docs, valid_mask.tolist()) if is_valid]

    if len(valid_docs) == 0:
        print("No valid non-empty cleaned texts found. Writing output with topic = -1 and topic_label = 'outlier' for all rows.")
        return result_df, topic_model

    topics, _ = topic_model.fit_transform(valid_docs)
    result_df.loc[valid_mask, "topic"] = topics

    topic_label_map = get_topic_label_map(topic_model)
    result_df["topic_label"] = result_df["topic"].map(topic_label_map).fillna("unknown")
    return result_df, topic_model


def save_outputs(df: pd.DataFrame, output_path: Path, keep_intermediate_text: bool) -> None:
    final_df = df.copy()
    if not keep_intermediate_text:
        final_df = final_df.drop(columns=["_topic_input"], errors="ignore")
    ensure_parent_dir(output_path)
    final_df.to_csv(output_path, index=False)


def run_topic_analysis_dataframe(
    df: pd.DataFrame,
    text_column: str = DEFAULT_TOPIC_TEXT_COLUMN,
    min_topic_size: int = 5,
    nr_topics=None,
    language: str = "multilingual",
    ngram_max: int = 3,
    min_df: int = 2,
    extraction_mode: str = "translation_focus",
    keep_intermediate_text: bool = False,
    output_path: Path | None = None,
) -> pd.DataFrame:
    validate_input_dataframe(df, text_column)

    if nr_topics is not None and nr_topics != "auto":
        try:
            nr_topics = int(nr_topics)
        except ValueError as exc:
            raise ValueError('--nr-topics must be an integer, "auto", or None') from exc

    df_prepared, docs = prepare_documents(
        df=df,
        text_column=text_column,
        extraction_mode=extraction_mode,
    )

    topic_model = build_topic_model(
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        language=language,
        ngram_max=ngram_max,
        min_df=min_df,
    )

    df_with_topics, _ = assign_topics(
        df=df_prepared,
        docs=docs,
        topic_model=topic_model,
    )

    if not keep_intermediate_text:
        df_with_topics = df_with_topics.drop(columns=["_topic_input"], errors="ignore")

    if output_path is not None:
        save_outputs(df_with_topics, output_path, keep_intermediate_text=keep_intermediate_text)

    return df_with_topics


def print_topic_summary(df: pd.DataFrame) -> None:
    summary = (
        df[["topic", "topic_label"]]
        .value_counts(dropna=False)
        .reset_index(name="count")
        .sort_values(["topic", "count"], ascending=[True, False])
    )
    print("\nTopic counts:")
    print(summary.to_string(index=False))


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    text_column = args.text_column
    min_topic_size = args.min_topic_size
    nr_topics = args.nr_topics
    language = args.language
    ngram_max = args.ngram_max
    min_df = args.min_df
    extraction_mode = args.extraction_mode

    if nr_topics is not None and nr_topics != "auto":
        try:
            nr_topics = int(nr_topics)
        except ValueError as exc:
            raise ValueError('--nr-topics must be an integer, "auto", or omitted') from exc

    print(f"Reading input CSV: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    validate_input_dataframe(df, text_column)

    print(f"Loaded {len(df)} rows.")
    print(f'Using text column: "{text_column}"')
    print(f"Extraction mode: {extraction_mode}")

    df_prepared, docs = prepare_documents(
        df=df,
        text_column=text_column,
        extraction_mode=extraction_mode,
    )

    topic_model = build_topic_model(
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        language=language,
        ngram_max=ngram_max,
        min_df=min_df,
    )

    df_with_topics, _ = assign_topics(
        df=df_prepared,
        docs=docs,
        topic_model=topic_model,
    )

    output_df = run_topic_analysis_dataframe(
        df=df,
        text_column=text_column,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        language=language,
        ngram_max=ngram_max,
        min_df=min_df,
        extraction_mode=extraction_mode,
        keep_intermediate_text=args.keep_intermediate_text,
        output_path=output_path,
    )

    print(f"Saved output CSV to: {output_path}")
    print_topic_summary(output_df)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
