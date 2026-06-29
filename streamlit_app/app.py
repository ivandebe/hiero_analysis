import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scraper.find_content_from_sentenceids import download_sentence_contents
from scraper.find_sentenceids_by_lemmaid import download_tla_lemma_sentences
import tempfile
import io
import zipfile
from utils.upload_json_sentence_contents_into_postgres import upload_sentence_contents_to_postgres
from utils.upload_json_lemmas import upload_lemmas_to_postgres

st.set_page_config(page_title="HieroAnalysis", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[1]
LEMMA = "ḥm.t"


def load_json(uploaded_file):
    return json.load(uploaded_file)


def fetch_lemma_from_postgres(postgres_conn_string: str, lemma_id: str):
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise ImportError(
            "psycopg is required to query Postgres. Install psycopg and retry."
        ) from exc

    query = sql.SQL(
        "SELECT lemma_id, transliteration, sentences_id FROM hiero_lemmas WHERE lemma_id = %s"
    )

    with psycopg.connect(postgres_conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (lemma_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "lemma_id": row[0],
                "transliteration": row[1],
                "sentences_id": row[2] or [],
            }


def extract_neighbors(items, lemma):
    counter = Counter()

    for item in items:
        translit = item.get("transliteration", [])
        if not isinstance(translit, list):
            continue

        cleaned = [str(x).strip() for x in translit if str(x).strip()]

        for i, token in enumerate(cleaned):
            if token == lemma:
                if i - 1 >= 0:
                    counter[cleaned[i - 1]] += 1
                if i + 1 < len(cleaned):
                    counter[cleaned[i + 1]] += 1

    return counter


def build_top10_df(counter):
    top10 = counter.most_common(10)
    if not top10:
        return pd.DataFrame(columns=["token", "count"])
    return pd.DataFrame(top10, columns=["token", "count"])


def build_sentences_df(items):
    rows = []

    for item in items:
        translit = item.get("transliteration", [])
        if not isinstance(translit, list):
            translit = []

        cleaned = [str(x).strip() for x in translit if str(x).strip()]
        sentence_text = " ".join(cleaned)

        rows.append(
            {
                "sentence_id": item.get("sentence_id", ""),
                "transliteration_sentence": sentence_text,
            }
        )

    return pd.DataFrame(rows)


st.title("Hieroglyphic Analysis")

with st.sidebar:
    images_dir = Path(__file__).parent / "images"
    logo_path = images_dir / "heka2.png"
    if logo_path.exists():
        st.image(str(logo_path), width=300)
    else:
        st.warning("Logo image not found.")
    app_mode = st.selectbox("Select mode", ["Scraper", "Lemma Analysis"])
    lemma_id = st.number_input("Lemma ID", min_value=1, value=125040, step=1, format="%d")
    postgres_conn_string = os.getenv("POSTGRES_CONN_STRING", "")
    if not postgres_conn_string:
        st.warning("POSTGRES_CONN_STRING is not set. Set it in the environment to upload results to Postgres.")

    scrape_button = False
    analyze_button = False
    if app_mode == "Scraper":
        scrape_button = st.button("Scrape Thesaurus Linguae Aegyptiae")
    else:
        analyze_button = st.button("Analyze Lemma")

if app_mode == "Scraper":
    if scrape_button:
        # Use temporary files so we don't persist files into the project folder.
        tmp_ids = tempfile.NamedTemporaryFile(delete=False, suffix=f"_lemma_{lemma_id}_sentence_ids.json")
        tmp_ids_path = tmp_ids.name
        tmp_ids.close()

        tmp_contents = tempfile.NamedTemporaryFile(delete=False, suffix=f"_lemma_{lemma_id}_sentence_contents.json")
        tmp_contents_path = tmp_contents.name
        tmp_contents.close()

        with st.spinner("Downloading sentence ids..."):
            try:
                ids_payload = download_tla_lemma_sentences(lemma_id, out_json=str(tmp_ids_path))
                st.success(f"Downloaded {ids_payload.get('returned', 0)} sentence ids.")

                if postgres_conn_string:
                    upload_lemmas_to_postgres(
                        postgres_conn_string=postgres_conn_string,
                        json_file_path=str(tmp_ids_path),
                        table_name="hiero_lemmas",
                    )
                    st.success("Uploaded lemma to Postgres.")

                contents_payload = download_sentence_contents(str(tmp_ids_path), str(tmp_contents_path))
                st.success(f"Downloaded contents for {contents_payload.get('returned', 0)} sentences.")

                # Create an in-memory ZIP containing both payloads and provide single download button
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
                    z.writestr(f"lemma_{lemma_id}_sentence_ids.json", json.dumps(ids_payload, ensure_ascii=False, indent=2))
                    z.writestr(f"lemma_{lemma_id}_sentence_contents.json", json.dumps(contents_payload, ensure_ascii=False, indent=2))
                zip_buf.seek(0)
                st.download_button(
                    label="Download both payloads (ZIP)",
                    data=zip_buf.getvalue(),
                    file_name=f"lemma_{lemma_id}_payloads.zip",
                    mime="application/zip",
                )

                if postgres_conn_string:
                    upload_sentence_contents_to_postgres(
                        postgres_conn_string=postgres_conn_string,
                        json_file_path=str(tmp_contents_path),
                        table_name="hiero_sentence_contents",
                    )
                    st.success("Uploaded sentence contents to Postgres.")

                # Clean up temporary files
                try:
                    os.unlink(tmp_ids_path)
                except Exception:
                    pass
                try:
                    os.unlink(tmp_contents_path)
                except Exception:
                    pass

            except Exception as exc:
                st.error(f"Scraper failed: {exc}")
    else:
        st.info("Enter a lemma ID and click 'scrape website' to download sentence ids, sentence contents, and upload them to Postgres.")
    st.stop()

if app_mode == "Lemma Analysis":
    if analyze_button:
        if not postgres_conn_string:
            st.error("POSTGRES_CONN_STRING is not configured. Set it in the environment to query Postgres.")
            st.stop()

        try:
            lemma_row = fetch_lemma_from_postgres(postgres_conn_string, str(lemma_id))
        except Exception as e:
            st.error(f"Could not query Postgres: {e}")
            st.stop()

        if lemma_row is None:
            st.warning(f"No lemma found for lemma_id={lemma_id} in hiero_lemmas.")
            st.stop()

        sentence_ids = lemma_row.get("sentences_id") or []
        col1, col2, col3 = st.columns(3)
        col1.metric("Lemma ID", lemma_row.get("lemma_id", ""))
        col2.metric("Transliteration", lemma_row.get("transliteration", ""))
        col3.metric("Sentence count", len(sentence_ids))

        st.subheader("Sentence IDs")
        st.write(sentence_ids)
        st.stop()
    else:
        st.info("Enter a lemma ID and click 'Analyze Lemma' to query hiero_lemmas.")
        st.stop()

