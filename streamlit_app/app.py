import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scraper.find_content_from_sentenceids import download_sentence_contents
from scraper.find_sentenceids_by_lemmaid import download_tla_lemma_sentences
from utils.upload_json_sentence_contents_into_postgres import upload_sentence_contents_to_postgres

st.set_page_config(page_title="HieroAnalysis", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[1]
LEMMA = "ḥm.t"


def load_json(uploaded_file):
    return json.load(uploaded_file)


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


st.title("HieroAnalysis")

with st.sidebar:
    app_mode = st.selectbox("Select mode", ["Scraper", "Lemma Analysis"])
    scrape_button = False

    if app_mode == "Scraper":
        lemma_id = st.number_input("Lemma ID", min_value=1, value=125040, step=1, format="%d")
        postgres_conn_string = os.getenv("POSTGRES_CONN_STRING", "")
        if not postgres_conn_string:
            st.warning("POSTGRES_CONN_STRING is not set. Set it in the environment to upload results to Postgres.")
        scrape_button = st.button("scrape website")
    else:
        st.header("Input file")
        uploaded_file = st.file_uploader("Upload JSON file", type=["json"])

if app_mode == "Scraper":
    if scrape_button:
        data_dir = ROOT_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        ids_file = data_dir / f"lemma_{lemma_id}_sentence_ids.json"
        contents_file = data_dir / f"lemma_{lemma_id}_sentence_contents.json"

        with st.spinner("Downloading sentence ids..."):
            try:
                ids_payload = download_tla_lemma_sentences(lemma_id, out_json=str(ids_file))
                st.success(f"Downloaded {ids_payload.get('returned', 0)} sentence ids.")

                contents_payload = download_sentence_contents(str(ids_file), str(contents_file))
                st.success(f"Downloaded contents for {contents_payload.get('returned', 0)} sentences.")

                if not postgres_conn_string:
                    raise ValueError("POSTGRES_CONN_STRING is not configured for Postgres upload.")

                upload_sentence_contents_to_postgres(
                    postgres_conn_string=postgres_conn_string,
                    json_file_path=str(contents_file),
                    table_name="hiero_sentence_contents",
                )
                st.success("Uploaded sentence contents to Postgres.")

                st.markdown("**Output files:**")
                st.write(f"- {ids_file}")
                st.write(f"- {contents_file}")
            except Exception as exc:
                st.error(f"Scraper failed: {exc}")
    else:
        st.info("Enter a lemma ID and click 'scrape website' to download sentence ids, sentence contents, and upload them to Postgres.")
    st.stop()

if uploaded_file is None:
    st.info("Upload a JSON file from the sidebar to start the analysis.")
    st.stop()

try:
    payload = load_json(uploaded_file)
except Exception as e:
    st.error(f"Could not read the JSON file: {e}")
    st.stop()

returned = payload.get("returned", 0)
source_json = payload.get("source_json", "Unknown")
items = payload.get("items", [])

col1, col2, col3 = st.columns(3)
col1.metric("Returned", returned)
col2.metric("Source JSON", source_json)
col3.metric("Lemma", LEMMA)


df_sentences = build_sentences_df(items)

with st.expander(f"Show all transliteration sentences ({len(df_sentences)})", expanded=False):
    st.dataframe(df_sentences, width="stretch", hide_index=True)

neighbor_counts = extract_neighbors(items, LEMMA)
df_top10 = build_top10_df(neighbor_counts)

st.subheader(f"Top 10 transliteration items closest to {LEMMA}")

if df_top10.empty:
    st.warning("No matching transliteration neighbors found for the selected lemma.")
else:
    fig = px.bar(
        df_top10.sort_values("count", ascending=True),
        x="count",
        y="token",
        orientation="h",
        text="count",
        title=f"Top 10 items immediately before or after {LEMMA}",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Token",
        height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, width="stretch")
    st.dataframe(df_top10, width="stretch", hide_index=True)
