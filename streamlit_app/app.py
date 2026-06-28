import json
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="HieroAnalysis", layout="wide")

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
    st.header("Input file")
    uploaded_file = st.file_uploader("Upload JSON file", type=["json"])

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
