import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scraper.find_content_from_sentenceids import download_sentence_contents
from scraper.find_sentenceids_by_lemmaid import download_tla_lemma_sentences
import tempfile
import io
import zipfile
from utils.upload_json_sentence_contents_into_postgres import upload_sentence_contents_to_postgres
from utils.upload_json_lemmas import upload_lemmas_to_postgres
from utils.lemma_cooccurrence_plotly import (
    create_lemma_cooccurrence_figure,
    create_lemma_cooccurrence_tables,
    )

st.set_page_config(page_title="HieroAnalysis", layout="wide")


ALLOWED_EMAILS = {
    email.strip().lower()
    for email in st.secrets["users_permissions"].get("allowed_emails", [])
}

def login_screen():
    st.title("Private app")
    st.write("Please sign in with Google to continue.")
    st.button("Log in with Google", on_click=st.login)

if not st.user.is_logged_in:
    login_screen()
    st.stop()

user_email = str(st.user.get("email", "")).strip().lower()
user_name = str(st.user.get("name", "")).strip()

if user_email not in ALLOWED_EMAILS:
    st.error("Your account is not authorized for this app.")
    st.write(f"Signed in as: {user_email or 'unknown email'}")
    st.button("Log out", on_click=st.logout)
    st.stop()

st.sidebar.success(f"Signed in as {user_email}")
st.sidebar.button("Log out", on_click=st.logout)
st.sidebar.divider()


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


def fetch_lemma_ids_from_postgres(postgres_conn_string: str):
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise ImportError(
            "psycopg is required to query Postgres. Install psycopg and retry."
        ) from exc

    query = sql.SQL(
        "SELECT DISTINCT lemma_id FROM hiero_lemmas ORDER BY lemma_id"
    )

    with psycopg.connect(postgres_conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]


def fetch_sentence_contents_from_postgres(postgres_conn_string: str, sentence_ids: list):
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise ImportError(
            "psycopg is required to query Postgres. Install psycopg and retry."
        ) from exc

    if not sentence_ids:
        return []

    query = sql.SQL(
        "SELECT url, sentence_id, transliteration, german_translation, sources, dating "
        "FROM hiero_sentence_contents WHERE sentence_id = ANY(%s)"
    )

    with psycopg.connect(postgres_conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (sentence_ids,))
            rows = cur.fetchall()
            return [
                {
                    "url": row[0],
                    "sentence_id": row[1],
                    "transliteration": row[2] or [],
                    "german_translation": row[3] or "",
                    "sources": row[4] or [],
                    "dating": row[5] or "",
                }
                for row in rows
            ]


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
    app_mode = st.selectbox("Select mode", ["Lemma Analysis", "Scraper"])
    postgres_conn_string = st.secrets.get("database", {}).get("POSTGRES_CONN_STRING") or os.getenv("POSTGRES_CONN_STRING", "")
    if not postgres_conn_string:
        st.warning("POSTGRES_CONN_STRING is not set. Set it in the environment to upload results to Postgres.")

    if app_mode == "Lemma Analysis":
        lemma_id_options = []
        if postgres_conn_string:
            try:
                lemma_id_options = fetch_lemma_ids_from_postgres(postgres_conn_string)
            except Exception as e:
                st.error(f"Could not load lemma ids from Postgres: {e}")
                lemma_id_options = []

        if lemma_id_options:
            lemma_id = st.selectbox("Lemma ID", lemma_id_options)
        else:
            lemma_id = st.selectbox("Lemma ID", ["No lemma IDs available"])
            if postgres_conn_string:
                st.warning("No lemmas found in hiero_lemmas.")
            else:
                st.warning("Connect POSTGRES_CONN_STRING to load lemma IDs.")

    else:
        lemma_id = st.number_input("Lemma ID", min_value=1, value=125040, step=1, format="%d")
        max_lemmas = 40
        closeness = 3

    scrape_button = False
    analyze_button = False
    if app_mode == "Scraper":
        scrape_button = st.button("Scrape Thesaurus Linguae Aegyptiae")
    else:
        analyze_button = st.button("Analyze Lemma")

    if app_mode == "Lemma Analysis" and analyze_button:
        st.session_state.analysis_ready = True
        st.session_state.analysis_lemma_id = lemma_id
        st.session_state.graph_params = {}

if app_mode == "Lemma Analysis":
    if "analysis_ready" not in st.session_state:
        st.session_state.analysis_ready = False
    if "analysis_lemma_id" not in st.session_state:
        st.session_state.analysis_lemma_id = None
    if "graph_params" not in st.session_state:
        st.session_state.graph_params = {}

    if st.session_state.analysis_lemma_id != lemma_id:
        st.session_state.analysis_ready = False

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
    if st.session_state.analysis_ready:
        if not postgres_conn_string:
            st.error("POSTGRES_CONN_STRING is not configured. Set it in the environment to query Postgres.")
            st.stop()

        if lemma_id == "No lemma IDs available":
            st.error("No lemma_id options are available for analysis.")
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

        if not sentence_ids:
            st.warning("This lemma has no sentence_ids stored in hiero_lemmas.")
            st.stop()

        try:
            master_content_rows = fetch_sentence_contents_from_postgres(postgres_conn_string, sentence_ids)
        except Exception as e:
            st.error(f"Could not query hiero_sentence_contents: {e}")
            st.stop()

        master_content_df = pd.DataFrame(master_content_rows)
        st.subheader("Master sentence contents")
        st.dataframe(master_content_df[['sentence_id', 'transliteration','german_translation', 'sources', 'dating']], width="stretch")

        df = master_content_df[master_content_df["transliteration"].apply(lambda x: isinstance(x, list) and lemma_row.get("transliteration") in x)]

        tab1, tab2, tab3 = st.tabs(["Co-Occurrence Analysis", "Period Analysis", "Topic Analysis"])

        with tab1:
            st.markdown("### Graph settings")
            with st.form("generate_graph_form"):
                max_lemmas = st.slider(
                    "Max number of lemmas",
                    min_value=5,
                    max_value=100,
                    value=st.session_state.graph_params.get("max_lemmas", 40),
                    step=5,
                    help="Limit the number of lemmas shown in the co-occurrence graph.",
                )
                closeness = st.slider(
                    "Closeness from core lemma",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.graph_params.get("closeness", 3),
                    step=1,
                    help="Include lemmas up to this many positions away from the core lemma in each sentence.",
                )
                generate_graph = st.form_submit_button("Generate graph")

            if generate_graph:
                st.session_state.graph_params = {
                    "max_lemmas": max_lemmas,
                    "closeness": closeness,
                }

            if st.session_state.graph_params:
                max_lemmas = st.session_state.graph_params["max_lemmas"]
                closeness = st.session_state.graph_params["closeness"]

                st.subheader("Co-Occurrence Graph")
                st.markdown(
                    """
                    This graph shows how lemmas co-occur within sentences that contain the selected core lemma.

                    - **Nodes** represent lemmas.
                    - **Edges** connect lemmas that appear together in the same sentence.
                    - **Weight** shows how many sentences contain both lemmas.
                    - **Closeness** limits the chart to lemmas appearing within the specified number of positions left or right of the core lemma.
                    - **Max lemmas** limits the number of lemmas shown based on frequency.
                    """
                )

                fig = create_lemma_cooccurrence_figure(
                    df,
                    deduplicate_lemmas_per_sentence=True,
                    min_edge_weight=1,
                    max_lemmas=max_lemmas,
                    core_lemma=lemma_row.get("transliteration"),
                    closeness=closeness,
                    title="Lemma co-occurrence by sentence",
                )

                st.plotly_chart(fig, width="stretch")

                nodes_df, edges_df = create_lemma_cooccurrence_tables(
                    df,
                    deduplicate_lemmas_per_sentence=True,
                    min_edge_weight=1,
                    max_lemmas=max_lemmas,
                    core_lemma=lemma_row.get("transliteration"),
                    closeness=closeness,
                )

                st.subheader("Top 10 Co-Occurrence Lemmas")
                st.write(f"Showing lemmas that co-occur with '{lemma_row.get('transliteration')}' within {closeness} positions in sentences.")
                st.dataframe(nodes_df.sort_values("sentence_count", ascending=False).head(10), width="stretch")

            else:
                st.info("Select settings above and click Generate graph to display the co-occurrence chart.")

        with tab2:
            st.subheader("Period Analysis")
            st.markdown(
                """
                This view shows the distribution of sentence datings for the selected lemma.
                The horizontal bar chart counts how many sentences are assigned to each dating value.
                """
            )
            dating_counts = (
                master_content_df["dating"]
                .replace({None: "Unknown", "": "Unknown"})
                .fillna("Unknown")
                .value_counts()
                .rename_axis("dating")
                .reset_index(name="count")
            )
            if dating_counts.empty:
                st.warning("No dating values are available for this lemma.")
            else:
                dating_counts = dating_counts.sort_values("count", ascending=True)
                dating_fig = px.bar(
                    dating_counts,
                    x="count",
                    y="dating",
                    orientation="h",
                    title="Sentence counts by dating",
                    labels={"count": "Sentence count", "dating": "Dating"},
                )
                dating_fig.update_layout(yaxis=dict(autorange="reversed"), width=900, height=500)
                st.plotly_chart(dating_fig, width="stretch")

        with tab3:
            st.subheader("Topic Analysis")
            st.markdown(
                """
                Run topic modeling on the German translations found in the master sentence table.
                The result will include topic assignments and labels for each sentence.
                """
            )
            with st.form("run_topic_analysis_form"):
                run_topic_analysis = st.form_submit_button("Run topic analysis")

            if run_topic_analysis:
                with st.spinner("Running topic analysis..."):
                    try:
                        from src.topic_analysis_german_egyptian import run_topic_analysis_dataframe
                    except Exception as exc:
                        st.error(f"Could not import topic analysis module: {exc}")
                        run_topic_analysis = False
                        topic_df = None
                    else:
                        topic_df = run_topic_analysis_dataframe(
                            master_content_df,
                            text_column="german_translation",
                            min_topic_size=5,
                            nr_topics=None,
                            language="multilingual",
                            ngram_max=3,
                            min_df=2,
                            extraction_mode="translation_focus",
                            keep_intermediate_text=False,
                            output_path=None,
                        )
                if run_topic_analysis and topic_df is not None:
                    st.success("Topic analysis completed.")
                    st.subheader("Topic Analysis Results")
                    display_topic_columns = ["sentence_id", "german_translation", "topic", "topic_label"]
                    st.dataframe(topic_df[display_topic_columns], width="stretch")

                    st.subheader("Most common topics")

                    topic_counts = (
                        topic_df["topic_label"]
                        .value_counts()
                        .reset_index()
                    )
                    topic_counts.columns = ["topic_label", "sentence_count"]

                    fig_bar = px.bar(
                        topic_counts.head(15),
                        x="sentence_count",
                        y="topic_label",
                        orientation="h",
                        title="Top 15 topics by sentence count",
                    )
                    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_bar, width="stretch")

        st.stop()
    else:
        st.info("Enter a lemma ID and click 'Analyze Lemma' to query hiero_lemmas.")
        st.stop()

