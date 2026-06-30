from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable
import networkx as nx
import ast

import pandas as pd
from pandas.api.types import is_list_like
import plotly.graph_objects as go

REQUIRED_COLUMNS = {"sentence_id", "transliteration"}


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input dataframe is missing required columns: {sorted(missing)}"
        )


def _parse_lemma_list(value) -> list[str]:
    """
    Accepts:
    - a Python list: ['stp', 'r', 'nswt']
    - a tuple/set
    - a comma-separated string: 'stp,r,nswt'
    - a string representation of a Python list: "['stp', 'r', 'nswt']"
    """
    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    raw = list(parsed)
                else:
                    raw = [text]
            except Exception:
                raw = [part.strip() for part in text.split(",")]
        else:
            raw = [part.strip() for part in text.split(",")]
    elif is_list_like(value):
        raw = list(value)
    elif pd.isna(value):
        return []
    else:
        raw = [str(value)]

    lemmas = []
    for item in raw:
        lemma = str(item).strip()
        if lemma:
            lemmas.append(lemma)

    return lemmas


def _filter_to_core_closeness(
    lemma_list: list[str],
    core_lemma: str | None,
    closeness: int | None,
) -> list[str]:
    if not core_lemma or closeness is None or closeness < 1:
        return lemma_list

    nearby = set()
    for index, lemma in enumerate(lemma_list):
        if lemma != core_lemma:
            continue
        for distance in range(1, closeness + 1):
            if index - distance >= 0:
                nearby.add(lemma_list[index - distance])
            if index + distance < len(lemma_list):
                nearby.add(lemma_list[index + distance])

    if not nearby:
        return [core_lemma] if core_lemma in lemma_list else []

    nearby.add(core_lemma)
    return [lemma for lemma in lemma_list if lemma in nearby]


def _prepare_dataframe(
    df: pd.DataFrame,
    deduplicate_lemmas_per_sentence: bool = True,
    core_lemma: str | None = None,
    closeness: int | None = None,
) -> pd.DataFrame:
    _validate_input(df)

    data = df.copy()
    data = data.dropna(subset=["sentence_id", "transliteration"]).copy()

    data["sentence_id"] = data["sentence_id"].astype(str).str.strip()
    data = data[data["sentence_id"] != ""].copy()

    data["lemma_list"] = data["transliteration"].apply(_parse_lemma_list)

    if core_lemma or closeness is not None:
        data["lemma_list"] = data["lemma_list"].apply(
            lambda lemmas: _filter_to_core_closeness(lemmas, core_lemma, closeness)
        )

    if deduplicate_lemmas_per_sentence:
        data["lemma_list"] = data["lemma_list"].apply(lambda x: sorted(set(x)))

    data["lemma_count"] = data["lemma_list"].apply(len)
    data = data[data["lemma_count"] >= 2].copy()

    if data.empty:
        raise ValueError(
            "No valid rows available after preprocessing. "
            "Each sentence must contain at least two lemmas."
        )

    return data.reset_index(drop=True)


def _compute_graph_components(
    df: pd.DataFrame,
    min_edge_weight: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lemma_counter: Counter[str] = Counter()
    edge_counter: Counter[tuple[str, str]] = Counter()

    for lemmas in df["lemma_list"]:
        lemma_counter.update(lemmas)
        for a, b in combinations(sorted(lemmas), 2):
            edge_counter[(a, b)] += 1

    edge_rows = [
        {
            "source": a,
            "target": b,
            "weight": w,
        }
        for (a, b), w in edge_counter.items()
        if w >= min_edge_weight
    ]
    edges_df = pd.DataFrame(edge_rows)

    if edges_df.empty:
        connected_lemmas = set()
    else:
        connected_lemmas = set(edges_df["source"]).union(set(edges_df["target"]))

    node_rows = [
        {
            "lemma": lemma,
            "sentence_count": int(lemma_counter.get(lemma, 0)),
            "degree": int(
                ((edges_df["source"] == lemma) | (edges_df["target"] == lemma)).sum()
            ) if not edges_df.empty else 0,
        }
        for lemma in sorted(lemma_counter.keys())
        if (not connected_lemmas) or (lemma in connected_lemmas)
    ]
    nodes_df = pd.DataFrame(node_rows)

    return nodes_df, edges_df


def create_lemma_cooccurrence_figure(
    df: pd.DataFrame,
    *,
    deduplicate_lemmas_per_sentence: bool = True,
    min_edge_weight: int = 2,
    max_lemmas: int | None = 50,
    core_lemma: str | None = None,
    closeness: int | None = None,
    layout_seed: int = 42,
    layout_iterations: int = 100,
    title: str = "Lemma co-occurrence graph",
) -> go.Figure:
    """
    Create a Plotly network graph of lemma co-occurrence across sentences.

    Parameters
    ----------
    df:
        Dataframe containing at least:
        - sentence_id
        - transliteration
    deduplicate_lemmas_per_sentence:
        If True, each lemma is counted at most once per sentence.
        Usually this is the preferred option.
    min_edge_weight:
        Minimum number of sentences in which two lemmas must co-occur
        to draw an edge.
    max_lemmas:
        Keep only the top N lemmas by sentence frequency before building the graph.
        Useful to avoid overcrowded graphs.
    layout_seed:
        Seed used by the spring layout for reproducible positioning.
    layout_iterations:
        Number of spring-layout iterations.
    title:
        Figure title.
    """
    data = _prepare_dataframe(
        df,
        deduplicate_lemmas_per_sentence=deduplicate_lemmas_per_sentence,
        core_lemma=core_lemma,
        closeness=closeness,
    )

    exploded = data[["sentence_id", "lemma_list"]].explode("lemma_list")
    lemma_freq = exploded["lemma_list"].value_counts()

    if max_lemmas is not None and max_lemmas > 0:
        top_lemmas = set(lemma_freq.head(max_lemmas).index)
        data["lemma_list"] = data["lemma_list"].apply(
            lambda lemmas: [lemma for lemma in lemmas if lemma in top_lemmas]
        )
        data["lemma_count"] = data["lemma_list"].apply(len)
        data = data[data["lemma_count"] >= 2].copy()

    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            annotations=[
                dict(
                    text="Not enough lemmas remain after filtering to build a co-occurrence graph.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    nodes_df, edges_df = _compute_graph_components(
        data,
        min_edge_weight=min_edge_weight,
    )

    if nodes_df.empty or edges_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            annotations=[
                dict(
                    text="No edges survived the current threshold. Try a lower min_edge_weight or increase max_lemmas.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    graph = nx.Graph()

    for _, row in nodes_df.iterrows():
        graph.add_node(
            row["lemma"],
            sentence_count=int(row["sentence_count"]),
            degree=int(row["degree"]),
        )

    for _, row in edges_df.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            weight=int(row["weight"]),
        )

    positions = nx.spring_layout(
        graph,
        seed=layout_seed,
        weight="weight",
        iterations=layout_iterations,
    )

    edge_x = []
    edge_y = []
    edge_text_x = []
    edge_text_y = []
    edge_text = []

    for source, target, attrs in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text_x.append((x0 + x1) / 2)
        edge_text_y.append((y0 + y1) / 2)
        edge_text.append(
            f"{source} ↔ {target}<br>Co-occurrence count: {attrs['weight']}"
        )

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_labels = []

    max_sentences = max(nx.get_node_attributes(graph, "sentence_count").values())

    for node, attrs in graph.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)
        node_color.append(attrs["degree"])
        size = 18 + 42 * (attrs["sentence_count"] / max_sentences)
        node_size.append(size)
        node_text.append(
            f"Lemma: {node}<br>"
            f"Sentences: {attrs['sentence_count']}<br>"
            f"Connections: {attrs['degree']}"
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="rgba(120,120,120,0.45)"),
        hoverinfo="skip",
        showlegend=False,
    )

    edge_hover_trace = go.Scatter(
        x=edge_text_x,
        y=edge_text_y,
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)"),
        text=edge_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="YlGnBu",
            showscale=True,
            colorbar=dict(title="Degree"),
            line=dict(width=1, color="white"),
            opacity=0.92,
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace])
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=750,
    )

    return fig


def create_lemma_cooccurrence_tables(
    df: pd.DataFrame,
    *,
    deduplicate_lemmas_per_sentence: bool = True,
    min_edge_weight: int = 1,
    max_lemmas: int | None = 50,
    core_lemma: str | None = None,
    closeness: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return node and edge tables used to build the graph.
    Useful for debugging or showing supporting tables in Streamlit.
    """
    data = _prepare_dataframe(
        df,
        deduplicate_lemmas_per_sentence=deduplicate_lemmas_per_sentence,
        core_lemma=core_lemma,
        closeness=closeness,
    )

    exploded = data[["sentence_id", "lemma_list"]].explode("lemma_list")
    lemma_freq = exploded["lemma_list"].value_counts()

    if max_lemmas is not None and max_lemmas > 0:
        top_lemmas = set(lemma_freq.head(max_lemmas).index)
        data["lemma_list"] = data["lemma_list"].apply(
            lambda lemmas: [lemma for lemma in lemmas if lemma in top_lemmas]
        )
        data["lemma_count"] = data["lemma_list"].apply(len)
        data = data[data["lemma_count"] >= 2].copy()

    return _compute_graph_components(
        data,
        min_edge_weight=min_edge_weight,
    )
