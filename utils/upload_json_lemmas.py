import json
from pathlib import Path

SQL_QUERY_INITIAL = """
CREATE TABLE hiero_lemmas (
    lemma_id           TEXT PRIMARY KEY,
    transliteration    TEXT,      
    sentences_id       TEXT[]         -- array of strings
);
"""


def upload_lemmas_to_postgres(postgres_conn_string: str, json_file_path: str, table_name: str) -> None:
    """Upload lemma data from a JSON file into a PostgreSQL table.

    Args:
        postgres_conn_string: A PostgreSQL connection string.
        json_file_path: Path to the JSON file containing lemma data.
        table_name: Target table name in PostgreSQL.
    """
    json_path = Path(json_file_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support multiple JSON shapes:
    # - A dict with an "items" array
    # - A top-level list of items
    # - A single-entry dict that contains `lemma_id` and `sentence_ids` (the file used here)
    # - A mapping of lemma_id -> list(sentence_ids)
    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            items = data["items"]
        elif "lemma_id" in data or "lemmaId" in data or "sentence_ids" in data or "sentences_id" in data:
            # Single lemma entry
            items = [data]
        else:
            # Try to interpret as mapping lemma_id -> list(sentence_ids)
            possible_items = []
            for k, v in data.items():
                if isinstance(v, list):
                    possible_items.append({"lemma_id": k, "sentences_id": v})

            if possible_items:
                items = possible_items

    if not items:
        raise ValueError(f"JSON file does not contain recognizable lemma entries: {json_file_path}")

    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise ImportError(
            "psycopg is required to upload sentence contents to Postgres. "
            "Install psycopg and retry."
        ) from exc

    insert_query = sql.SQL(
        """
        INSERT INTO {table} (lemma_id, transliteration, sentences_id)
        VALUES (%s, %s, %s)
        ON CONFLICT (lemma_id)
        DO UPDATE SET
            transliteration = EXCLUDED.transliteration,
            sentences_id = EXCLUDED.sentences_id;
        """
    ).format(table=sql.Identifier(table_name))

    with psycopg.connect(postgres_conn_string) as conn:
        with conn.cursor() as cur:
            for item in items:
                # Normalize keys that might appear in different JSON formats
                lemma_id = (
                    item.get("lemma_id")
                    or item.get("lemmaId")
                    or item.get("lemma")
                )

                transliteration = item.get("transliteration")

                # sentence keys may be named differently across files
                sentences_id = (
                    item.get("sentences_id")
                    or item.get("sentence_ids")
                    or item.get("sentences")
                    or item.get("sentenceIds")
                    or item.get("sentence_id")
                ) or []

                if lemma_id is None:
                    raise ValueError("Missing lemma_id for an item in JSON file")

                # Ensure sentences_id is a list (Postgres TEXT[])
                if not isinstance(sentences_id, list):
                    sentences_id = [sentences_id]

                cur.execute(
                    insert_query,
                    (str(lemma_id), transliteration, sentences_id),
                )

        conn.commit()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    postgres_conn_string = os.getenv("POSTGRES_CONN_STRING")
    if not postgres_conn_string:
        raise ValueError("POSTGRES_CONN_STRING must be set in the environment to run as a script")

    upload_lemmas_to_postgres(
        postgres_conn_string=postgres_conn_string,
        json_file_path=r"/Users/ivandebe/Projects/hiero_analysis/data/lemma_104730_sentence_ids.json",
        table_name="hiero_lemmas",
    )

