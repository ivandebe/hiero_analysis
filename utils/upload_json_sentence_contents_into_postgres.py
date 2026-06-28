import json
from pathlib import Path
import psycopg
from psycopg import sql

SQL_QUERY_INITIAL = """
CREATE TABLE hiero_sentence_contents (
    url                TEXT,
    sentence_id        TEXT PRIMARY KEY,
    transliteration    TEXT[],         -- array of strings
    german_translation TEXT,           -- long free-text field
    sources            TEXT[],         -- array of strings
    dating             TEXT            -- free-text dating info
);
"""


def upload_sentence_contents_to_postgres(postgres_conn_string: str, json_file_path: str, table_name: str) -> None:
    """Upload sentence contents from a JSON file into a PostgreSQL table.

    Args:
        postgres_conn_string: A PostgreSQL connection string.
        json_file_path: Path to the JSON file containing sentence contents.
        table_name: Target table name in PostgreSQL.
    """
    json_path = Path(json_file_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items")
    if items is None:
        raise ValueError(f"JSON file does not contain an 'items' array: {json_file_path}")

    insert_query = sql.SQL(
        """
        INSERT INTO {table} (url, sentence_id, transliteration, german_translation, sources, dating)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (sentence_id)
        DO UPDATE SET
            url = EXCLUDED.url,
            transliteration = EXCLUDED.transliteration,
            german_translation = EXCLUDED.german_translation,
            sources = EXCLUDED.sources,
            dating = EXCLUDED.dating;
        """
    ).format(table=sql.Identifier(table_name))

    with psycopg.connect(postgres_conn_string) as conn:
        with conn.cursor() as cur:
            for item in items:
                url = item.get("url")
                sentence_id = item.get("sentence_id")
                transliteration = item.get("transliteration") or []
                german_translation = item.get("german_translation")
                sources = item.get("sources") or []
                dating = item.get("dating")

                if sentence_id is None:
                    raise ValueError("Missing sentence_id for an item in JSON file")

                cur.execute(
                    insert_query,
                    (url, sentence_id, transliteration, german_translation, sources, dating),
                )

        conn.commit()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    postgres_conn_string = os.getenv("POSTGRES_CONN_STRING")
    if not postgres_conn_string:
        raise ValueError("POSTGRES_CONN_STRING must be set in the environment to run as a script")

    upload_sentence_contents_to_postgres(
        postgres_conn_string=postgres_conn_string,
        json_file_path="lemma_125040_sentence_contents.json",
        table_name="hiero_sentence_contents",
    )

