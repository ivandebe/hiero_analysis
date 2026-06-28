import os
from dotenv import load_dotenv
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import pandas as pd
import streamlit as st

load_dotenv()

def get_conn_string():
    return (
        os.getenv("POSTGRES_CONN_STRING")
        or st.secrets["database"]["POSTGRES_CONN_STRING"]
    )

def get_db_connection():
    conn_string = get_conn_string()
    if not conn_string:
        raise ValueError(
            "POSTGRES_CONN_STRING not found. Set it in .env locally or Streamlit secrets in production."
        )
    return psycopg.connect(conn_string, row_factory=dict_row)


def fetch_entire_table(table_name: str) -> pd.DataFrame:
    query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    return df
