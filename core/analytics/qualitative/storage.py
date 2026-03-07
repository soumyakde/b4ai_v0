"""
storage.py
-------------------------------------------------------
Qualitative Rating Storage Layer

Responsibilities:
- SQLite schema creation
- Append-only rating persistence
- Arrow-safe dataframe retrieval
- CPI-ready exports
- Versioned prompt tracking

NO LLM LOGIC
NO UI LOGIC
-------------------------------------------------------
"""

from typing import List, Dict, Any, Optional
import sqlite3
import pandas as pd
from pathlib import Path
import json
import time


# =====================================================
# DB CONNECTION
# =====================================================

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    return conn


# =====================================================
# SCHEMA
# =====================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS qualitative_ratings (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    timestamp_utc REAL NOT NULL,

    student_id TEXT NOT NULL,
    module_id TEXT NOT NULL,
    question_id TEXT NOT NULL,

    construct TEXT NOT NULL,
    rating REAL NOT NULL,
    evidence TEXT,
    confidence REAL,

    model TEXT NOT NULL,

    prompt_hash TEXT NOT NULL,
    response_hash TEXT NOT NULL,

    latency_seconds REAL,
    attempt INTEGER,

    raw_llm_output TEXT,

    UNIQUE(
        student_id,
        question_id,
        construct,
        prompt_hash,
        response_hash
    )
);

CREATE INDEX IF NOT EXISTS idx_qr_student
ON qualitative_ratings(student_id);

CREATE INDEX IF NOT EXISTS idx_qr_module
ON qualitative_ratings(module_id);

CREATE INDEX IF NOT EXISTS idx_qr_prompt
ON qualitative_ratings(prompt_hash);
"""


def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.commit()


# =====================================================
# INSERTION
# =====================================================

def insert_ratings(
    conn: sqlite3.Connection,
    flat_rows: List[Dict[str, Any]],
):
    """
    Insert flattened ratings (from engine.flatten_ratings).

    Append-only semantics.
    Duplicate runs safely ignored.
    """

    ensure_schema(conn)

    sql = """
    INSERT OR IGNORE INTO qualitative_ratings (
        timestamp_utc,
        student_id,
        module_id,
        question_id,
        construct,
        rating,
        evidence,
        confidence,
        model,
        prompt_hash,
        response_hash,
        latency_seconds,
        attempt,
        raw_llm_output
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    rows = []

    now = time.time()

    for r in flat_rows:

        rows.append((
            now,
            r["student_id"],
            r["module_id"],
            r["question_id"],
            r["construct"],
            float(r["rating"]),
            r.get("evidence"),
            r.get("confidence"),
            r["model"],
            r["prompt_hash"],
            r["response_hash"],
            r.get("latency_seconds"),
            r.get("attempt"),
            r.get("raw_llm_output"),
        ))

    conn.executemany(sql, rows)
    conn.commit()


# =====================================================
# RETRIEVAL
# =====================================================

def load_ratings_df(
    conn: sqlite3.Connection,
    module_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns Arrow-safe dataframe for Streamlit.
    """

    query = "SELECT * FROM qualitative_ratings"

    params = []

    if module_id:
        query += " WHERE module_id=?"
        params.append(module_id)

    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    return sanitize_dataframe_for_streamlit(df)


# =====================================================
# CPI EXPORT (CRITICAL)
# =====================================================

def export_cpi_long_format(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Returns CPI-ready long format:

    student | module | construct | rating
    """

    query = """
    SELECT
        student_id,
        module_id,
        construct,
        rating
    FROM qualitative_ratings
    """

    df = pd.read_sql_query(query, conn)

    return sanitize_dataframe_for_streamlit(df)


# =====================================================
# PROMPT VERSION AUDIT
# =====================================================

def get_prompt_versions(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT
        prompt_hash,
        model,
        COUNT(*) as n_ratings,
        MIN(timestamp_utc) as first_seen,
        MAX(timestamp_utc) as last_seen
    FROM qualitative_ratings
    GROUP BY prompt_hash, model
    ORDER BY last_seen DESC
    """

    df = pd.read_sql_query(query, conn)

    return sanitize_dataframe_for_streamlit(df)


# =====================================================
# STREAMLIT SAFETY
# =====================================================

def sanitize_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prevent Arrow serialization crashes.

    Converts object columns → safe strings.
    """

    df = df.copy()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = df[col].apply(
                lambda x:
                json.dumps(x)
                if isinstance(x, (dict, list))
                else ("" if x is None else str(x))
            )

    return df


# =====================================================
# CONVENIENCE WRAPPER
# =====================================================

class QualitativeStore:
    """
    High-level storage interface.
    """

    def __init__(self, db_path: str):

        Path(db_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.conn = get_connection(db_path)
        ensure_schema(self.conn)

    # -------------------------

    def save(self, flat_rows: List[Dict[str, Any]]):
        insert_ratings(self.conn, flat_rows)

    # -------------------------

    def dataframe(self, module_id=None):
        return load_ratings_df(self.conn, module_id)

    # -------------------------

    def cpi_dataset(self):
        return export_cpi_long_format(self.conn)

    # -------------------------

    def prompt_versions(self):
        return get_prompt_versions(self.conn)