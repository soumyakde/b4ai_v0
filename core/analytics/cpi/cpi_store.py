# core/analytics/cpi/cpi_store.py
"""
CPI Persistence Layer
=====================
SQLite storage for CPI_qual scoring runs and results.
Mirrors the dta_pipeline.py / ita_pipeline.py DB pattern exactly:
  - _find_db()        path resolution
  - _get_conn()       connection factory
  - _init_cpi_schema() idempotent table creation
  - create_cpi_run()  run registration
  - save_cpi_qual_result() upsert one scored reflection
  - save_cpi_summary()     upsert one participant CPI+ row
  - load_cpi_qual_results() load all scores for a run
  - load_cpi_summary()      load combined table for a run
  - list_cpi_runs()         list all runs newest first

CPI_quant (CTT and IRT) is pure computation from canonical_df —
no storage needed. Only CPI_qual LLM scores and the final combined
CPI+ are persisted.

Tables created in responses.db:
    cpi_runs
    cpi_qual_scores
    cpi_summary
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------
# DB path resolution — mirrors ita_pipeline._find_db exactly
# -----------------------------------------------------------------------

def _find_db() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "responses.db"
        if candidate.exists():
            return candidate
    return here.parents[min(3, len(here.parents) - 1)] / "responses.db"


_DB_PATH = _find_db()


def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or _DB_PATH)


# -----------------------------------------------------------------------
# Schema initialisation — idempotent
# -----------------------------------------------------------------------

def _init_cpi_schema(db_path: Optional[Path] = None) -> None:
    """
    Create CPI tables if they do not exist.
    Also applies safe column migrations on existing tables.
    Call once at app boot or before first write.
    """
    conn = _get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cpi_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        TEXT UNIQUE NOT NULL,
            created_by    TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            model         TEXT NOT NULL,
            temperature   REAL NOT NULL DEFAULT 0.0,
            module_id     TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            irt_model     TEXT,
            w1            REAL NOT NULL DEFAULT 0.5,
            w2            REAL NOT NULL DEFAULT 0.5,
            status        TEXT NOT NULL DEFAULT 'created',
            notes         TEXT
        );

        CREATE TABLE IF NOT EXISTS cpi_qual_scores (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id         TEXT NOT NULL,
            participant_id TEXT NOT NULL,
            module_id      TEXT NOT NULL,
            question_id    TEXT NOT NULL,
            dimension      TEXT NOT NULL,
            score          INTEGER NOT NULL,
            max_score      INTEGER NOT NULL DEFAULT 4,
            justification  TEXT,
            raw_prompt     TEXT,
            raw_response   TEXT,
            model          TEXT,
            tokens_used    INTEGER DEFAULT 0,
            created_at     TEXT,
            UNIQUE(run_id, participant_id, question_id, dimension)
        );

        CREATE TABLE IF NOT EXISTS cpi_summary (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id         TEXT NOT NULL,
            participant_id TEXT NOT NULL,
            module_id      TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            cpi_quant_ctt  REAL,
            cpi_quant_irt  REAL,
            cpi_quant      REAL,
            cpi_qual       REAL,
            cpi_plus       REAL,
            n_mcq_items    INTEGER,
            theta          REAL,
            theta_se       REAL,
            quant_method   TEXT,
            w1             REAL DEFAULT 0.5,
            w2             REAL DEFAULT 0.5,
            created_at     TEXT,
            UNIQUE(run_id, participant_id)
        );
    """)

    # Safe migrations: add any missing columns
    for table, col, typedef in [
        ("cpi_runs",        "notes",       "TEXT"),
        ("cpi_qual_scores", "tokens_used", "INTEGER DEFAULT 0"),
        ("cpi_qual_scores", "justification", "TEXT"),
        ("cpi_summary",     "quant_method",  "TEXT"),
    ]:
        existing = {
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------
# Public: create_cpi_run
# -----------------------------------------------------------------------

def create_cpi_run(
    created_by: str,
    model: str,
    module_id: str,
    instrument_key: str,
    irt_model: Optional[str] = None,
    temperature: float = 0.0,
    w1: float = 0.5,
    w2: float = 0.5,
    notes: str = "",
    db_path: Optional[Path] = None,
) -> str:
    """
    Register a new CPI run and return its run_id.

    Parameters match the cpi_runs table schema.
    Status is set to "created" on registration.
    """
    _init_cpi_schema(db_path)

    run_id = str(uuid.uuid4())
    conn   = _get_conn(db_path)
    conn.execute(
        """
        INSERT INTO cpi_runs
        (run_id, created_by, created_at, model, temperature,
         module_id, instrument_key, irt_model, w1, w2, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'created', ?)
        """,
        (
            run_id, created_by,
            datetime.utcnow().isoformat(),
            model, temperature, module_id, instrument_key,
            irt_model, w1, w2, notes or "",
        ),
    )
    conn.commit()
    conn.close()
    return run_id


def update_cpi_run_status(
    run_id: str,
    status: str,
    db_path: Optional[Path] = None,
) -> None:
    """Update run status: 'created' | 'scoring' | 'done' | 'error'."""
    conn = _get_conn(db_path)
    conn.execute(
        "UPDATE cpi_runs SET status = ? WHERE run_id = ?",
        (status, run_id),
    )
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------
# Public: save_cpi_qual_result
# -----------------------------------------------------------------------

def save_cpi_qual_result(
    run_id: str,
    scored: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> None:
    """
    Upsert one scored reflection (output of score_reflection_llm()).

    Expands the nested scores dict into individual rows — one row
    per (participant, question, dimension). This matches the normalized
    schema and makes per-dimension aggregation trivial in SQL.

    Parameters
    ----------
    run_id : str
    scored : dict — output of cpi_engine.score_reflection_llm()
    """
    _init_cpi_schema(db_path)

    if scored.get("error"):
        # Still record that we attempted this response (for audit)
        return

    conn = _get_conn(db_path)
    now  = datetime.utcnow().isoformat()

    for dim_id, score_val in scored["scores"].items():
        conn.execute(
            """
            INSERT OR REPLACE INTO cpi_qual_scores
            (run_id, participant_id, module_id, question_id, dimension,
             score, max_score, justification, raw_prompt, raw_response,
             model, tokens_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 4, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                scored["participant_id"],
                scored["module_id"],
                scored["question_id"],
                dim_id,
                int(score_val),
                scored.get("justification", "")[:500],
                scored.get("raw_prompt", "")[:4000],
                scored.get("raw_response", "")[:4000],
                scored.get("model", ""),
                int(scored.get("tokens_used", 0)),
                now,
            ),
        )
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------
# Public: save_cpi_summary
# -----------------------------------------------------------------------

def save_cpi_summary(
    run_id: str,
    participant_id: str,
    module_id: str,
    instrument_key: str,
    cpi_quant_ctt: Optional[float],
    cpi_quant_irt: Optional[float],
    cpi_quant: float,
    cpi_qual: Optional[float],
    cpi_plus: Optional[float],
    n_mcq_items: int,
    theta: Optional[float],
    theta_se: Optional[float],
    quant_method: str,
    w1: float,
    w2: float,
    db_path: Optional[Path] = None,
) -> None:
    """Upsert one row into cpi_summary."""
    _init_cpi_schema(db_path)

    conn = _get_conn(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO cpi_summary
        (run_id, participant_id, module_id, instrument_key,
         cpi_quant_ctt, cpi_quant_irt, cpi_quant, cpi_qual, cpi_plus,
         n_mcq_items, theta, theta_se, quant_method, w1, w2, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, participant_id, module_id, instrument_key,
            cpi_quant_ctt, cpi_quant_irt, cpi_quant, cpi_qual, cpi_plus,
            n_mcq_items, theta, theta_se, quant_method, w1, w2,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------
# Public: load_cpi_qual_results
# -----------------------------------------------------------------------

def load_cpi_qual_results(
    run_id: str,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load all CPI_qual dimension scores for a run.

    Returns
    -------
    pd.DataFrame — columns:
        participant_id | module_id | question_id | dimension
        | score | max_score | justification | model | tokens_used | created_at
    """
    _init_cpi_schema(db_path)
    conn = _get_conn(db_path)
    df   = pd.read_sql_query(
        """
        SELECT participant_id, module_id, question_id, dimension,
               score, max_score, justification, model, tokens_used, created_at
        FROM   cpi_qual_scores
        WHERE  run_id = ?
        ORDER  BY participant_id, question_id, dimension
        """,
        conn,
        params=(run_id,),
    )
    conn.close()
    return df


# -----------------------------------------------------------------------
# Public: load_cpi_summary
# -----------------------------------------------------------------------

def load_cpi_summary(
    run_id: str,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the combined CPI summary table for a run.

    Returns
    -------
    pd.DataFrame — columns:
        participant_id | module_id | instrument_key
        | cpi_quant_ctt | cpi_quant_irt | cpi_quant
        | cpi_qual | cpi_plus | n_mcq_items
        | theta | theta_se | quant_method | w1 | w2 | created_at
    """
    _init_cpi_schema(db_path)
    conn = _get_conn(db_path)
    df   = pd.read_sql_query(
        """
        SELECT participant_id, module_id, instrument_key,
               cpi_quant_ctt, cpi_quant_irt, cpi_quant,
               cpi_qual, cpi_plus, n_mcq_items,
               theta, theta_se, quant_method, w1, w2, created_at
        FROM   cpi_summary
        WHERE  run_id = ?
        ORDER  BY participant_id
        """,
        conn,
        params=(run_id,),
    )
    conn.close()
    return df


# -----------------------------------------------------------------------
# Public: list_cpi_runs
# -----------------------------------------------------------------------

def list_cpi_runs(
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    List all CPI runs, newest first.

    Returns
    -------
    list of dicts: run_id | created_by | created_at | model | module_id
                   | instrument_key | irt_model | w1 | w2 | status | notes
    """
    _init_cpi_schema(db_path)
    conn = _get_conn(db_path)
    rows = conn.execute(
        """
        SELECT run_id, created_by, created_at, model, module_id,
               instrument_key, irt_model, w1, w2, status, notes
        FROM   cpi_runs
        ORDER  BY created_at DESC
        """
    ).fetchall()
    conn.close()

    cols = [
        "run_id", "created_by", "created_at", "model", "module_id",
        "instrument_key", "irt_model", "w1", "w2", "status", "notes",
    ]
    return [dict(zip(cols, row)) for row in rows]
