# core/db_utils.py

import sqlite3
from pathlib import Path
from datetime import datetime

# Default database path
DB_PATH = Path("responses.db")


def get_connection(db_path: Path = None):
    """
    Get a connection to the SQLite database.

    Args:
        db_path (Path, optional): Path to the SQLite DB file.
                                  Defaults to global DB_PATH.
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = None):
    """
    Initialize the database and create tables if they do not exist.

    Args:
        db_path (Path, optional): Path to the SQLite DB file.
                                  Defaults to global DB_PATH.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    # -----------------------------
    # RESPONSES TABLE (Unified)
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            instrument_name TEXT NOT NULL,
            question_id TEXT NOT NULL,
            response_value TEXT,
            submitted_at TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_responses_user_instrument
        ON responses(user_id, instrument_name);
    """)

    # -----------------------------
    # SURVEY SCORES
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS survey_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            survey_key TEXT NOT NULL,
            score REAL,
            calculated_at TEXT NOT NULL,
            UNIQUE(user_id, survey_key)
        );
    """)

    # -----------------------------
    # ASSESSMENT SCORES
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS assessment_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            assessment_code TEXT NOT NULL,
            score REAL,
            calculated_at TEXT NOT NULL,
            UNIQUE(user_id, assessment_code)
        );
    """)

    # -----------------------------
    # COMPLETIONS (Unlock source of truth)
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            module_id TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            UNIQUE(user_id, module_id, instrument_key)
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_completions_user_module
        ON completions(user_id, module_id);
    """)

    conn.commit()
    conn.close()


# =====================================================
# COMPLETION HELPERS (Unlock-critical — DO NOT MODIFY)
# =====================================================

def mark_instrument_complete(user_id: str, module_id: str, instrument_key: str, db_path: Path = None):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR IGNORE INTO completions
        (user_id, module_id, instrument_key, completed_at)
        VALUES (?, ?, ?, ?)
    """, (user_id, module_id, instrument_key, datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()


def get_completed_instruments(user_id: str, module_id: str, db_path: Path = None):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT instrument_key
        FROM completions
        WHERE user_id = ? AND module_id = ?
    """, (user_id, module_id))

    results = [row["instrument_key"] for row in cur.fetchall()]
    conn.close()
    return results


def is_instrument_complete(user_id: str, module_id: str, instrument_key: str, db_path: Path = None):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT 1 FROM completions
        WHERE user_id = ? AND module_id = ? AND instrument_key = ?
    """, (user_id, module_id, instrument_key))

    exists = cur.fetchone() is not None
    conn.close()
    return exists