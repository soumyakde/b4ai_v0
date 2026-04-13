# core/db_utils.py
"""
Database abstraction layer — Phase B.

Supports SQLite (default) and PostgreSQL via environment variable:
    DB_TYPE=sqlite   → uses SQLITE_PATH (default: responses.db)
    DB_TYPE=postgres → uses DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

Switch by setting DB_TYPE in .env — no code changes needed.
"""

import os
import sqlite3
from pathlib import Path
from datetime import datetime

# ── Backend selection ─────────────────────────────────────────────────────────
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()

# ── SQLite path resolution ────────────────────────────────────────────────────
# Priority 1: SQLITE_PATH env var (set in docker-compose.yml and Railway)
# Priority 2: DATA_DIR env var / default /app/data (Railway volume mount)
# Priority 3: project root relative path (local dev without Docker)
_ENV_SQLITE_PATH = os.getenv("SQLITE_PATH")
if _ENV_SQLITE_PATH:
    DB_PATH = Path(_ENV_SQLITE_PATH)
else:
    _DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parents[1])))
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = _DATA_DIR / "responses.db"

def get_connection(db_path: Path = None):
    """
    Return a database connection.

    SQLite:    returns sqlite3.Connection with row_factory set
    PostgreSQL: returns psycopg2 connection (Phase C)

    Args:
        db_path: Override path for SQLite only. Ignored for PostgreSQL.
    """
    if DB_TYPE == "postgres":
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                dbname=os.getenv("DB_NAME", "b4ai"),
                user=os.getenv("DB_USER", "b4ai"),
                password=os.getenv("DB_PASSWORD", ""),
            )
            return conn
        except ImportError:
            raise RuntimeError(
                "DB_TYPE=postgres but psycopg2 is not installed. "
                "Add psycopg2-binary to requirements.txt."
            )
    else:
        # SQLite (default) before journal_mode=WAL implementation
        #path = db_path or DB_PATH
        #conn = sqlite3.connect(path)
        #conn.row_factory = sqlite3.Row
        # - WAL Implementation
        path = db_path or DB_PATH
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn


def _placeholder() -> str:
    """Return the correct SQL placeholder for the active backend."""
    return "%s" if DB_TYPE == "postgres" else "?"


def init_db(db_path: Path = None):
    """
    Initialize the database and create tables if they do not exist.
    """
    conn = get_connection(db_path)
    # Enable WAL mode — critical for concurrent reads/writes
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, faster than FULL
    conn.execute("PRAGMA busy_timeout=5000")    # wait up to 5s if DB is locked
    conn.commit()
    cur = conn.cursor()

    # -----------------------------
    # RESPONSES TABLE (Unified)
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            rid TEXT,
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
            rid TEXT,
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
            rid TEXT,
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
            rid TEXT,
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

    # ── Auto-migrate existing DBs: add rid column if missing (idempotent) ──
    # Handles the case where the DB was created before Migration 1.
    # Safe to run on every startup — fails silently if column already exists.
    for _tbl in ["responses", "completions", "survey_scores", "assessment_scores"]:
        try:
            conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN rid TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists — expected on all non-fresh DBs

    conn.commit()
    conn.close()


# =====================================================
# COMPLETION HELPERS (Unlock-critical — DO NOT MODIFY)
# =====================================================

def mark_instrument_complete(
    user_id: str,
    module_id: str,
    instrument_key: str,
    db_path: Path = None,
    rid: str | None = None,
):
    if rid is None:
        try:
            from auth.user_manager import get_user_rid
            rid = get_user_rid(user_id)
        except Exception:
            rid = None
    conn = get_connection(db_path)
    cur  = conn.cursor()
    p    = _placeholder()

    cur.execute(f"""
        INSERT OR IGNORE INTO completions
        (user_id, rid, module_id, instrument_key, completed_at)
        VALUES ({p}, {p}, {p}, {p}, {p})
    """, (user_id, rid, module_id, instrument_key, datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()


def get_completed_instruments(
    user_id: str,
    module_id: str,
    db_path: Path = None,
):
    conn = get_connection(db_path)
    cur  = conn.cursor()
    p    = _placeholder()

    cur.execute(f"""
        SELECT instrument_key
        FROM completions
        WHERE user_id = {p} AND module_id = {p}
    """, (user_id, module_id))

    results = [row["instrument_key"] if hasattr(row, '__getitem__')
               else row[0] for row in cur.fetchall()]
    conn.close()
    return results


def is_instrument_complete(
    user_id: str,
    module_id: str,
    instrument_key: str,
    db_path: Path = None,
) -> bool:
    conn = get_connection(db_path)
    cur  = conn.cursor()
    p    = _placeholder()

    cur.execute(f"""
        SELECT 1 FROM completions
        WHERE user_id = {p} AND module_id = {p} AND instrument_key = {p}
    """, (user_id, module_id, instrument_key))

    exists = cur.fetchone() is not None
    conn.close()
    return exists
