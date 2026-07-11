"""
auth/login_security.py
Login Attempt Tracking and Account Lockout

Responsibilities:
- Create and maintain the login_attempts table in users.db
- Record every login attempt (success or failure)
- Enforce username-based lockout after 3 failures in 5 minutes

This module has:
- No Streamlit
- No session_state  (lockout state lives in DB, not per-tab memory)
- No dependency on user_manager.py  (reads same users.db independently)

Design decisions:
- IP is always stored as 'unknown'. Streamlit cannot guarantee real
  client IPs behind a reverse proxy, and lockout is username-based only.
  The ip column exists for audit trail completeness.
- Lockout is purely time-windowed: the 5-minute window rolls forward
  from the FIRST failure in the current window, not from the last.
  This means a lockout expires exactly LOCKOUT_MINUTES after it began,
  regardless of subsequent attempts during the window.
- All timestamps are UTC ISO-8601 strings (consistent with audit_logger.py).

DB table created:
    login_attempts(
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT NOT NULL,
        ip        TEXT NOT NULL DEFAULT 'unknown',
        timestamp TEXT NOT NULL,
        success   INTEGER NOT NULL DEFAULT 0   -- 1=success, 0=failure
    )

Configuration (via .env, read at import time):
    USERS_DB_PATH     — override path to users.db (same var as user_service.py)
    LOGIN_MAX_ATTEMPTS — failures before lockout (default: 3)
    LOGIN_LOCKOUT_MINUTES — lockout duration in minutes (default: 5)
    LOGIN_WINDOW_MINUTES  — rolling failure window (default: 5)
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

_ENV_USERS_PATH = os.getenv("USERS_DB_PATH")
_BASE_DIR       = Path(__file__).resolve().parents[1]
_DB_PATH        = Path(_ENV_USERS_PATH) if _ENV_USERS_PATH else _BASE_DIR / "users.db"

MAX_ATTEMPTS     = int(os.getenv("LOGIN_MAX_ATTEMPTS",     "3"))
LOCKOUT_MINUTES  = int(os.getenv("LOGIN_LOCKOUT_MINUTES",  "5"))
WINDOW_MINUTES   = int(os.getenv("LOGIN_WINDOW_MINUTES",   "5"))


# ------------------------------------------------------------------
# Pilot-mode lockout override (in-memory, process-local)
# ------------------------------------------------------------------
# Lets an admin disable the lockout mechanism entirely during a
# time-boxed in-person pilot session, where a participant losing
# LOCKOUT_MINUTES to a mistyped password is costly. Intentionally
# process-local, not persisted to the DB: it resets to enabled (the
# safe default) on every app restart, including any scheduled restart,
# rather than silently staying off indefinitely.

_LOCKOUT_ENABLED = True


def set_lockout_enabled(enabled: bool) -> None:
    """Globally enable/disable lockout checking for this running process."""
    global _LOCKOUT_ENABLED
    _LOCKOUT_ENABLED = enabled


def is_lockout_enabled() -> bool:
    """Whether the lockout mechanism is currently active platform-wide."""
    return _LOCKOUT_ENABLED


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


# ------------------------------------------------------------------
# Table initialisation (idempotent — safe to call on every boot)
# ------------------------------------------------------------------

def init_login_attempts_table() -> None:
    """
    Create the login_attempts table and index if they do not exist.
    Called once at app.py boot — idempotent, never destructive.
    """
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    NOT NULL,
            ip        TEXT    NOT NULL DEFAULT 'unknown',
            timestamp TEXT    NOT NULL,
            success   INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_login_attempts_user_ts
        ON login_attempts(username, timestamp)
    """)
    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def record_attempt(username: str, success: bool) -> None:
    """
    Persist a login attempt to the database.

    Called by app.py show_login() after every authentication attempt,
    whether successful or not.

    Args:
        username: The username that was attempted (stripped, as typed).
        success:  True if authenticate_user() returned True, else False.
    """
    conn = _conn()
    conn.execute(
        """
        INSERT INTO login_attempts (username, ip, timestamp, success)
        VALUES (?, 'unknown', ?, ?)
        """,
        (username, datetime.utcnow().isoformat(), 1 if success else 0),
    )
    conn.commit()
    conn.close()


def is_locked_out(username: str) -> bool:
    """
    Return True if this username is currently under a timed lockout.

    Lockout condition: MAX_ATTEMPTS or more failures recorded within
    the past WINDOW_MINUTES minutes.

    Always returns False if the login_attempts table doesn't exist yet
    (first boot before init_login_attempts_table has run), or if the
    lockout mechanism has been globally disabled via set_lockout_enabled().
    """
    if not _LOCKOUT_ENABLED:
        return False
    try:
        window_start = (
            datetime.utcnow() - timedelta(minutes=WINDOW_MINUTES)
        ).isoformat()
        conn = _conn()
        cur  = conn.execute(
            """
            SELECT COUNT(*) FROM login_attempts
            WHERE username = ? AND success = 0 AND timestamp >= ?
            """,
            (username, window_start),
        )
        count = cur.fetchone()[0]
        conn.close()
        return count >= MAX_ATTEMPTS
    except sqlite3.OperationalError:
        # Table not yet created — fail open (no lockout)
        return False


def get_lockout_remaining_seconds(username: str) -> int:
    """
    Return the number of seconds remaining in the active lockout,
    or 0 if the user is not currently locked out.

    The lockout timer starts from the FIRST failure in the current
    window and expires LOCKOUT_MINUTES later.
    """
    try:
        window_start = (
            datetime.utcnow() - timedelta(minutes=WINDOW_MINUTES)
        ).isoformat()
        conn = _conn()
        cur  = conn.execute(
            """
            SELECT timestamp FROM login_attempts
            WHERE username = ? AND success = 0 AND timestamp >= ?
            ORDER BY timestamp ASC
            LIMIT 1
            """,
            (username, window_start),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return 0
        first_failure = datetime.fromisoformat(row[0])
        expires_at    = first_failure + timedelta(minutes=LOCKOUT_MINUTES)
        remaining     = (expires_at - datetime.utcnow()).total_seconds()
        return max(0, int(remaining))
    except sqlite3.OperationalError:
        return 0


def clear_lockout(username: str) -> int:
    """
    Clear a user's current lockout by deleting their recent failed
    attempts within the rolling window (the ones that count toward
    is_locked_out()'s MAX_ATTEMPTS threshold).

    Does not touch successful-login rows or failures outside the
    window — those are already irrelevant to the lockout calculation,
    so this only removes exactly what's blocking the user right now.

    Returns the number of attempt rows deleted (0 if the user wasn't
    actually locked out, or the table doesn't exist yet).
    """
    try:
        window_start = (
            datetime.utcnow() - timedelta(minutes=WINDOW_MINUTES)
        ).isoformat()
        conn = _conn()
        cur = conn.execute(
            """
            DELETE FROM login_attempts
            WHERE username = ? AND success = 0 AND timestamp >= ?
            """,
            (username, window_start),
        )
        conn.commit()
        deleted = cur.rowcount
        conn.close()
        return deleted
    except sqlite3.OperationalError:
        return 0


def get_recent_attempts(username: str, limit: int = 10) -> list[dict]:
    """
    Return recent login attempts for a username.
    Used by admin diagnostics — not used in the login flow itself.
    """
    try:
        conn = _conn()
        cur  = conn.execute(
            """
            SELECT timestamp, success
            FROM login_attempts
            WHERE username = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (username, limit),
        )
        rows = [
            {"timestamp": r[0], "success": bool(r[1])}
            for r in cur.fetchall()
        ]
        conn.close()
        return rows
    except sqlite3.OperationalError:
        return []
