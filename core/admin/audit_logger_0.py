"""
audit_logger.py

Administrative action logging for the BasicsB4AI platform.
Provides structured logging of admin actions for research
provenance and system accountability.

Changes:
  - Added RESTORE_DATABASE action (for restore + log feature)
  - Added LOGIN_LOCKOUT action (for security audit trail)
  - Added get_all_logs() returning a pandas DataFrame (for CSV export)
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from enum import Enum


# ---------------------------------------------------------
# DATABASE LOCATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "users.db"


# ---------------------------------------------------------
# ADMIN ACTION ENUM
# ---------------------------------------------------------

class AdminAction(Enum):

    CREATE_USER       = "CREATE_USER"
    DELETE_USER       = "DELETE_USER"
    DEACTIVATE_USER   = "DEACTIVATE_USER"
    CHANGE_ROLE       = "CHANGE_ROLE"
    UPDATE_COHORT     = "UPDATE_COHORT"
    IMPERSONATE_USER  = "IMPERSONATE_USER"

    RESET_STUDENT_DATA = "RESET_STUDENT_DATA"
    RESET_INSTRUMENT   = "RESET_INSTRUMENT"
    RESET_STUDY        = "RESET_STUDY"

    IMPORT_USERS         = "IMPORT_USERS"
    IMPORT_HUMAN_RATINGS = "IMPORT_HUMAN_RATINGS"

    BACKUP_DATABASE  = "BACKUP_DATABASE"
    RESTORE_DATABASE = "RESTORE_DATABASE"     # ← NEW
    CLONE_DATABASE   = "CLONE_DATABASE"
    CLEAR_CACHE      = "CLEAR_CACHE"

    LOGIN_LOCKOUT    = "LOGIN_LOCKOUT"        # ← NEW: recorded when lockout triggers

    RUN_DIAGNOSTICS  = "RUN_DIAGNOSTICS"


# ---------------------------------------------------------
# TABLE INITIALIZATION
# ---------------------------------------------------------

def initialize_admin_log_table():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            admin_user TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT
        )
        """
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# LOG ACTION
# ---------------------------------------------------------

def log_admin_action(admin_user: str, action: AdminAction, details: str = ""):

    initialize_admin_log_table()

    timestamp = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO admin_logs (timestamp, admin_user, action, details)
        VALUES (?, ?, ?, ?)
        """,
        (timestamp, admin_user, action.value, details)
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# FETCH LOGS — for dashboard display (last N rows)
# ---------------------------------------------------------

def get_recent_logs(limit: int = 50):
    """Return recent log rows as list of tuples (for legacy callers)."""
    initialize_admin_log_table()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT timestamp, admin_user, action, details
        FROM admin_logs
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


# ---------------------------------------------------------
# FETCH LOGS — full export as pandas DataFrame (for CSV)
# ---------------------------------------------------------

def get_all_logs_df(limit: int = 5000):
    """
    Return all audit log entries as a pandas DataFrame.
    Used by admin dashboard CSV export button.
    Columns: timestamp, admin_user, action, details
    """
    import pandas as pd
    initialize_admin_log_table()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT timestamp, admin_user, action, details
        FROM admin_logs
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()

    return pd.DataFrame(
        rows,
        columns=["timestamp", "admin_user", "action", "details"],
    )
