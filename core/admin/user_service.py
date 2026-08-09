"""
core/admin/user_service.py

Administrative user management operations.
Aligned with actual users.db schema.
"""

import os
import sqlite3
from pathlib import Path

from core.admin.audit_logger import log_admin_action, AdminAction
from auth.user_manager import hash_password, generate_password

BASE_DIR = Path(__file__).resolve().parents[2]

# ── users.db path — respects USERS_DB_PATH env var for Docker/cloud ──────────
_ENV_USERS_PATH = os.getenv("USERS_DB_PATH")
DB_PATH = Path(_ENV_USERS_PATH) if _ENV_USERS_PATH else BASE_DIR / "users.db"


def get_connection():
    """Always SQLite for users.db (auth store — not migrated to Postgres yet)."""
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------
# GET ALL USERS
# ---------------------------------------------------------

def get_all_users():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, role, cohort_id, status
        FROM users
        ORDER BY role, username
        """
    )
    rows = cursor.fetchall()
    conn.close()

    import pandas as pd
    return pd.DataFrame(rows, columns=["id", "username", "role", "cohort_id", "status"])


# ---------------------------------------------------------
# COHORT FETCH  (fixed: was using hardcoded relative path)
# ---------------------------------------------------------

def get_all_cohorts():
    """
    Returns list of cohort IDs.

    Works both:
    - locally (table already exists)
    - Railway fresh deployments (table auto-created)
    """

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------
    # NEW: Ensure cohorts table exists (safe + idempotent)
    # --------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cohorts (
            cohort_id TEXT PRIMARY KEY
        )
    """)
    conn.commit()
    # --------------------------------------------------

    # Existing logic (unchanged)
    cursor.execute(
        "SELECT cohort_id FROM cohorts ORDER BY cohort_id"
    )

    rows = cursor.fetchall()
    conn.close()

    return [row[0] for row in rows]


# ---------------------------------------------------------
# ADD COHORT
# ---------------------------------------------------------

def add_cohort(cohort_id: str) -> None:
    """
    Register a new cohort_id in the cohorts table.

    Extracted 2026-08-08 from inline SQL that used to live only in
    admin_dashboard.py's "Cohort Management" section, so a second call
    site (the transcript-upload cohort tagging UI) can create a cohort
    without duplicating the insert logic. Idempotent — INSERT OR IGNORE,
    safe to call with a cohort_id that already exists.
    """
    cohort_id = (cohort_id or "").strip()
    if not cohort_id:
        return

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cohorts (
            cohort_id TEXT PRIMARY KEY
        )
    """)
    cursor.execute("INSERT OR IGNORE INTO cohorts (cohort_id) VALUES (?)", (cohort_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# COHORT USAGE / DELETE
# ---------------------------------------------------------

def count_cohort_usage(cohort_id: str) -> dict:
    """
    Count how many users and transcripts currently reference cohort_id.

    Checks both DBs a cohort_id can appear in: users.db (registered
    students assigned to this cohort) and responses.db's transcripts
    table (interview/observer transcripts tagged with this cohort,
    added in Task E). A cohort with non-zero counts here can't be
    safely deleted without orphaning those references.

    Returns {"users": int, "transcripts": int}.
    """
    cohort_id = (cohort_id or "").strip()
    counts = {"users": 0, "transcripts": 0}
    if not cohort_id:
        return counts

    conn = get_connection()
    counts["users"] = conn.execute(
        "SELECT COUNT(*) FROM users WHERE cohort_id = ?", (cohort_id,)
    ).fetchone()[0]
    conn.close()

    try:
        from core.db_utils import get_connection as _get_responses_conn
        rconn = _get_responses_conn()
        try:
            counts["transcripts"] = rconn.execute(
                "SELECT COUNT(*) FROM transcripts WHERE cohort_id = ?", (cohort_id,)
            ).fetchone()[0]
        except Exception:
            pass  # transcripts table may not exist yet on a fresh DB
        rconn.close()
    except Exception:
        pass

    return counts


def delete_cohort(cohort_id: str) -> dict:
    """
    Delete cohort_id from the registry, but only if nothing currently
    references it (see count_cohort_usage) -- deleting an in-use cohort
    would silently orphan any user or transcript still tagged with it
    (they'd keep the tag, but it would vanish from every picker built
    from get_all_cohorts()).

    Returns {"deleted": bool, "users": int, "transcripts": int} -- the
    usage counts are always included so the caller can explain a
    refusal without a second query.
    """
    usage = count_cohort_usage(cohort_id)
    if usage["users"] or usage["transcripts"]:
        return {"deleted": False, **usage}

    conn = get_connection()
    conn.execute("DELETE FROM cohorts WHERE cohort_id = ?", (cohort_id,))
    conn.commit()
    conn.close()
    return {"deleted": True, **usage}


# ---------------------------------------------------------
# CREATE USER
# ---------------------------------------------------------

def create_user(admin_user, username, role, cohort_id=None, status="approved"):
    """
    Create a new user with a randomly generated, hashed password.

    Parameters
    ----------
    admin_user : str  — username of the admin performing the action
    username   : str  — new user's login name
    role       : str  — 'student' | 'teacher' | 'admin'
    cohort_id  : str | None
    status     : str  — 'approved' (default, admin-created) | 'pending'

    Returns
    -------
    str  — plaintext generated password (display once to admin, then discard)
    """
    plaintext_pw = generate_password()
    hashed_pw    = hash_password(plaintext_pw)

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO users (username, password, role, cohort_id, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, hashed_pw, role, cohort_id, status)
    )
    conn.commit()
    conn.close()

    log_admin_action(
        admin_user, AdminAction.CREATE_USER,
        f"username={username}, role={role}, cohort_id={cohort_id}"
    )
    return plaintext_pw


# ---------------------------------------------------------
# DELETE USER
# ---------------------------------------------------------

def delete_user(admin_user, username):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()
    log_admin_action(admin_user, AdminAction.DELETE_USER, f"username={username}")


# ---------------------------------------------------------
# CHANGE ROLE
# ---------------------------------------------------------

def change_role(admin_user, username, new_role):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET role = ? WHERE username = ?",
        (new_role, username)
    )
    conn.commit()
    conn.close()
    log_admin_action(
        admin_user, AdminAction.CHANGE_ROLE,
        f"username={username}, new_role={new_role}"
    )


# ---------------------------------------------------------
# UPDATE USER COHORT
# ---------------------------------------------------------

def update_user_cohort(admin_user, username, new_cohort_id):
    """Assign or reassign a user to a cohort (None clears it)."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET cohort_id = ? WHERE username = ?",
        (new_cohort_id, username)
    )
    conn.commit()
    conn.close()
    log_admin_action(
        admin_user, AdminAction.UPDATE_COHORT,
        f"username={username}, new_cohort_id={new_cohort_id}"
    )
