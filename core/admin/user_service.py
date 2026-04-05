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
        SELECT id, username, role, cohort_id
        FROM users
        ORDER BY role, username
        """
    )
    rows = cursor.fetchall()
    conn.close()

    import pandas as pd
    return pd.DataFrame(rows, columns=["id", "username", "role", "cohort_id"])


# ---------------------------------------------------------
# COHORT FETCH  (fixed: was using hardcoded relative path)
# ---------------------------------------------------------

def get_all_cohorts():
    conn   = get_connection()          # uses DB_PATH, not bare "users.db"
    cursor = conn.cursor()
    cursor.execute("SELECT cohort_id FROM cohorts ORDER BY cohort_id")
    cohorts = [row[0] for row in cursor.fetchall()]
    conn.close()
    return cohorts


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
