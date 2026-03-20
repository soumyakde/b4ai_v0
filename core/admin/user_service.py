"""
core/admin/user_service.py

Administrative user management operations.
Aligned with actual users.db schema.
"""

import sqlite3
from pathlib import Path

from core.admin.audit_logger import log_admin_action, AdminAction
from auth.user_manager import hash_password, generate_password

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "users.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------
# GET ALL USERS
# ---------------------------------------------------------

def get_all_users():

    conn = get_connection()
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
    return pd.DataFrame(
        rows, columns=["id", "username", "role", "cohort_id"]
    )

# ---------------------------------------------------------
# COHORT FETCH
# ---------------------------------------------------------
def get_all_cohorts():
    import sqlite3
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT cohort_id FROM cohorts ORDER BY cohort_id")
    cohorts = [row[0] for row in cursor.fetchall()]
    conn.close()
    return cohorts


# ---------------------------------------------------------
# CREATE USER
# ---------------------------------------------------------

def create_user(admin_user, username, role, cohort_id=None):
    """
    Create a new user with a randomly generated, hashed password.

    Returns
    -------
    str
        The plaintext generated password — show this once to the admin.
    """
    plaintext_pw = generate_password()
    hashed_pw    = hash_password(plaintext_pw)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO users (username, password, role, cohort_id)
        VALUES (?, ?, ?, ?)
        """,
        (username, hashed_pw, role, cohort_id)
    )

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.CREATE_USER,
        f"username={username}, role={role}, cohort_id={cohort_id}"
    )

    return plaintext_pw


# ---------------------------------------------------------
# DELETE USER
# ---------------------------------------------------------

def delete_user(admin_user, username):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM users WHERE username=?",
        (username,)
    )

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.DELETE_USER,
        f"username={username}"
    )


# ---------------------------------------------------------
# CHANGE ROLE
# ---------------------------------------------------------

def change_role(admin_user, username, new_role):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET role = ?
        WHERE username = ?
        """,
        (new_role, username)
    )

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.CHANGE_ROLE,
        f"username={username}, new_role={new_role}"
    )
# ---------------------------------------------------------
# UPDATE USER COHORT
# ---------------------------------------------------------

def update_user_cohort(admin_user, username, new_cohort_id):
    """
    Assign or reassign a user to a cohort.
    If new_cohort_id is None, the user's cohort is cleared.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET cohort_id = ?
        WHERE username = ?
        """,
        (new_cohort_id, username)
    )

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.UPDATE_COHORT,
        f"username={username}, new_cohort_id={new_cohort_id}"
    )