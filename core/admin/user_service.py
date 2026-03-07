"""
user_service.py

Administrative user management operations.
Aligned with actual users.db schema.
"""

import sqlite3
from pathlib import Path

from core.admin.audit_logger import log_admin_action, AdminAction

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
        SELECT id, username, role
        FROM users
        ORDER BY username
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


# ---------------------------------------------------------
# CREATE USER
# ---------------------------------------------------------

def create_user(admin_user, username, role, password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO users (username, password, role)
        VALUES (?, ?, ?)
        """,
        (username, password, role)
    )

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.CREATE_USER,
        f"username={username}, role={role}"
    )


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