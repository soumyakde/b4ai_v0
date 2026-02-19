"""
user_manager.py

User management module for 3-role system:
- student
- teacher
- admin

Handles:
✅ Database initialization
✅ User registration
✅ Authentication
✅ Role lookup
✅ Password change
✅ Admin password reset
✅ User deletion
"""

import sqlite3
from pathlib import Path
import hashlib
import secrets
import string

# ---------------------------------------------------------------------
#  Database Configuration
# ---------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parents[1] / "users.db"


def get_connection():
    """Return a new SQLite connection."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Initialize database with proper 3-role schema.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('student', 'teacher', 'admin'))
            )
        """)

        conn.commit()


# ---------------------------------------------------------------------
#  Password Utilities
# ---------------------------------------------------------------------

def hash_password(password: str) -> str:
    """
    Hash password using SHA256.
    (Later we can upgrade to bcrypt or pbkdf2 for production.)
    """
    return hashlib.sha256(password.encode()).hexdigest()


def generate_password(length: int = 12) -> str:
    """Generate a secure random password."""
    characters = string.ascii_letters + string.digits
    return "".join(secrets.choice(characters) for _ in range(length))


# ---------------------------------------------------------------------
#  User Management
# ---------------------------------------------------------------------

class UserAlreadyExistsError(Exception):
    """Raised when attempting to register a duplicate username."""
    pass


def register_user(username: str, password: str, role: str = "student"):
    """
    Register a new user with specified role.
    Default role is 'student'.
    """
    hashed_pw = hash_password(password)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed_pw, role)
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise UserAlreadyExistsError("Username already exists.")


def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticate user credentials.
    Returns True if valid.
    """
    hashed_pw = hash_password(password)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE username=? AND password=?",
            (username, hashed_pw)
        )
        return cursor.fetchone() is not None


def get_user_role(username: str) -> str | None:
    """
    Return the role of a given user.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role FROM users WHERE username=?",
            (username,)
        )
        result = cursor.fetchone()

    return result[0] if result else None


def list_users() -> list[dict]:
    """
    Return list of all users with roles.
    """
    if not DB_PATH.exists():
        return []

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, role FROM users")
        rows = cursor.fetchall()

    return [
        {"username": r[0], "role": r[1]}
        for r in rows
    ]


def delete_user(username: str) -> bool:
    """
    Delete a user account.
    Returns True if user was removed.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username=?", (username,))
        deleted = cursor.rowcount > 0
        conn.commit()

    return deleted


# ---------------------------------------------------------------------
#  Password Change / Reset
# ---------------------------------------------------------------------

def change_password(username: str, old_password: str, new_password: str) -> str:
    """
    Allow a user to change their password.
    """
    if not authenticate_user(username, old_password):
        return "Incorrect current password."

    if old_password == new_password:
        return "New password must differ from the old password."

    hashed_new = hash_password(new_password)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password=? WHERE username=?",
            (hashed_new, username)
        )
        conn.commit()

    return "Password updated successfully."


def reset_password(admin_username: str, target_user: str) -> str | None:
    """
    Admin-only: reset another user's password.
    Returns new password if successful.
    """
    if get_user_role(admin_username) != "admin":
        return None

    new_pw = generate_password()
    hashed_pw = hash_password(new_pw)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password=? WHERE username=?",
            (hashed_pw, target_user)
        )
        if cursor.rowcount == 0:
            return None
        conn.commit()

    return new_pw


# ---------------------------------------------------------------------
#  Script Helper
# ---------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    print(f"✅ Database ready at {DB_PATH}")