"""
auth/user_manager.py

User management module for Basics4AI — 4-role system:
  - student
  - teacher
  - admin
  - super_admin  (skde — seeded automatically on first boot)

SQLite constraint note
----------------------
The existing database table has CHECK(role IN ('student','teacher','admin')).
SQLite cannot ALTER a CHECK constraint on an existing table without
recreating it.  To avoid a destructive migration, super_admin is stored
as role='admin' in the database and the is_super_admin=1 flag is used
to signal elevated privileges.  get_user_role() returns 'super_admin'
transparently whenever that flag is set, so the rest of the app sees
the correct role string without knowing the storage detail.

Handles:
✅ Database initialization (safe schema migration — adds columns only)
✅ User registration         (status: 'pending' | 'approved' | 'rejected')
✅ Authentication
✅ Role lookup               (returns 'super_admin' via is_super_admin flag)
✅ Status lookup
✅ Super-admin seeding & permission checks
✅ Pending-user approval / rejection / bulk-approve
✅ Password change & admin password reset
✅ User deletion
✅ Cohort helpers
✅ Dataset-builder cohort map
"""

import os
import sqlite3
import hashlib
import secrets
import string
from pathlib import Path
from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
#load_dotenv(_HERE.parents[1] / ".env")
# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------
# DB_PATH = Path(os.getenv("USERS_DB_PATH", str(Path(__file__).resolve().parents[1] / "users.db")))
# Above line replaced b/c docker build test failed, smoke_test.py
DB_PATH = Path(os.getenv("USERS_DB_PATH", str(_HERE.parents[1] / "users.db")))

SUPER_ADMIN_USERNAME         = "skde"
SUPER_ADMIN_DEFAULT_PASSWORD = "ChangeMe@2025!"   # <- change after first login

# ---------------------------------------------------------------------
#  Connection helper
# ---------------------------------------------------------------------
#def get_connection() -> sqlite3.Connection:
#    """Return a new SQLite connection to the users database."""
#    return sqlite3.connect(DB_PATH)
# Above 3 lines replaced b/c docker build test failed, smoke_test.py
def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection to the users database."""
    #load_dotenv(_HERE.parents[1] / ".env")
    path = Path(os.getenv("USERS_DB_PATH", str(_HERE.parents[1] / "users.db")))
    return sqlite3.connect(path)


# ---------------------------------------------------------------------
#  Database Initialization  (safe, non-destructive migration)
# ---------------------------------------------------------------------
def init_db() -> None:
    """
    Initialise the users table and add new columns if missing.

    The original CHECK constraint (role IN student|teacher|admin) is
    intentionally preserved — see module docstring for why.
    New columns default to values that leave all existing rows valid.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # Original table definition — idempotent
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT UNIQUE NOT NULL,
                password  TEXT NOT NULL,
                role      TEXT NOT NULL CHECK(role IN ('student', 'teacher', 'admin')),
                cohort_id TEXT
            )
        """)

        # New columns — wrapped individually; fails silently if already present
        new_columns = [
            "ALTER TABLE users ADD COLUMN status         TEXT    NOT NULL DEFAULT 'approved'",
            "ALTER TABLE users ADD COLUMN is_super_admin INTEGER NOT NULL DEFAULT 0",
        ]
        for sql in new_columns:
            try:
                cursor.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists

        conn.commit()

# ---------------------------------------------------------------------
#  Password utilities
# ---------------------------------------------------------------------
def hash_password(password: str) -> str:
    """SHA-256 hash — kept identical to original for backward compatibility."""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_password(length: int = 12) -> str:
    """Generate a cryptographically random alphanumeric password."""
    characters = string.ascii_letters + string.digits
    return "".join(secrets.choice(characters) for _ in range(length))

# ---------------------------------------------------------------------
#  Custom exceptions
# ---------------------------------------------------------------------
class UserAlreadyExistsError(Exception):
    pass

# ---------------------------------------------------------------------
#  Super-admin seeding  (call once at app boot — fully idempotent)
# ---------------------------------------------------------------------
def seed_super_admin() -> None:
    """
    Ensure the super-admin account (skde) exists and is correctly flagged.

    KEY DESIGN:
      role is stored as 'admin' in the DB to satisfy the existing CHECK
      constraint (which cannot be altered without recreating the table).
      is_super_admin=1 is the actual privilege flag.
      get_user_role() maps (is_super_admin=1) -> 'super_admin' transparently.

    Safe to call on every Streamlit run — no-ops if already correct.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM users WHERE username = ?", (SUPER_ADMIN_USERNAME,)
        )
        if cursor.fetchone():
            # Account exists — set flag and status only (role stays 'admin'
            # to avoid the CHECK constraint violation)
            cursor.execute(
                """UPDATE users
                   SET    status = 'approved',
                          is_super_admin = 1
                   WHERE  username = ?""",
                (SUPER_ADMIN_USERNAME,),
            )
        else:
            # First boot — insert with role='admin' and is_super_admin=1
            cursor.execute(
                """INSERT INTO users
                       (username, password, role, cohort_id, status, is_super_admin)
                   VALUES (?, ?, 'admin', NULL, 'approved', 1)""",
                (SUPER_ADMIN_USERNAME, hash_password(SUPER_ADMIN_DEFAULT_PASSWORD)),
            )
        conn.commit()

# ---------------------------------------------------------------------
#  User registration
# ---------------------------------------------------------------------
def register_user(
    username:  str,
    password:  str,
    role:      str = "student",
    cohort_id: str | None = None,
    status:    str = "pending",      # all self-registrations start pending
) -> None:
    """
    Register a new user.

    Parameters
    ----------
    username  : unique login name
    password  : plain-text password (hashed before storage)
    role      : 'student' | 'teacher' | 'admin'
    cohort_id : optional cohort assignment
    status    : 'pending' (default for self-registration) |
                'approved' (use when an admin creates the account directly)
    """
    hashed_pw = hash_password(password)
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO users
                       (username, password, role, cohort_id, status, is_super_admin)
                   VALUES (?, ?, ?, ?, ?, 0)""",
                (username, hashed_pw, role, cohort_id, status),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise UserAlreadyExistsError(f"Username '{username}' already exists.")

# ---------------------------------------------------------------------
#  Authentication
# ---------------------------------------------------------------------
def authenticate_user(username: str, password: str) -> bool:
    """
    Verify credentials only.  Does NOT check approval status — that is
    the caller's responsibility (app.py) so the user sees a specific
    'pending' message rather than a generic auth failure.
    """
    hashed_pw = hash_password(password)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE username = ? AND password = ?",
            (username, hashed_pw),
        )
        return cursor.fetchone() is not None

# ---------------------------------------------------------------------
#  Role & status lookup
# ---------------------------------------------------------------------
def get_user_role(username: str) -> str | None:
    """
    Return the effective role string.

    If is_super_admin = 1, returns 'super_admin' regardless of the
    stored role column value (which is 'admin' due to the CHECK constraint).
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, is_super_admin FROM users WHERE username = ?",
            (username,),
        )
        result = cursor.fetchone()
    if result is None:
        return None
    role, is_sa = result
    return "super_admin" if is_sa else role


def get_user_status(username: str) -> str:
    """
    Return the account status: 'pending', 'approved', or 'rejected'.
    Defaults to 'approved' for pre-migration rows with no status value.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
    return result[0] if result else "approved"


def is_super_admin(username: str) -> bool:
    """Return True only if the account has the is_super_admin flag set."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT is_super_admin FROM users WHERE username = ?", (username,)
        )
        result = cursor.fetchone()
    return bool(result and result[0])

# ---------------------------------------------------------------------
#  Pending-user approval workflow
# ---------------------------------------------------------------------
def get_pending_users() -> list[dict]:
    """Return all users whose status is 'pending', ordered by username."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT username, role, cohort_id
               FROM   users
               WHERE  status = 'pending'
               ORDER  BY username""",
        )
        rows = cursor.fetchall()
    return [{"username": r[0], "role": r[1], "cohort_id": r[2]} for r in rows]


def approve_user(approver: str, username: str) -> None:
    """Set a user's status to 'approved'. Restricted to super_admin."""
    if not is_super_admin(approver):
        raise PermissionError("Only the super admin can approve users.")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET status = 'approved' WHERE username = ?", (username,)
        )
        conn.commit()


def reject_user(approver: str, username: str) -> None:
    """Set a user's status to 'rejected'. Restricted to super_admin."""
    if not is_super_admin(approver):
        raise PermissionError("Only the super admin can reject users.")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET status = 'rejected' WHERE username = ?", (username,)
        )
        conn.commit()


def bulk_approve(approver: str, usernames: list[str]) -> int:
    """
    Approve multiple users in a single transaction.
    Returns the number of rows updated.
    """
    if not is_super_admin(approver):
        raise PermissionError("Only the super admin can bulk-approve users.")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE users SET status = 'approved' WHERE username = ?",
            [(u,) for u in usernames],
        )
        count = cursor.rowcount
        conn.commit()
    return count

# ---------------------------------------------------------------------
#  Cohort helpers  (unchanged from original)
# ---------------------------------------------------------------------
def get_user_cohort(username: str) -> str | None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT cohort_id FROM users WHERE username=?", (username,))
        result = cursor.fetchone()
    return result[0] if result else None


def get_users_by_cohort(cohort_id: str) -> list[str]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE cohort_id=?", (cohort_id,))
        return [r[0] for r in cursor.fetchall()]


def get_user_cohort_map() -> dict[str, str | None]:
    """
    Return a dict mapping every username to its cohort_id (None if unassigned).

    Example
    -------
    {
        "student1": "cohort_A",
        "teacher1": "faculty_X",
        "student2": None,
    }
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, cohort_id FROM users")
        rows = cursor.fetchall()
    return {username: cohort_id for username, cohort_id in rows}

# ---------------------------------------------------------------------
#  General user helpers
# ---------------------------------------------------------------------
def list_users() -> list[dict]:
    """Return all users (username, role, cohort_id, status)."""
    if not DB_PATH.exists():
        return []
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, role, cohort_id, status FROM users")
        rows = cursor.fetchall()
    return [
        {"username": r[0], "role": r[1], "cohort_id": r[2], "status": r[3]}
        for r in rows
    ]


def delete_user(username: str) -> bool:
    """Delete a user. Returns True if a row was removed."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username=?", (username,))
        deleted = cursor.rowcount > 0
        conn.commit()
    return deleted

# ---------------------------------------------------------------------
#  Password management
# ---------------------------------------------------------------------
def change_password(username: str, old_password: str, new_password: str) -> str:
    """User-initiated password change. Returns a status message."""
    if not authenticate_user(username, old_password):
        return "Incorrect current password."
    if old_password == new_password:
        return "New password must differ from the old password."
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password=? WHERE username=?",
            (hash_password(new_password), username),
        )
        conn.commit()
    return "Password updated successfully."


def reset_password(admin_username: str, target_user: str) -> str | None:
    """
    Admin-initiated password reset.
    Returns the new plain-text password, or None on failure.
    Permitted for both 'admin' and 'super_admin' roles.
    """
    role = get_user_role(admin_username)
    if role not in ("admin", "super_admin"):
        return None
    new_pw    = generate_password()
    hashed_pw = hash_password(new_pw)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password=? WHERE username=?",
            (hashed_pw, target_user),
        )
        if cursor.rowcount == 0:
            return None
        conn.commit()
    return new_pw

# ---------------------------------------------------------------------
#  Script helper
# ---------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    seed_super_admin()
    print(f"✅ Database ready at {DB_PATH}")
    print(f"✅ Super admin '{SUPER_ADMIN_USERNAME}' seeded.")
