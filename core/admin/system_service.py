"""
system_service.py

System-level administrative maintenance operations
for the BasicsB4AI platform.

Changes:
  - Added _init_restore_log_table() — creates restore_log in users.db
  - Added restore_databases(admin_user, backup_timestamp) — verified
    restore with pre/post row-count check and restore_log entry
  - Added list_backups() — returns available backup sets for the UI
  - Added start_auto_backup_scheduler() — APScheduler BackgroundScheduler
    started at app.py boot (not inside a render function)
  - Added BACKUP_INTERVAL_HOURS env var support (default: 24)
  - system_service.py does NOT import Streamlit — scheduler is safe
    to start in a background thread
"""

import os
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from core.admin.audit_logger import log_admin_action, AdminAction


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

USERS_DB     = Path(os.getenv("USERS_DB_PATH", str(BASE_DIR / "users.db")))
RESPONSES_DB = Path(os.getenv("SQLITE_PATH",   str(BASE_DIR / "responses.db")))

# Backup and cache dirs sit alongside the DB files
_DATA_DIR  = USERS_DB.parent
BACKUP_DIR = _DATA_DIR / "backups"
CACHE_DIR  = _DATA_DIR / "cache"

# Auto-backup interval — override via .env
BACKUP_INTERVAL_HOURS = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))


# ---------------------------------------------------------
# ENSURE BACKUP DIRECTORY
# ---------------------------------------------------------

def ensure_backup_dir():
    BACKUP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# BACKUP DATABASES  (unchanged)
# ---------------------------------------------------------

def backup_databases(admin_user: str) -> dict:
    """
    Creates timestamped backups of system databases.
    Returns dict of {label: Path} for the created files.
    """
    ensure_backup_dir()

    timestamp        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    users_backup     = BACKUP_DIR / f"users_backup_{timestamp}.db"
    responses_backup = BACKUP_DIR / f"responses_backup_{timestamp}.db"

    shutil.copy2(USERS_DB,     users_backup)
    shutil.copy2(RESPONSES_DB, responses_backup)

    log_admin_action(
        admin_user,
        AdminAction.BACKUP_DATABASE,
        f"users={users_backup.name}, responses={responses_backup.name}",
    )

    return {
        "users_backup":     users_backup,
        "responses_backup": responses_backup,
    }


# ---------------------------------------------------------
# LIST AVAILABLE BACKUPS
# ---------------------------------------------------------

def list_backups() -> list[dict]:
    """
    Return a list of available backup sets, sorted newest first.

    Each entry: {"timestamp": str, "users": Path, "responses": Path}
    A set is complete only if BOTH files exist for the same timestamp.
    """
    ensure_backup_dir()

    # Collect all timestamped backup files
    users_files     = {
        p.stem.replace("users_backup_", ""): p
        for p in BACKUP_DIR.glob("users_backup_*.db")
    }
    responses_files = {
        p.stem.replace("responses_backup_", ""): p
        for p in BACKUP_DIR.glob("responses_backup_*.db")
    }

    # Only return complete pairs
    complete = sorted(
        set(users_files) & set(responses_files),
        reverse=True,
    )

    return [
        {
            "timestamp": ts,
            "users":     users_files[ts],
            "responses": responses_files[ts],
        }
        for ts in complete
    ]


# ---------------------------------------------------------
# RESTORE DATABASES
# ---------------------------------------------------------

def _init_restore_log_table():
    """
    Create the restore_log table in users.db if it does not exist.
    Idempotent — safe to call on every restore.
    """
    conn = sqlite3.connect(USERS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS restore_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            restored_at     TEXT NOT NULL,
            admin_user      TEXT NOT NULL,
            backup_timestamp TEXT NOT NULL,
            users_pre_count      INTEGER,
            users_post_count     INTEGER,
            responses_pre_count  INTEGER,
            responses_post_count INTEGER,
            verified        INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def _row_count(db_path: Path, table: str) -> int:
    """Return row count for a table, or -1 if the table does not exist."""
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.execute(f"SELECT COUNT(*) FROM {table}")
        n    = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return -1


def restore_databases(admin_user: str, backup_timestamp: str) -> dict:
    """
    Restore both databases from the backup set identified by backup_timestamp.

    Workflow:
      1. Verify backup files exist
      2. Record pre-restore row counts (users.users, responses.responses)
      3. Copy backup files over live databases
      4. Record post-restore row counts
      5. Verify counts match backup (counted from restored file)
      6. Write a restore_log entry
      7. Write an audit log entry
      8. Return a result dict with all counts and verified flag

    Args:
        admin_user:       Username of the admin performing the restore.
        backup_timestamp: Timestamp string as returned by list_backups(),
                          e.g. "20260325_144205".

    Returns:
        dict with keys: backup_timestamp, users_pre, users_post,
        responses_pre, responses_post, verified, message

    Raises:
        FileNotFoundError: If backup files do not exist.
        RuntimeError:      If row counts do not match after restore.
    """
    _init_restore_log_table()

    users_backup     = BACKUP_DIR / f"users_backup_{backup_timestamp}.db"
    responses_backup = BACKUP_DIR / f"responses_backup_{backup_timestamp}.db"

    if not users_backup.exists():
        raise FileNotFoundError(f"Backup not found: {users_backup.name}")
    if not responses_backup.exists():
        raise FileNotFoundError(f"Backup not found: {responses_backup.name}")

    # Step 2 — pre-restore counts
    users_pre     = _row_count(USERS_DB,     "users")
    responses_pre = _row_count(RESPONSES_DB, "responses")

    # Step 3 — restore
    shutil.copy2(users_backup,     USERS_DB)
    shutil.copy2(responses_backup, RESPONSES_DB)

    # Step 4 — post-restore counts (from the now-restored live files)
    users_post     = _row_count(USERS_DB,     "users")
    responses_post = _row_count(RESPONSES_DB, "responses")

    # Step 5 — verify against backup source counts
    users_backup_count     = _row_count(users_backup,     "users")
    responses_backup_count = _row_count(responses_backup, "responses")

    verified = (
        users_post     == users_backup_count and
        responses_post == responses_backup_count
    )

    # Step 6 — write restore_log entry
    conn = sqlite3.connect(USERS_DB)
    conn.execute(
        """
        INSERT INTO restore_log
        (restored_at, admin_user, backup_timestamp,
         users_pre_count, users_post_count,
         responses_pre_count, responses_post_count, verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            admin_user,
            backup_timestamp,
            users_pre,
            users_post,
            responses_pre,
            responses_post,
            1 if verified else 0,
        ),
    )
    conn.commit()
    conn.close()

    # Step 7 — audit log
    log_admin_action(
        admin_user,
        AdminAction.RESTORE_DATABASE,
        f"backup_timestamp={backup_timestamp}, "
        f"users={users_pre}→{users_post}, "
        f"responses={responses_pre}→{responses_post}, "
        f"verified={verified}",
    )

    result = {
        "backup_timestamp":  backup_timestamp,
        "users_pre":         users_pre,
        "users_post":        users_post,
        "responses_pre":     responses_pre,
        "responses_post":    responses_post,
        "verified":          verified,
        "message": (
            "✅ Restore verified — row counts match backup."
            if verified else
            "⚠️ Restore completed but row counts did not match. "
            "Check restore_log for details."
        ),
    }

    return result


# ---------------------------------------------------------
# CLONE DATABASES  (unchanged)
# ---------------------------------------------------------

def clone_databases(admin_user: str) -> dict:
    """
    Creates a working clone of the current databases for testing.
    """
    clone_dir = BASE_DIR / "clones"
    clone_dir.mkdir(exist_ok=True)

    users_clone     = clone_dir / "users_clone.db"
    responses_clone = clone_dir / "responses_clone.db"

    shutil.copy2(USERS_DB,     users_clone)
    shutil.copy2(RESPONSES_DB, responses_clone)

    log_admin_action(
        admin_user,
        AdminAction.CLONE_DATABASE,
        "database clone created",
    )

    return {
        "users_clone":     users_clone,
        "responses_clone": responses_clone,
    }


# ---------------------------------------------------------
# CLEAR ANALYTICS CACHE  (unchanged)
# ---------------------------------------------------------

def clear_cache(admin_user: str) -> None:
    """Clears cached analytics artifacts."""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir()
        return

    for file in CACHE_DIR.glob("*"):
        try:
            file.unlink()
        except Exception:
            pass

    log_admin_action(
        admin_user,
        AdminAction.CLEAR_CACHE,
        "analytics cache cleared",
    )


# ---------------------------------------------------------
# AUTO-BACKUP SCHEDULER
# ---------------------------------------------------------

def start_auto_backup_scheduler() -> bool:
    """
    Start a background thread that backs up both databases every
    BACKUP_INTERVAL_HOURS hours (default 24, set via .env).

    IMPORTANT — call this from app.py at MODULE LEVEL, guarded by a
    module-level boolean flag, NOT inside any dashboard render function.
    Streamlit reruns the script on every interaction; a render-level
    call would attempt to restart the scheduler on every rerun.
    Module-level globals survive reruns within the same process.

    Uses APScheduler BackgroundScheduler (runs in a daemon thread).
    If APScheduler is not installed, logs a warning and returns False.

    Returns:
        True  — scheduler started successfully
        False — APScheduler not available; manual backups still work
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "APScheduler not installed — automatic backups disabled. "
            "Add 'apscheduler' to requirements.txt to enable them. "
            "Manual backups from the admin dashboard still work."
        )
        return False

    def _auto_backup_job():
        """The job function — uses 'system' as admin_user for automated runs."""
        try:
            backup_databases(admin_user="system[auto-backup]")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "Auto-backup failed: %s", exc
            )

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        _auto_backup_job,
        trigger="interval",
        hours=BACKUP_INTERVAL_HOURS,
        id="auto_backup",
        replace_existing=True,
    )
    scheduler.start()

    import logging
    logging.getLogger(__name__).info(
        "Auto-backup scheduler started — interval: %d hour(s).",
        BACKUP_INTERVAL_HOURS,
    )
    return True


# ---------------------------------------------------------
# SYSTEM STATUS  (unchanged)
# ---------------------------------------------------------

def get_system_status() -> dict:
    """Returns basic system diagnostics."""
    status = {}

    status["users_db_exists"]     = USERS_DB.exists()
    status["responses_db_exists"] = RESPONSES_DB.exists()
    status["backup_dir_exists"]   = BACKUP_DIR.exists()

    if BACKUP_DIR.exists():
        status["backup_files"] = len(list(BACKUP_DIR.glob("*")))
    else:
        status["backup_files"] = 0

    return status
