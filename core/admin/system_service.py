"""
system_service.py

System-level administrative maintenance operations
for the BasicsB4AI platform.
"""

import shutil
from pathlib import Path
from datetime import datetime

from core.admin.audit_logger import log_admin_action, AdminAction


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

#DATA_DIR = BASE_DIR / "data"
DATA_DIR = BASE_DIR 
BACKUP_DIR = BASE_DIR / "backups"
CACHE_DIR = BASE_DIR / "cache"

USERS_DB = DATA_DIR / "users.db"
RESPONSES_DB = DATA_DIR / "responses.db"


# ---------------------------------------------------------
# ENSURE BACKUP DIRECTORY
# ---------------------------------------------------------

def ensure_backup_dir():

    BACKUP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# BACKUP DATABASES
# ---------------------------------------------------------

def backup_databases(admin_user):
    """
    Creates timestamped backups of system databases.
    """

    ensure_backup_dir()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    users_backup = BACKUP_DIR / f"users_backup_{timestamp}.db"
    responses_backup = BACKUP_DIR / f"responses_backup_{timestamp}.db"

    shutil.copy2(USERS_DB, users_backup)
    shutil.copy2(RESPONSES_DB, responses_backup)

    log_admin_action(
        admin_user,
        AdminAction.BACKUP_DATABASE,
        f"users={users_backup.name}, responses={responses_backup.name}"
    )

    return {
        "users_backup": users_backup,
        "responses_backup": responses_backup
    }


# ---------------------------------------------------------
# CLONE DATABASES
# ---------------------------------------------------------

def clone_databases(admin_user):
    """
    Creates a working clone of the current databases
    for testing or analysis.
    """

    clone_dir = BASE_DIR / "clones"
    clone_dir.mkdir(exist_ok=True)

    users_clone = clone_dir / "users_clone.db"
    responses_clone = clone_dir / "responses_clone.db"

    shutil.copy2(USERS_DB, users_clone)
    shutil.copy2(RESPONSES_DB, responses_clone)

    log_admin_action(
        admin_user,
        AdminAction.CLONE_DATABASE,
        "database clone created"
    )

    return {
        "users_clone": users_clone,
        "responses_clone": responses_clone
    }


# ---------------------------------------------------------
# CLEAR ANALYTICS CACHE
# ---------------------------------------------------------

def clear_cache(admin_user):
    """
    Clears cached analytics artifacts.
    """

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
        "analytics cache cleared"
    )


# ---------------------------------------------------------
# SYSTEM STATUS
# ---------------------------------------------------------

def get_system_status():
    """
    Returns basic system diagnostics.
    """

    status = {}

    status["users_db_exists"] = USERS_DB.exists()
    status["responses_db_exists"] = RESPONSES_DB.exists()
    status["backup_dir_exists"] = BACKUP_DIR.exists()

    if BACKUP_DIR.exists():
        status["backup_files"] = len(list(BACKUP_DIR.glob("*")))
    else:
        status["backup_files"] = 0

    return status