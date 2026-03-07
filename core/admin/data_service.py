"""
data_service.py

Administrative reset operations aligned with responses.db schema.
"""

import sqlite3
from pathlib import Path

from core.admin.audit_logger import log_admin_action, AdminAction

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "responses.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------
# RESET STUDENT DATA
# ---------------------------------------------------------

def reset_student_data(admin_user, user_id):

    conn = get_connection()
    cursor = conn.cursor()

    tables = [
        "responses",
        "completions",
        "survey_scores",
        "assessment_scores"
    ]

    for table in tables:
        cursor.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.RESET_STUDENT_DATA,
        f"user_id={user_id}"
    )


# ---------------------------------------------------------
# RESET INSTRUMENT DATA
# ---------------------------------------------------------

def reset_instrument(admin_user, instrument_name):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM responses
        WHERE instrument_name = ?
        """,
        (instrument_name,)
    )

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.RESET_INSTRUMENT,
        f"instrument={instrument_name}"
    )


# ---------------------------------------------------------
# RESET ENTIRE STUDY
# ---------------------------------------------------------

def reset_study(admin_user):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM responses")
    cursor.execute("DELETE FROM completions")
    cursor.execute("DELETE FROM survey_scores")
    cursor.execute("DELETE FROM assessment_scores")

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.RESET_STUDY,
        "all response data cleared"
    )


# ---------------------------------------------------------
# DATASET COUNTS
# ---------------------------------------------------------

def get_dataset_counts():

    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    tables = [
        "responses",
        "completions",
        "survey_scores",
        "assessment_scores"
    ]

    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        except:
            stats[table] = 0

    conn.close()

    return stats