"""
core/admin/data_service.py

Administrative reset operations aligned with responses.db schema.

Bug fixes applied:
  1. Removed private get_connection() that hardcoded "responses.db" and
     ignored the SQLITE_PATH env var. All operations now use
     db_utils.get_connection() — the single authoritative connection that
     respects SQLITE_PATH. This ensures resets hit the same database that
     the rest of the application writes to (e.g. research.db when
     SQLITE_PATH=research.db is set in .env).

  2. reset_instrument() now clears all four tables for the given instrument,
     not just "responses". Previously, completions/survey_scores/
     assessment_scores were left intact, so students could not retake a
     reset instrument (the dashboard still showed ✅).
"""

from pathlib import Path

from core.admin.audit_logger import log_admin_action, AdminAction
from core.db_utils import get_connection  # ← single source of truth for DB path


# ---------------------------------------------------------
# RESET STUDENT DATA
# ---------------------------------------------------------

def reset_student_data(admin_user: str, user_id: str) -> None:
    """
    Delete ALL data for a single student across all four tables.
    After this call the student's dashboard shows every module as
    locked/incomplete and they can retake all instruments from scratch.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    tables = [
        "responses",
        "completions",
        "survey_scores",
        "assessment_scores",
    ]

    for table in tables:
        cursor.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.RESET_STUDENT_DATA,
        f"user_id={user_id}",
    )


# ---------------------------------------------------------
# COUNT STUDENT DATA FOOTPRINT (no deletion — preview only)
# ---------------------------------------------------------

def count_student_data_footprint(user_id: str) -> dict:
    """
    Return row counts across all four data tables for a single student,
    without deleting anything. Same table list as reset_student_data() —
    used to detect "zero saved data" candidates and to preview exactly
    what a bulk cleanup would remove before it happens.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    tables = [
        "responses",
        "completions",
        "survey_scores",
        "assessment_scores",
    ]

    counts = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,))
        counts[table] = cursor.fetchone()[0]

    conn.close()
    return counts


# ---------------------------------------------------------
# RESET INSTRUMENT DATA
# ---------------------------------------------------------

def reset_instrument(admin_user: str, instrument_name: str) -> None:
    """
    Delete ALL data for a single instrument across ALL students.

    Clears four tables so that:
      - Raw responses are gone (responses)
      - Completion flags are gone so students can retake (completions)
      - Computed scores are gone (survey_scores, assessment_scores)

    The instrument_name / instrument_key / survey_key / assessment_code
    are all the same value at the admin level — the key the admin types
    into the "Instrument ID" field.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # Raw responses
    cursor.execute(
        "DELETE FROM responses WHERE instrument_name = ?",
        (instrument_name,),
    )

    # Completion flags  (instrument_key matches instrument_name)
    cursor.execute(
        "DELETE FROM completions WHERE instrument_key = ?",
        (instrument_name,),
    )

    # Survey scores  (survey_key matches instrument_name for survey instruments)
    cursor.execute(
        "DELETE FROM survey_scores WHERE survey_key = ?",
        (instrument_name,),
    )

    # Assessment scores  (assessment_code matches instrument_name for MCQ instruments)
    cursor.execute(
        "DELETE FROM assessment_scores WHERE assessment_code = ?",
        (instrument_name,),
    )

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.RESET_INSTRUMENT,
        f"instrument={instrument_name}",
    )


# ---------------------------------------------------------
# RESET ONE INSTRUMENT FOR ONE STUDENT
# ---------------------------------------------------------

def reset_user_instrument(admin_user: str, user_id: str, instrument_name: str) -> None:
    """
    Delete data for a single instrument, for a single student only.

    Combines the WHERE clauses of reset_student_data() (user scope) and
    reset_instrument() (instrument scope) — same four tables, same
    delete pattern, just intersected. Use this when a participant
    answered one instrument (e.g. a module they jumped ahead into)
    that needs to be undone without touching their legitimate data
    from every other module/instrument.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM responses WHERE user_id = ? AND instrument_name = ?",
        (user_id, instrument_name),
    )
    cursor.execute(
        "DELETE FROM completions WHERE user_id = ? AND instrument_key = ?",
        (user_id, instrument_name),
    )
    cursor.execute(
        "DELETE FROM survey_scores WHERE user_id = ? AND survey_key = ?",
        (user_id, instrument_name),
    )
    cursor.execute(
        "DELETE FROM assessment_scores WHERE user_id = ? AND assessment_code = ?",
        (user_id, instrument_name),
    )

    conn.commit()
    conn.close()

    log_admin_action(
        admin_user,
        AdminAction.RESET_USER_INSTRUMENT,
        f"user_id={user_id}, instrument={instrument_name}",
    )


def count_user_instrument_rows(user_id: str, instrument_name: str) -> dict:
    """
    Return row counts that reset_user_instrument() would delete, without
    deleting anything — used for a live preview before the admin confirms.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    counts = {}
    for table, col in [
        ("responses", "instrument_name"),
        ("completions", "instrument_key"),
        ("survey_scores", "survey_key"),
        ("assessment_scores", "assessment_code"),
    ]:
        cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE user_id = ? AND {col} = ?",
            (user_id, instrument_name),
        )
        counts[table] = cursor.fetchone()[0]

    conn.close()
    return counts


# ---------------------------------------------------------
# RESET ENTIRE STUDY
# ---------------------------------------------------------

def reset_study(admin_user: str) -> None:
    """
    Delete ALL research data across ALL students and ALL instruments.
    Requires the admin to type RESET in the confirmation box (enforced
    in admin_dashboard.py — not re-enforced here).
    """
    conn   = get_connection()
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
        "all response data cleared",
    )


# ---------------------------------------------------------
# DATASET COUNTS
# ---------------------------------------------------------

def get_dataset_counts() -> dict:
    """
    Return row counts for all four core tables.
    Used by diagnostics and admin dashboard metrics.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    stats  = {}
    tables = [
        "responses",
        "completions",
        "survey_scores",
        "assessment_scores",
    ]

    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row = cursor.fetchone()
            # db_utils sets row_factory = sqlite3.Row; fetchone()[0] works for both
            stats[table] = row[0] if row else 0
        except Exception:
            stats[table] = 0

    conn.close()
    return stats
