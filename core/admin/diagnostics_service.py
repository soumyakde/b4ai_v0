"""
diagnostics_service.py

Provides system metrics and audit log data for the BasicsB4AI admin dashboard.
"""

import sqlite3
from pathlib import Path

from core.admin.data_service import get_dataset_counts
from core.admin.audit_logger import get_recent_logs

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
USERS_DB = BASE_DIR / "users.db"
RESPONSES_DB = BASE_DIR / "responses.db"


# ---------------------------------------------------------
# USER METRICS
# ---------------------------------------------------------

def get_user_stats():
    """
    Returns counts of all users by role, keyed for admin dashboard.
    """
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) FROM users")
    stats["total_users"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users WHERE role='student'")
    stats["total_students"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users WHERE role='teacher'")
    stats["total_teachers"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    stats["total_admins"] = cursor.fetchone()[0]

    conn.close()
    return stats


# ---------------------------------------------------------
# RESEARCH METRICS
# ---------------------------------------------------------

def get_research_metrics():
    """
    Returns counts of research-related tables in responses.db.
    """
    conn = sqlite3.connect(RESPONSES_DB)
    cursor = conn.cursor()

    stats = {}

    try:
        cursor.execute("SELECT COUNT(*) FROM responses")
        stats["total_responses"] = cursor.fetchone()[0]
    except sqlite3.Error:
        stats["total_responses"] = 0

    try:
        cursor.execute("SELECT COUNT(*) FROM completions")
        stats["total_completions"] = cursor.fetchone()[0]
    except sqlite3.Error:
        stats["total_completions"] = 0

    try:
        cursor.execute("SELECT COUNT(*) FROM survey_scores")
        stats["total_survey_scores"] = cursor.fetchone()[0]
    except sqlite3.Error:
        stats["total_survey_scores"] = 0

    try:
        cursor.execute("SELECT COUNT(*) FROM assessment_scores")
        stats["total_assessment_scores"] = cursor.fetchone()[0]
    except sqlite3.Error:
        stats["total_assessment_scores"] = 0

    conn.close()
    return stats


# ---------------------------------------------------------
# FULL DIAGNOSTICS
# ---------------------------------------------------------

def get_full_diagnostics():
    """
    Returns combined user and research metrics for the admin dashboard.
    """
    stats = {}
    stats.update(get_user_stats())
    stats.update(get_research_metrics())
    return stats


# ---------------------------------------------------------
# AUDIT LOG
# ---------------------------------------------------------

def get_audit_log(limit=50):
    """
    Returns recent audit log entries, limited to `limit`.
    """
    return get_recent_logs(limit)