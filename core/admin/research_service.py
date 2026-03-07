"""
research_service.py

Research operations for the BasicsB4AI platform.

Supports:
- instrument discovery
- research dataset preparation
"""

import sqlite3
import csv
from pathlib import Path

from core.admin.audit_logger import log_admin_action, AdminAction

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RESPONSES_DB = BASE_DIR / "responses.db"
INSTRUMENTS_DIR = BASE_DIR / "streamlit_app" / "surveys"

# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

def get_connection():
    return sqlite3.connect(RESPONSES_DB)

# ---------------------------------------------------------
# DISCOVER LOADED INSTRUMENTS
# ---------------------------------------------------------

def get_loaded_instruments():
    """
    Returns list of YAML instruments loaded in system.
    """
    instruments = []

    if not INSTRUMENTS_DIR.exists():
        return instruments

    for file in INSTRUMENTS_DIR.glob("*.yaml"):
        instruments.append(file.stem)

    return sorted(instruments)

# ---------------------------------------------------------
# EXPORT RESEARCH DATASET
# ---------------------------------------------------------

def export_research_dataset(admin_user):
    """
    Generates a research dataset for statistical analysis.
    Combines responses, completions, survey_scores, and assessment_scores.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            r.user_id,
            r.instrument_name,
            r.question_id,
            r.response_value,
            r.submitted_at,
            c.module_id,
            c.instrument_key,
            c.completed_at,
            s.survey_key,
            s.score AS survey_score,
            s.calculated_at AS survey_calculated_at,
            a.assessment_code,
            a.score AS assessment_score,
            a.calculated_at AS assessment_calculated_at
        FROM responses r
        LEFT JOIN completions c
            ON r.user_id = c.user_id
        LEFT JOIN survey_scores s
            ON r.user_id = s.user_id
        LEFT JOIN assessment_scores a
            ON r.user_id = a.user_id
        ORDER BY r.user_id, r.instrument_name, r.question_id
    """)

    rows = cursor.fetchall()
    conn.close()

    output_path = BASE_DIR / "exports" / "research_dataset.csv"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "user_id",
            "instrument_name",
            "question_id",
            "response_value",
            "submitted_at",
            "module_id",
            "instrument_key",
            "completed_at",
            "survey_key",
            "survey_score",
            "survey_calculated_at",
            "assessment_code",
            "assessment_score",
            "assessment_calculated_at"
        ])

        writer.writerows(rows)

    log_admin_action(
        admin_user,
        AdminAction.RUN_DIAGNOSTICS,
        f"research dataset exported with {len(rows)} rows"
    )

    return output_path