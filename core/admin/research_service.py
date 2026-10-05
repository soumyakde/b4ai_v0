"""
research_service.py

Research operations for the BasicsB4AI platform.

Supports:
- instrument discovery
- research dataset preparation
"""

import os
import sqlite3
from pathlib import Path

from core.admin.audit_logger import log_admin_action, AdminAction
from dotenv import load_dotenv
# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
RESPONSES_DB = Path(os.getenv("SQLITE_PATH",   str(BASE_DIR / "responses.db")))
INSTRUMENTS_DIR = BASE_DIR / "streamlit_app" / "surveys"

# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------
# --survey scores zero despite submitting surveys
#def get_connection():
#    return sqlite3.connect(RESPONSES_DB)
from core.db_utils import get_connection

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
    Exports the item-level research dataset (one row per response) for
    statistical analysis, from the same canonical scored data the dashboards use.

    Columns: user_id, module_id, instrument_key, question_id, response_value,
    item_score, construct, grade, submitted_at, completed_at, cohort_id.

    It deliberately does not read the survey_scores/assessment_scores tables:
    those hold submission-time scores only, and the previous query joined them
    on user_id alone, multiplying each response row by every score and
    completion row of the same student (about 17.6 million rows from 30,789
    responses on the pilot database).
    """
    from core.analytics.datasets.canonical_loader import load_canonical_data

    canonical_df, _, _ = load_canonical_data()
    canonical_df = canonical_df.sort_values(
        ["user_id", "instrument_key", "question_id"]
    )

    output_path = BASE_DIR / "exports" / "research_dataset.csv"
    output_path.parent.mkdir(exist_ok=True)
    canonical_df.to_csv(output_path, index=False, encoding="utf-8")

    log_admin_action(
        admin_user,
        AdminAction.RUN_DIAGNOSTICS,
        f"research dataset exported with {len(canonical_df)} rows"
    )

    return output_path