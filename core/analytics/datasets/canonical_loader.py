# core/analytics/datasets/canonical_loader.py
"""
Canonical Data Loader
=====================
Single entry point for building the canonical research dataset
from the project's SQLite database and YAML instrument definitions.

Returns three objects consumed by teacher_dashboard.py:
    canonical_df    — scored, metadata-enriched DataFrame
    demographics_df — per-user grade / gender / language
    cohort_map      — {user_id: cohort_id | None}

Architecture rules:
- No Streamlit imports — pure data loading, fully testable
- Uses db_utils.get_connection() for DB access (shared path)
- Uses DatasetBuilder for deterministic canonical construction
- Instruments dict and scoring dict keyed by EXACT DB instrument_name
  values so DatasetBuilder receives them without pre-resolution
- Caching (st.cache_data) is the CALLER's responsibility

File layout assumptions (derived from __file__ path):
    core/analytics/datasets/canonical_loader.py
    → parents[3] = project root
    → project_root/responses.db
    → project_root/streamlit_app/surveys/*.yaml
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sqlite3

import yaml
import pandas as pd

from core.analytics.datasets.dataset_builder import DatasetBuilder
from core.analytics.filters.demographics_extractor import extract_demographics
from auth.user_manager import get_user_cohort_map
from modules.registry.discover import discover_all_module_numbers as _all_module_numbers

# -----------------------------------------------------------------------
# Project-relative paths (resolved from this file's location)
# -----------------------------------------------------------------------
_HERE        = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]          # core/analytics/datasets → root
_SURVEYS_DIR  = _PROJECT_ROOT / "streamlit_app" / "surveys"
_DB_PATH      = _PROJECT_ROOT / "responses.db"


# -----------------------------------------------------------------------
# Internal YAML loader
# -----------------------------------------------------------------------

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML file from streamlit_app/surveys/."""
    path = _SURVEYS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {filename}")
    return data


# -----------------------------------------------------------------------
# Internal: build instruments_dict
# Keyed by EXACT DB instrument_name values so DatasetBuilder
# can look them up directly after _normalize_schema().
# -----------------------------------------------------------------------

def _build_instruments_dict() -> Dict[str, Any]:
    """
    Build instruments_dict for DatasetBuilder.

    Each entry maps a DB instrument_name to its YAML metadata
    (with module_id overridden to the actual module number).
    DatasetBuilder uses this for:
      - module_id assignment
      - construct mapping (sections/questions → construct name)
      - question_bank for MCQ metadata
    """
    instruments = {}

    # ---- Surveys: SCCCES and SIMS (shared YAML, per-module module_id) ----
    sccces_yaml = _load_yaml("b4ai_sccces_survey.yaml")
    sims_yaml   = _load_yaml("b4ai_sims_survey.yaml")

    for n in _all_module_numbers():
        instruments[f"module{n}_b4ai_sccces_survey"] = {
            **sccces_yaml,
            "module_id": f"module{n}",   # override YAML global → actual module
        }
        instruments[f"module{n}_b4ai_sims_survey"] = {
            **sims_yaml,
            "module_id": f"module{n}",
        }

    # ---- MCQ content assessments (per-module question bank from scoring YAML) ----
    for n in _all_module_numbers():
        try:
            scoring = _load_yaml(f"module{n}_content_mcq_assessment_scoring.yaml")
            question_bank = [
                {"id": qid}
                for qid in scoring["correct_answers"].keys()
            ]
        except FileNotFoundError:
            question_bank = []

        instruments[f"module{n}_content_mcq_assessment"] = {
            "module_id": f"module{n}",
            "question_bank": question_bank,
        }

    # ---- Pre/post assessments (global scope) ----
    for yaml_name, db_name in [
        ("pre_ai_misconceptions_assessment.yaml",
         "precourse_pre_ai_misconceptions_assessment"),
        ("post_ai_misconceptions_assessment.yaml",
         "postcourse_post_ai_misconceptions_assessment"),
        ("pre_aici_assessment.yaml",
         "precourse_pre_aici_assessment"),
        ("post_aici_assessment.yaml",
         "postcourse_post_aici_assessment"),
    ]:
        try:
            instr_yaml = _load_yaml(yaml_name)
            instruments[db_name] = {**instr_yaml, "module_id": "global"}
        except FileNotFoundError:
            instruments[db_name] = {"module_id": "global"}

    # ---- Demographics ----
    try:
        demo_yaml = _load_yaml("demographics_survey.yaml")
        instruments["precourse_demographics_survey"] = {
            **demo_yaml,
            "module_id": "demographics",
        }
    except FileNotFoundError:
        instruments["precourse_demographics_survey"] = {"module_id": "demographics"}

    # ---- Module reflections (no scoring — module_id only) ----
    try:
        refl_yaml = _load_yaml("module_reflections.yaml")
        for n in _all_module_numbers():
            instruments[f"module{n}_module_reflections"] = {
                **refl_yaml,
                "module_id": f"module{n}",
            }
    except FileNotFoundError:
        for n in _all_module_numbers():
            instruments[f"module{n}_module_reflections"] = {
                "module_id": f"module{n}",
            }

    return instruments


# -----------------------------------------------------------------------
# Internal: build scoring_dict
# Keyed by EXACT DB instrument_name values.
# DatasetBuilder._apply_scoring() matches these against
# df["instrument_key"] (which holds the DB name post-rename).
# -----------------------------------------------------------------------

def _build_scoring_dict() -> Dict[str, Any]:
    """
    Build scoring_dict for DatasetBuilder.

    Contains only the fields DatasetBuilder._apply_scoring() needs:
        Binary:  {"scoring_type", "correct_answers"}
        Likert:  {"scoring_type", "default_scale", "reverse_scale",
                  "reverse_questions"}
    """
    scoring = {}

    # ---- Surveys: SCCCES ----
    sccces_s = _load_yaml("b4ai_sccces_scoring.yaml")
    sccces_scoring = {
        "scoring_type":     sccces_s["scoring_type"],
        "default_scale":    sccces_s["default_scale"],
        "reverse_scale":    sccces_s["reverse_scale"],
        "reverse_questions": sccces_s.get("reverse_questions", []),
    }
    for n in _all_module_numbers():
        scoring[f"module{n}_b4ai_sccces_survey"] = sccces_scoring

    # ---- Surveys: SIMS ----
    sims_s = _load_yaml("b4ai_sims_scoring.yaml")
    sims_scoring = {
        "scoring_type":     sims_s["scoring_type"],
        "default_scale":    sims_s["default_scale"],
        "reverse_scale":    sims_s["reverse_scale"],
        "reverse_questions": sims_s.get("reverse_questions", []),
    }
    for n in _all_module_numbers():
        scoring[f"module{n}_b4ai_sims_survey"] = sims_scoring

    # ---- MCQ content assessments ----
    for n in _all_module_numbers():
        try:
            mcq_s = _load_yaml(f"module{n}_content_mcq_assessment_scoring.yaml")
            scoring[f"module{n}_content_mcq_assessment"] = {
                "scoring_type":   mcq_s["scoring_type"],
                "correct_answers": mcq_s["correct_answers"],
            }
        except FileNotFoundError:
            pass   # no scoring for this module yet — item_score stays NaN

    # ---- Pre/post assessments ----
    for yaml_name, db_name in [
        ("pre_ai_misconceptions_assessment_scoring.yaml",
         "precourse_pre_ai_misconceptions_assessment"),
        ("post_ai_misconceptions_assessment_scoring.yaml",
         "postcourse_post_ai_misconceptions_assessment"),
        ("pre_aici_assessment_scoring.yaml",
         "precourse_pre_aici_assessment"),
        ("post_aici_assessment_scoring.yaml",
         "postcourse_post_aici_assessment"),
    ]:
        try:
            s = _load_yaml(yaml_name)
            scoring[db_name] = {
                "scoring_type":   s["scoring_type"],
                "correct_answers": s["correct_answers"],
            }
        except FileNotFoundError:
            pass

    # Reflections and demographics intentionally excluded —
    # no scoring defined; item_score stays NaN for those rows.

    return scoring


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_canonical_data(
    db_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Load and build the full canonical research dataset.

    Parameters
    ----------
    db_path : Path or None
        Path to responses.db. Defaults to project_root/responses.db.

    Returns
    -------
    canonical_df : pd.DataFrame
        Scored, metadata-enriched canonical dataset.
        Schema: user_id, module_id, instrument_key, question_id,
                response_value, item_score, construct, grade,
                submitted_at, completed_at, cohort_id

    demographics_df : pd.DataFrame
        Per-user demographics for dashboard filtering.
        Schema: user_id, grade, grade_level, gender,
                first_language_english

    cohort_map : dict
        {user_id: cohort_id | None} for all registered users.

    Raises
    ------
    FileNotFoundError
        If responses.db or required YAML files are missing.
    ValueError
        If DatasetBuilder detects schema or scoring issues.
    """
    # Changed by Claude as part of migration: the below was working locally
    # db_path = Path(db_path) if db_path else _DB_PATH
    # Changed to:
    if not db_path:
        from core.db_utils import DB_PATH as _DB_PATH
    db_path = Path(db_path) if db_path else _DB_PATH
    
    # End of the above change

    if not db_path.exists():
        raise FileNotFoundError(f"responses.db not found at: {db_path}")

    # ---- 1. Load raw responses ----
    conn = sqlite3.connect(db_path)
    responses_df = pd.read_sql_query(
        """
        SELECT user_id, instrument_name, question_id,
               response_value, submitted_at
        FROM   responses
        """,
        conn,
    )
    conn.close()

    #if responses_df.empty: #results in teacher dashboard being inaccessible due to a crash when no data is in the responses.db
    #    raise ValueError("responses table is empty.")
    # replaced by 
    if responses_df.empty:
        empty_canonical = pd.DataFrame(columns=["user_id","module_id","instrument_key","question_id","response_value","item_score","construct","grade","submitted_at","completed_at","cohort_id"])
        empty_demo = pd.DataFrame(columns=["user_id","grade","grade_level","gender","first_language_english"])
        return empty_canonical, empty_demo, {}

    # ---- 2. Load demographics ----
    demographics_df = extract_demographics(db_path)

    # ---- 3. Load cohort map ----
    cohort_map = get_user_cohort_map()

    # ---- 4. Build instruments_dict and scoring_dict ----
    instruments_dict = _build_instruments_dict()
    scoring_dict     = _build_scoring_dict()

    # ---- 5. Build canonical_df via DatasetBuilder ----
    # Pass demographics_df[["user_id","grade"]] so grade merges in.
    # DatasetBuilder also attaches cohort_id via get_user_cohort_map().
    builder = DatasetBuilder(
        responses_df    = responses_df,
        instruments_dict= instruments_dict,
        scoring_dict    = scoring_dict,
        demographics_df = demographics_df[["user_id", "grade"]],
    )
    canonical_df = builder.build()

    return canonical_df, demographics_df, cohort_map
