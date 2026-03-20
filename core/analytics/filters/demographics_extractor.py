# core/analytics/filters/demographics_extractor.py
"""
Demographics Extractor
======================
Extracts and normalizes participant demographics from the responses
table into a clean, typed DataFrame ready for DatasetBuilder.

Architecture rules:
- Analytics pipeline use only
- No Streamlit, no side effects, no global state
- Read-only DB access via provided path
- Returns a clean wide-format DataFrame (one row per user_id)
- All normalization is explicit, documented, and tested

Source:
    responses table, instrument_name = 'precourse_demographics_survey'

Question → Column mapping (confirmed against YAML + DB):
    Q2_2  →  grade                (raw string, e.g. "Fourth (4th) grade")
    Q2_2  →  grade_level          (int 4-8 or str "Adult")
    Q2_3  →  first_language_english  (bool, normalized from "True"/"False")
    Q2_4  →  gender               (str "Male"/"Female")

Questions intentionally excluded from demographics_df:
    Q1_1  →  initials (PII)
    Q2_1  →  event location (not a demographic variable)
    Q2_5  →  prior AI awareness (separate research variable)
    Q2_6  →  AI exposure sources (multi-select, separate variable)

Output schema:
    user_id                 str
    grade                   str | None
    grade_level             int | str | None   (4/5/6/7/8 or "Adult")
    gender                  str | None         ("Male" / "Female")
    first_language_english  bool | None
"""

import sqlite3
from pathlib import Path
from typing import Optional, Union

import pandas as pd


# -----------------------------------------------------------------------
# Grade normalization map
# Keyed on exact DB-stored strings from YAML options
# -----------------------------------------------------------------------
_GRADE_LEVEL_MAP = {
    "Fourth (4th) grade":  4,
    "Fifth (5th) grade":   5,
    "Sixth (6th) grade":   6,
    "Seventh (7th) grade": 7,
    "Eighth (8th) grade":  8,
    "Adult":               "Adult",
}

# DB instrument_name for demographics (as stored by submission_engine)
_DEMOGRAPHICS_INSTRUMENT = "precourse_demographics_survey"

# Questions to extract
_TARGET_QUESTIONS = {"Q2_2", "Q2_3", "Q2_4"}


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _normalize_boolean(value: Optional[str]) -> Optional[bool]:
    """
    Normalize Q2_3 (first language English) to Python bool.

    The DB stores Python bool repr strings with variable casing:
        "True", "true", "False", "false"

    Returns None if value is missing or unrecognized.
    """
    if value is None or (isinstance(value, float)):
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _normalize_grade_level(
    raw_grade: Optional[str],
) -> Optional[Union[int, str]]:
    """
    Convert raw grade string to numeric level or 'Adult'.

    Returns None if grade is missing or unrecognized.
    """
    if raw_grade is None or (isinstance(raw_grade, float)):
        return None
    return _GRADE_LEVEL_MAP.get(str(raw_grade).strip(), None)


def _normalize_gender(value: Optional[str]) -> Optional[str]:
    """
    Normalize Q2_4 gender value.

    YAML options are "Male" / "Female" — stored as-is in DB.
    Returns None if missing or unrecognized.
    """
    if value is None or (isinstance(value, float)):
        return None
    v = str(value).strip()
    if v in ("Male", "Female"):
        return v
    return None


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def extract_demographics(db_path: Union[str, Path]) -> pd.DataFrame:
    """
    Extract and normalize demographics from the responses table.

    Parameters
    ----------
    db_path : str or Path
        Absolute or relative path to responses.db.

    Returns
    -------
    pd.DataFrame
        One row per user_id with columns:
            user_id, grade, grade_level, gender, first_language_english

        Rows are sorted by user_id for determinism.
        Users who completed the demographics survey but left individual
        questions blank will have None in those columns.
        Users who never completed the demographics survey are not included.

    Raises
    ------
    FileNotFoundError
        If db_path does not exist.
    ValueError
        If the responses table is missing or has unexpected schema.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # -----------------------------------------------------------------------
    # 1. Load only the rows we need from DB
    # -----------------------------------------------------------------------
    conn = sqlite3.connect(db_path)
    try:
        raw = pd.read_sql_query(
            """
            SELECT user_id, question_id, response_value
            FROM   responses
            WHERE  instrument_name = ?
              AND  question_id     IN ('Q2_2', 'Q2_3', 'Q2_4')
            """,
            conn,
            params=(_DEMOGRAPHICS_INSTRUMENT,),
        )
    except Exception as e:
        raise ValueError(f"Failed to query responses table: {e}") from e
    finally:
        conn.close()

    if raw.empty:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            "user_id", "grade", "grade_level",
            "gender", "first_language_english",
        ])

    # -----------------------------------------------------------------------
    # 2. Pivot long → wide  (user_id × question_id)
    # -----------------------------------------------------------------------
    # Keep first response per user per question (single-attempt rule)
    raw = raw.drop_duplicates(subset=["user_id", "question_id"], keep="first")

    wide = raw.pivot(
        index="user_id",
        columns="question_id",
        values="response_value",
    ).reset_index()

    # Ensure all three question columns exist even if some users skipped them
    for qid in ("Q2_2", "Q2_3", "Q2_4"):
        if qid not in wide.columns:
            wide[qid] = None

    wide.columns.name = None  # remove MultiIndex artifact

    # -----------------------------------------------------------------------
    # 3. Build normalized output columns
    # -----------------------------------------------------------------------
    result = pd.DataFrame()
    result["user_id"] = wide["user_id"]

    # grade — raw string preserved for display / filtering
    result["grade"] = wide["Q2_2"].where(wide["Q2_2"].notna(), None)

    # grade_level — numeric (4–8) or "Adult"
    result["grade_level"] = wide["Q2_2"].apply(_normalize_grade_level)

    # gender — "Male" / "Female" / None
    result["gender"] = wide["Q2_4"].apply(_normalize_gender)

    # first_language_english — True / False / None
    result["first_language_english"] = wide["Q2_3"].apply(_normalize_boolean)

    # -----------------------------------------------------------------------
    # 4. Sort for determinism
    # -----------------------------------------------------------------------
    result = result.sort_values("user_id").reset_index(drop=True)

    return result


def extract_demographics_from_df(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    In-memory variant: extract demographics from an already-loaded
    responses DataFrame.

    Accepts a DataFrame with columns:
        user_id, instrument_name, question_id, response_value

    Useful for testing or when the caller already holds the full
    responses DataFrame in memory.

    Returns
    -------
    pd.DataFrame
        Same schema as extract_demographics().
    """
    required = {"user_id", "instrument_name", "question_id", "response_value"}
    missing = required - set(responses_df.columns)
    if missing:
        raise ValueError(f"responses_df missing columns: {missing}")

    subset = responses_df[
        (responses_df["instrument_name"] == _DEMOGRAPHICS_INSTRUMENT) &
        (responses_df["question_id"].isin(_TARGET_QUESTIONS))
    ][["user_id", "question_id", "response_value"]].copy()

    if subset.empty:
        return pd.DataFrame(columns=[
            "user_id", "grade", "grade_level",
            "gender", "first_language_english",
        ])

    # Reuse same normalization pipeline
    subset = subset.drop_duplicates(subset=["user_id", "question_id"], keep="first")

    wide = subset.pivot(
        index="user_id",
        columns="question_id",
        values="response_value",
    ).reset_index()

    for qid in ("Q2_2", "Q2_3", "Q2_4"):
        if qid not in wide.columns:
            wide[qid] = None

    wide.columns.name = None

    result = pd.DataFrame()
    result["user_id"] = wide["user_id"]
    result["grade"] = wide["Q2_2"].where(wide["Q2_2"].notna(), None)
    result["grade_level"] = wide["Q2_2"].apply(_normalize_grade_level)
    result["gender"] = wide["Q2_4"].apply(_normalize_gender)
    result["first_language_english"] = wide["Q2_3"].apply(_normalize_boolean)

    result = result.sort_values("user_id").reset_index(drop=True)
    return result
