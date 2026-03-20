# core/analytics/descriptive/descriptive_stats.py
"""
Descriptive Statistics — Participant Counts
============================================
Computes participant breakdown counts from demographics_df.

Architecture rules:
- Pure pandas — no DB access, no Streamlit, no side effects
- All inputs are DataFrames or plain dicts; all outputs are DataFrames or int
- No scoring logic — operates only on demographics_df
- None / NaN values are counted explicitly as "Unknown" not silently dropped

Expected demographics_df schema (from demographics_extractor.py):
    user_id                 str
    grade                   str | None     raw string e.g. "Fourth (4th) grade"
    grade_level             int | str | None  4/5/6/7/8 or "Adult"
    gender                  str | None     "Male" / "Female"
    first_language_english  bool | None

Public API:
-----------
count_participants(demographics_df)
    Total number of unique participants.

count_by_gender(demographics_df)
    Breakdown by gender (Male / Female / Unknown).

count_by_grade(demographics_df)
    Breakdown by grade level (4–8, Adult, Unknown).

count_by_language(demographics_df)
    Breakdown by first language (English / Non-English / Unknown).

count_by_cohort(demographics_df, cohort_map)
    Breakdown by cohort. cohort_map: {user_id: cohort_id | None}

participant_summary(demographics_df, cohort_map=None)
    All breakdowns in one dict.
"""

from typing import Dict, Optional, Union
import pandas as pd


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _validate_demographics_df(df: pd.DataFrame) -> None:
    required = {"user_id", "grade_level", "gender", "first_language_english"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"demographics_df missing required columns: {missing}. "
            f"Pass output of demographics_extractor.extract_demographics()."
        )


def _pct(n: int, total: int) -> float:
    """Round percentage to 2 decimal places. Returns 0.0 if total is 0."""
    if total == 0:
        return 0.0
    return round(n / total * 100, 2)


# -----------------------------------------------------------------------
# Public function 1: count_participants
# -----------------------------------------------------------------------

def count_participants(demographics_df: pd.DataFrame) -> int:
    """
    Return total number of unique participants in demographics_df.

    Parameters
    ----------
    demographics_df : pd.DataFrame
        Output of demographics_extractor.extract_demographics().

    Returns
    -------
    int
        Number of unique user_ids.
    """
    if demographics_df is None or demographics_df.empty:
        return 0
    return int(demographics_df["user_id"].nunique())


# -----------------------------------------------------------------------
# Public function 2: count_by_gender
# -----------------------------------------------------------------------

def count_by_gender(demographics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count participants by gender.

    Parameters
    ----------
    demographics_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: gender, n, pct
        Rows: one per gender value + one "Unknown" row if any nulls.
        Sorted: Male, Female, Unknown.
        pct = n / total_participants × 100.
    """
    _validate_demographics_df(demographics_df)

    total = count_participants(demographics_df)
    if total == 0:
        return pd.DataFrame(columns=["gender", "n", "pct"])

    df = demographics_df.copy()

    # Fill None/NaN as explicit "Unknown" category
    df["_gender"] = df["gender"].fillna("Unknown")

    counts = (
        df.groupby("_gender", sort=False)["user_id"]
        .nunique()
        .reset_index()
        .rename(columns={"_gender": "gender", "user_id": "n"})
    )
    counts["pct"] = counts["n"].apply(lambda n: _pct(n, total))

    # Canonical sort order: Male, Female, Unknown
    order = ["Male", "Female", "Unknown"]
    counts["_order"] = counts["gender"].map(
        {v: i for i, v in enumerate(order)}
    ).fillna(len(order))
    counts = (
        counts.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return counts


# -----------------------------------------------------------------------
# Public function 3: count_by_grade
# -----------------------------------------------------------------------

def count_by_grade(demographics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count participants by grade level.

    Parameters
    ----------
    demographics_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: grade_level, grade, n, pct
            grade_level  int (4-8), str ("Adult"), or str ("Unknown")
            grade        raw YAML string or "Unknown"
        Rows sorted numerically (4, 5, 6, 7, 8, Adult, Unknown).
        pct = n / total_participants × 100.
    """
    _validate_demographics_df(demographics_df)

    total = count_participants(demographics_df)
    if total == 0:
        return pd.DataFrame(columns=["grade_level", "grade", "n", "pct"])

    df = demographics_df.copy()

    # Fill missing grade values explicitly
    df["_grade_level"] = df["grade_level"].apply(
        lambda x: "Unknown" if (x is None or (isinstance(x, float) and pd.isna(x))) else x
    )
    df["_grade"] = df["grade"].fillna("Unknown")

    counts = (
        df.groupby(["_grade_level", "_grade"], sort=False)["user_id"]
        .nunique()
        .reset_index()
        .rename(columns={
            "_grade_level": "grade_level",
            "_grade":       "grade",
            "user_id":      "n",
        })
    )
    counts["pct"] = counts["n"].apply(lambda n: _pct(n, total))

    # Sort: numeric grades first, then Adult, then Unknown
    def _sort_key(gl):
        if isinstance(gl, int):
            return (0, gl)
        if gl == "Adult":
            return (1, 0)
        return (2, 0)  # Unknown

    counts["_sort"] = counts["grade_level"].apply(_sort_key)
    counts = (
        counts.sort_values("_sort")
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    return counts


# -----------------------------------------------------------------------
# Public function 4: count_by_language
# -----------------------------------------------------------------------

def count_by_language(demographics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count participants by first language (English vs Non-English).

    Parameters
    ----------
    demographics_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: first_language_english, label, n, pct
            first_language_english  True / False / None
            label                   "English" / "Non-English" / "Unknown"
        Rows sorted: English, Non-English, Unknown.
        pct = n / total_participants × 100.
    """
    _validate_demographics_df(demographics_df)

    total = count_participants(demographics_df)
    if total == 0:
        return pd.DataFrame(
            columns=["first_language_english", "label", "n", "pct"]
        )

    df = demographics_df.copy()

    def _lang_label(val):
        if val is True:
            return "English"
        if val is False:
            return "Non-English"
        # None or NaN
        return "Unknown"

    df["_label"] = df["first_language_english"].apply(_lang_label)

    counts = (
        df.groupby("_label", sort=False)["user_id"]
        .nunique()
        .reset_index()
        .rename(columns={"_label": "label", "user_id": "n"})
    )

    # Reconstruct bool column from label
    label_to_bool = {
        "English":     True,
        "Non-English": False,
        "Unknown":     None,
    }
    counts["first_language_english"] = counts["label"].map(label_to_bool)
    counts["pct"] = counts["n"].apply(lambda n: _pct(n, total))

    # Sort: English, Non-English, Unknown
    order = ["English", "Non-English", "Unknown"]
    counts["_order"] = counts["label"].map(
        {v: i for i, v in enumerate(order)}
    ).fillna(len(order))
    counts = (
        counts.sort_values("_order")
        .drop(columns="_order")
        [["first_language_english", "label", "n", "pct"]]
        .reset_index(drop=True)
    )

    return counts


# -----------------------------------------------------------------------
# Public function 5: count_by_cohort
# -----------------------------------------------------------------------

def count_by_cohort(
    demographics_df: pd.DataFrame,
    cohort_map: Dict[str, Optional[str]],
) -> pd.DataFrame:
    """
    Count participants by cohort.

    Parameters
    ----------
    demographics_df : pd.DataFrame
        Output of demographics_extractor. Only user_ids present here
        are counted — users without demographics are excluded.

    cohort_map : dict
        Mapping of {user_id: cohort_id | None}.
        From auth.user_manager.get_user_cohort_map().

    Returns
    -------
    pd.DataFrame
        Columns: cohort_id, n, pct
            cohort_id  str or "Unassigned" (if cohort_id is None)
        Sorted alphabetically by cohort_id, Unassigned last.
        pct = n / total_participants × 100.
    """
    _validate_demographics_df(demographics_df)

    total = count_participants(demographics_df)
    if total == 0:
        return pd.DataFrame(columns=["cohort_id", "n", "pct"])

    df = demographics_df.copy()

    # Attach cohort_id from map — only for users in demographics_df
    df["cohort_id"] = df["user_id"].map(cohort_map).fillna("Unassigned")

    counts = (
        df.groupby("cohort_id", sort=False)["user_id"]
        .nunique()
        .reset_index()
        .rename(columns={"user_id": "n"})
    )
    counts["pct"] = counts["n"].apply(lambda n: _pct(n, total))

    # Sort: alphabetical, Unassigned last
    non_unassigned = counts[counts.cohort_id != "Unassigned"].sort_values("cohort_id")
    unassigned     = counts[counts.cohort_id == "Unassigned"]
    counts = pd.concat([non_unassigned, unassigned]).reset_index(drop=True)

    return counts


# -----------------------------------------------------------------------
# Public function 6: participant_summary
# -----------------------------------------------------------------------

def participant_summary(
    demographics_df: pd.DataFrame,
    cohort_map: Optional[Dict[str, Optional[str]]] = None,
) -> dict:
    """
    Compute all participant breakdowns in one call.

    Parameters
    ----------
    demographics_df : pd.DataFrame
    cohort_map : dict or None
        If None, cohort breakdown is omitted from the summary.

    Returns
    -------
    dict with keys:
        "total"     int
        "by_gender" pd.DataFrame
        "by_grade"  pd.DataFrame
        "by_language" pd.DataFrame
        "by_cohort" pd.DataFrame | None  (None if cohort_map not provided)
    """
    _validate_demographics_df(demographics_df)

    summary = {
        "total":       count_participants(demographics_df),
        "by_gender":   count_by_gender(demographics_df),
        "by_grade":    count_by_grade(demographics_df),
        "by_language": count_by_language(demographics_df),
        "by_cohort":   (
            count_by_cohort(demographics_df, cohort_map)
            if cohort_map is not None
            else None
        ),
    }
    return summary
