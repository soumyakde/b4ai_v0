# core/analytics/descriptive/score_aggregator.py
"""
Score Aggregator
================
Computes per-user and group-level score summaries from the canonical
dataset produced by DatasetBuilder.

Architecture rules:
- Pure pandas — no DB access, no Streamlit, no side effects
- All inputs are DataFrames; all outputs are DataFrames
- No scoring logic — item_score is already in canonical_df
- No instrument classification hardcoded — uses instrument_key_resolver

Public API:
-----------
compute_assessment_scores(canonical_df, instrument_keys=None)
    Per-user % correct for binary assessments.

compute_construct_means(canonical_df, instrument_keys=None)
    Per-user per-construct mean for Likert surveys.

summarize_scores(scores_df, group_by_col=None, demographics_df=None)
    Group-level mean / median / mode across users.
    Works on output of either function above.

Expected canonical_df schema:
    user_id, module_id, instrument_key, question_id, response_value,
    item_score, construct, grade, submitted_at, completed_at, cohort_id

Expected demographics_df schema (optional, for gender/language groupby):
    user_id, grade, grade_level, gender, first_language_english
"""

from typing import Optional, List
import statistics

import pandas as pd


# -----------------------------------------------------------------------
# Instrument classification helpers
# -----------------------------------------------------------------------

# Survey base keys whose items carry construct labels
# Must stay in sync with instrument_key_resolver._SURVEY_BASE_KEYS
_SURVEY_BASE_KEYS = frozenset([
    "b4ai_sccces_survey",
    "b4ai_sims_survey",
])

# Columns that come from demographics_df (not in canonical_df)
_DEMOGRAPHICS_COLS = frozenset([
    "gender",
    "first_language_english",
])

# Columns available directly in canonical_df for groupby
_CANONICAL_GROUP_COLS = frozenset([
    "grade",
    "cohort_id",
])


def _is_survey(instrument_key: str) -> bool:
    """
    Return True if instrument_key is a known Likert survey.

    Handles both canonical keys ("b4ai_sccces_survey") and
    DB-style keys ("module1_b4ai_sccces_survey") produced by
    submission_engine's module prefix convention.
    """
    name = instrument_key.strip()
    return any(
        name == base or name.endswith("_" + base)
        for base in _SURVEY_BASE_KEYS
    )


def _is_assessment(instrument_key: str) -> bool:
    """Return True if instrument_key is a binary assessment."""
    return not _is_survey(instrument_key)


# -----------------------------------------------------------------------
# Internal validation
# -----------------------------------------------------------------------

def _validate_canonical_df(df: pd.DataFrame) -> None:
    required = {
        "user_id", "instrument_key", "question_id",
        "item_score", "construct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"canonical_df missing required columns: {missing}"
        )
    if df.empty:
        raise ValueError("canonical_df is empty.")


def _safe_mode(series: pd.Series):
    """
    Return the first mode of a numeric series.
    Returns None if the series is empty or all-NaN.
    If multiple modes exist (tie), returns the smallest.
    """
    clean = series.dropna()
    if clean.empty:
        return None
    modes = clean.mode()
    if modes.empty:
        return None
    return round(float(modes.iloc[0]), 4)


def _safe_mean(series: pd.Series):
    clean = series.dropna()
    if clean.empty:
        return None
    return round(float(clean.mean()), 4)


def _safe_median(series: pd.Series):
    clean = series.dropna()
    if clean.empty:
        return None
    return round(float(clean.median()), 4)


# -----------------------------------------------------------------------
# Public function 1: compute_assessment_scores
# -----------------------------------------------------------------------

def compute_assessment_scores(
    canonical_df: pd.DataFrame,
    instrument_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-user % correct for binary assessment instruments.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Output of DatasetBuilder.build(). Surveys are silently skipped.
    instrument_keys : list of str, optional
        Subset to these instrument_keys only. If None, all non-survey
        instruments in canonical_df are processed.

    Returns
    -------
    pd.DataFrame
        One row per (user_id, instrument_key). Columns:
            user_id          str
            instrument_key   str
            n_items_answered int    number of items the student answered
            raw_score        float  sum of item_scores (correct count)
            pct_correct      float  raw_score / n_items_answered × 100

        Sorted by instrument_key, then user_id.

    Notes
    -----
    - Denominator is n_items_answered (not the full bank size).
      This is the correct formula for randomised MCQ draws where
      each student answers a different 20-item subset of 57.
    - Rows with NaN item_score are excluded from both numerator
      and denominator (unanswered items do not count against the student).
    - Instruments classified as surveys are silently excluded.
    """
    _validate_canonical_df(canonical_df)

    df = canonical_df.copy()

    # Drop survey rows — assessments only
    df = df[df["instrument_key"].apply(_is_assessment)]

    if instrument_keys is not None:
        df = df[df["instrument_key"].isin(instrument_keys)]

    if df.empty:
        return pd.DataFrame(columns=[
            "user_id", "instrument_key",
            "n_items_answered", "raw_score", "pct_correct",
        ])

    # Drop unanswered items (NaN item_score)
    df = df[df["item_score"].notna()].copy()

    # Carry canonical groupby columns through — take first value per user
    # (grade and cohort_id are invariant within a user)
    carry_cols = [c for c in ("grade", "cohort_id") if c in df.columns]

    # Aggregate per user × instrument
    grouped = df.groupby(["user_id", "instrument_key"], sort=False)

    result = grouped["item_score"].agg(
        n_items_answered="count",
        raw_score="sum",
    ).reset_index()

    # Merge carried columns back
    if carry_cols:
        carry_df = (
            df[["user_id", "instrument_key"] + carry_cols]
            .drop_duplicates(subset=["user_id", "instrument_key"])
        )
        result = result.merge(carry_df, on=["user_id", "instrument_key"], how="left")

    # Percentage correct — denominator is n_items_answered
    result["pct_correct"] = (
        result["raw_score"] / result["n_items_answered"] * 100
    ).round(4)

    result = result.sort_values(
        ["instrument_key", "user_id"]
    ).reset_index(drop=True)

    return result


# -----------------------------------------------------------------------
# Public function 2: compute_construct_means
# -----------------------------------------------------------------------

def compute_construct_means(
    canonical_df: pd.DataFrame,
    instrument_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-user per-construct mean scores for Likert surveys.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Output of DatasetBuilder.build(). Non-survey instruments are
        silently skipped.
    instrument_keys : list of str, optional
        Subset to these instrument_keys only. If None, all survey
        instruments in canonical_df are processed.

    Returns
    -------
    pd.DataFrame
        One row per (user_id, instrument_key, construct). Columns:
            user_id          str
            instrument_key   str
            construct        str    canonical lowercase_snake_case name
            n_items          int    items with valid scores in construct
            total_score      float  sum of item_scores in construct
            mean_score       float  total_score / n_items

        Sorted by instrument_key, construct, user_id.

    Notes
    -----
    - Items with NaN item_score excluded from both numerator/denominator.
    - Items with NaN construct (e.g. Q1_1 initials field) are excluded.
    - Scale range reminder: Likert 1–4.
      Min mean = 1.0 (all Strongly disagree on forward items)
      Max mean = 4.0 (all Strongly agree on forward items)
      Reverse-scored items are already resolved in item_score by the
      DatasetBuilder — no further adjustment needed here.
    """
    _validate_canonical_df(canonical_df)

    df = canonical_df.copy()

    # Keep only survey rows
    df = df[df["instrument_key"].apply(_is_survey)]

    if instrument_keys is not None:
        df = df[df["instrument_key"].isin(instrument_keys)]

    # Drop rows with no construct label (e.g. Q1_1 initials)
    df = df[df["construct"].notna() & (df["construct"] != "")]

    # Drop unanswered items
    df = df[df["item_score"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "user_id", "instrument_key", "construct",
            "n_items", "total_score", "mean_score",
        ])

    # Carry canonical groupby columns through — take first value per user
    carry_cols = [c for c in ("grade", "cohort_id") if c in df.columns]

    # module_id is required to separate per-module construct means.
    # e.g. module_1 SCCCES engagement vs module_2 SCCCES engagement.
    module_id_in_df = "module_id" in df.columns

    # Aggregate per user × instrument × module_id × construct
    group_keys = ["user_id", "instrument_key"]
    if module_id_in_df:
        group_keys.append("module_id")
    group_keys.append("construct")

    grouped = df.groupby(group_keys, sort=False)

    result = grouped["item_score"].agg(
        n_items="count",
        total_score="sum",
    ).reset_index()

    # Merge carried columns back (grade, cohort_id)
    if carry_cols:
        carry_df = (
            df[["user_id", "instrument_key"] + carry_cols]
            .drop_duplicates(subset=["user_id", "instrument_key"])
        )
        result = result.merge(carry_df, on=["user_id", "instrument_key"], how="left")

    result["mean_score"] = (
        result["total_score"] / result["n_items"]
    ).round(4)

    sort_cols = ["instrument_key"]
    if module_id_in_df:
        sort_cols.append("module_id")
    sort_cols += ["construct", "user_id"]

    result = result.sort_values(sort_cols).reset_index(drop=True)

    return result


# -----------------------------------------------------------------------
# Public function 3: summarize_scores
# -----------------------------------------------------------------------

def summarize_scores(
    scores_df: pd.DataFrame,
    group_by_col: Optional[str] = None,
    demographics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute group-level mean / median / mode across users.

    Works on the output of either compute_assessment_scores() or
    compute_construct_means(). Automatically detects which by checking
    for the 'construct' column.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of compute_assessment_scores() or compute_construct_means().

    group_by_col : str or None
        Column to group by. Options:
            From canonical_df (always available):
                "grade"       — raw grade string e.g. "Fourth (4th) grade"
                "cohort_id"   — cohort identifier
            From demographics_df (requires demographics_df argument):
                "gender"                — "Male" / "Female"
                "first_language_english"— True / False
        If None, summary is computed across all users (no grouping).

    demographics_df : pd.DataFrame or None
        Required when group_by_col is "gender" or
        "first_language_english". Must contain columns:
            user_id, gender, first_language_english

    Returns
    -------
    pd.DataFrame
        For assessment scores — columns:
            [group_by_col,] instrument_key,
            n_users, mean_pct, median_pct, mode_pct

        For construct means — columns:
            [group_by_col,] instrument_key, construct,
            n_users, mean_score, median_score, mode_score

        Sorted by instrument_key [, construct] [, group_by_col].

    Raises
    ------
    ValueError
        If group_by_col requires demographics_df but it was not provided.
        If scores_df is missing required columns.
    """
    if scores_df is None or scores_df.empty:
        return pd.DataFrame()

    is_construct_mode = "construct" in scores_df.columns
    score_col = "mean_score" if is_construct_mode else "pct_correct"

    # Validate score column exists
    if score_col not in scores_df.columns:
        raise ValueError(
            f"scores_df missing expected score column '{score_col}'. "
            f"Pass output of compute_assessment_scores() or "
            f"compute_construct_means()."
        )

    df = scores_df.copy()

    # -----------------------------------------------------------------------
    # Attach groupby column if needed
    # -----------------------------------------------------------------------
    if group_by_col is not None:

        if group_by_col in _DEMOGRAPHICS_COLS:
            if demographics_df is None:
                raise ValueError(
                    f"group_by_col='{group_by_col}' requires demographics_df "
                    f"but none was provided."
                )
            if "user_id" not in demographics_df.columns:
                raise ValueError(
                    "demographics_df must contain a 'user_id' column."
                )
            if group_by_col not in demographics_df.columns:
                raise ValueError(
                    f"demographics_df missing column '{group_by_col}'."
                )
            # Merge the single groupby column from demographics
            demo_subset = demographics_df[["user_id", group_by_col]].drop_duplicates("user_id")
            df = df.merge(demo_subset, on="user_id", how="left")

        elif group_by_col not in df.columns:
            raise ValueError(
                f"group_by_col='{group_by_col}' not found in scores_df "
                f"or known demographics columns. "
                f"Available: {sorted(df.columns.tolist())}"
            )

    # -----------------------------------------------------------------------
    # Build group keys
    # -----------------------------------------------------------------------
    base_keys = ["instrument_key"]
    if is_construct_mode:
        base_keys.append("construct")
    if group_by_col is not None:
        base_keys.append(group_by_col)

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    rows = []
    for group_vals, group_df in df.groupby(base_keys, sort=True, dropna=False):

        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        score_series = group_df[score_col].dropna()

        row = dict(zip(base_keys, group_vals))
        row["n_users"]     = int(score_series.count())
        row["mean"]        = _safe_mean(score_series)
        row["median"]      = _safe_median(score_series)
        row["mode"]        = _safe_mode(score_series)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Rename mean/median/mode columns to be self-describing
    if is_construct_mode:
        result = result.rename(columns={
            "mean":   "mean_score",
            "median": "median_score",
            "mode":   "mode_score",
        })
    else:
        result = result.rename(columns={
            "mean":   "mean_pct",
            "median": "median_pct",
            "mode":   "mode_pct",
        })

    # Canonical column ordering
    id_cols = base_keys + ["n_users"]
    stat_cols = (
        ["mean_score", "median_score", "mode_score"]
        if is_construct_mode
        else ["mean_pct", "median_pct", "mode_pct"]
    )
    result = result[[c for c in id_cols + stat_cols if c in result.columns]]

    return result.reset_index(drop=True)


# -----------------------------------------------------------------------
# Public function 4: aggregate_construct_means
# -----------------------------------------------------------------------

def aggregate_construct_means(
    construct_means_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse per-module construct means into a single cross-module
    aggregate per (user_id, instrument_key, construct).

    This is always derived from the output of compute_construct_means()
    — never recomputed independently — to ensure consistency.

    Parameters
    ----------
    construct_means_df : pd.DataFrame
        Output of compute_construct_means(). Must contain columns:
            user_id, instrument_key, construct, n_items, total_score

    Returns
    -------
    pd.DataFrame
        One row per (user_id, instrument_key, construct). Columns:
            user_id          str
            instrument_key   str
            construct        str
            n_modules        int    number of modules aggregated
            n_items_total    int    total items across all modules
            total_score      float  sum of item scores across modules
            mean_score       float  total_score / n_items_total

        Sorted by instrument_key, construct, user_id.

    Notes
    -----
    - Mean is computed as total_score / n_items_total (not mean of means).
      This is the correct formula — it weights each item equally
      regardless of how many modules contributed.
    - If module_id is not in construct_means_df (no per-module breakdown
      was available), this function still works correctly.
    """
    required = {"user_id", "instrument_key", "construct", "n_items", "total_score"}
    missing  = required - set(construct_means_df.columns)
    if missing:
        raise ValueError(
            f"construct_means_df missing required columns: {missing}. "
            f"Pass output of compute_construct_means()."
        )

    if construct_means_df.empty:
        return pd.DataFrame(columns=[
            "user_id", "instrument_key", "construct",
            "n_modules", "n_items_total", "total_score", "mean_score",
        ])

    df = construct_means_df.copy()

    grouped = df.groupby(
        ["user_id", "instrument_key", "construct"],
        sort=False,
    )

    result = grouped.agg(
        n_modules  =("n_items",     "count"),
        n_items_total=("n_items",   "sum"),
        total_score=("total_score", "sum"),
    ).reset_index()

    result["mean_score"] = (
        result["total_score"] / result["n_items_total"]
    ).round(4)

    result = result.sort_values(
        ["instrument_key", "construct", "user_id"]
    ).reset_index(drop=True)

    return result
