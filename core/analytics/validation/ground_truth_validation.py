# core/analytics/validation/ground_truth_validation.py
"""
Ground Truth Validation Engine
================================
Infrastructure for comparing LLM-assisted qualitative analysis (ITA/DTA)
against human-coded reference data. Built ahead of having a completed
ITA/DTA run and matching human-coded data to validate against -- see
docs/theme-comparison-methodology-guide.md and PROJECT_STATUS.md for the
full context. Every function here is synthetic-data-tested; none has yet
been run against real ITA/DTA output, which is why the Teacher Dashboard
UI built on top of this module is explicitly labeled "To Be Developed."

Architecture rules (matching inferential_tests.py / correlation_engine.py):
- Pure functions -- no Streamlit, no DB access
- All inputs are plain Python/pandas objects; all outputs are plain dicts
- Optional heavy dependency (krippendorff) imported lazily, wrapped in
  try/except, so this module stays importable even if it's ever missing

Public API:
-----------
GROUND_TRUTH_REQUIRED_COLUMNS, GROUND_TRUTH_OPTIONAL_COLUMNS
    Column spec for the human-coded ground-truth upload template.

validate_ground_truth_df(df)
    Validate/normalize an uploaded ground-truth CSV/XLSX (already read
    into a DataFrame by the caller).

compute_krippendorffs_alpha(rater_a, rater_b, level_of_measurement)
    Inter-rater reliability between two coders (e.g. human vs. DTA) on
    the same set of coding units.

calibrate_threshold_youden(scores, true_labels)
    ROC/Youden's J (Youden, 1950) optimal decision threshold for a
    continuous score against binary human match/no-match judgments --
    e.g. calibrating theme_comparator.py's MATCH_FLOOR.

References
----------
Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a
    standard reliability measure for coding data. Communication Methods
    and Measures, 1(1), 77-89.
Youden, W. J. (1950). Index for rating diagnostic tests. Cancer, 3(1),
    32-35.
"""

from __future__ import annotations
from typing import Any, Dict, Hashable, List, Optional
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Ground-truth upload template
# -----------------------------------------------------------------------

GROUND_TRUTH_REQUIRED_COLUMNS: List[str] = ["category", "code_name", "quote"]

# participant_id is deliberately optional and never required -- a
# COPPA-conscious design choice (the human-coded codebook doesn't need to
# be linkable back to a specific platform account), not a gap to close.
GROUND_TRUTH_OPTIONAL_COLUMNS: List[str] = [
    "participant_id", "page_ref", "researcher_initials", "notes",
]

GROUND_TRUTH_ALL_COLUMNS: List[str] = (
    GROUND_TRUTH_REQUIRED_COLUMNS + GROUND_TRUTH_OPTIONAL_COLUMNS
)

LOW_N_UNITS_THRESHOLD = 20   # advisory warning threshold for alpha/ROC sample size


def validate_ground_truth_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate and normalize an uploaded ground-truth DataFrame against the
    documented template (GROUND_TRUTH_REQUIRED_COLUMNS / _OPTIONAL_COLUMNS).

    Does not require exact column order or the presence of optional
    columns -- missing optional columns are added as empty. Column name
    matching is case-insensitive and tolerant of surrounding whitespace.

    Parameters
    ----------
    df : pd.DataFrame
        Already read from the uploaded CSV/XLSX by the caller (this
        function has no file-I/O or Streamlit dependency).

    Returns
    -------
    dict:
        valid       bool
        errors      list[str]   -- non-empty only when valid is False
        warnings    list[str]   -- advisory, doesn't block validity
        n_rows      int
        df          pd.DataFrame | None  -- normalized (columns renamed/
            reordered to the template, missing optional columns added as
            "") if valid, else None
    """
    result: Dict[str, Any] = {
        "valid": False, "errors": [], "warnings": [], "n_rows": 0, "df": None,
    }

    if df is None or df.empty:
        result["errors"].append("The uploaded file has no rows.")
        return result

    # Case/whitespace-insensitive column matching
    col_map = {c.strip().lower(): c for c in df.columns}
    missing_required = [
        c for c in GROUND_TRUTH_REQUIRED_COLUMNS if c not in col_map
    ]
    if missing_required:
        result["errors"].append(
            f"Missing required column(s): {', '.join(missing_required)}. "
            f"Required columns are: {', '.join(GROUND_TRUTH_REQUIRED_COLUMNS)}."
        )
        return result

    normalized = pd.DataFrame()
    for col in GROUND_TRUTH_ALL_COLUMNS:
        if col in col_map:
            normalized[col] = df[col_map[col]].astype(str).str.strip()
        else:
            normalized[col] = ""

    # Drop fully-blank rows (common with trailing spreadsheet rows)
    non_blank_mask = (
        (normalized["category"] != "") |
        (normalized["code_name"] != "") |
        (normalized["quote"] != "")
    )
    normalized = normalized[non_blank_mask].reset_index(drop=True)

    if normalized.empty:
        result["errors"].append(
            "No rows remain after removing blank rows -- check that "
            "category/code_name/quote are actually populated."
        )
        return result

    empty_quotes = int((normalized["quote"] == "").sum())
    if empty_quotes:
        result["warnings"].append(
            f"{empty_quotes} row(s) have an empty quote -- they'll still "
            f"be counted as coded units, but carry no traceable evidence."
        )

    if (normalized["participant_id"] == "").all():
        result["warnings"].append(
            "No participant_id values present -- fine for aggregate "
            "theme/code-set comparisons, but participant-level matching "
            "against a specific ITA/DTA run's participant_id field won't "
            "be possible from this file alone."
        )

    result["valid"] = True
    result["n_rows"] = len(normalized)
    result["df"] = normalized
    return result


# -----------------------------------------------------------------------
# Krippendorff's alpha (inter-rater reliability)
# -----------------------------------------------------------------------

def compute_krippendorffs_alpha(
    rater_a: Dict[Hashable, Any],
    rater_b: Dict[Hashable, Any],
    level_of_measurement: str = "nominal",
) -> Dict[str, Any]:
    """
    Krippendorff's alpha between two coders on the same set of units.

    Designed for the closed-codebook case where this statistic actually
    applies cleanly (e.g. DTA construct coding: same units -- participant
    x construct pairs -- same fixed category/value set for both coders).
    Do not use this directly on free-form/open-ended theme sets (e.g. ITA
    output) where the two "coders" may not share a category scheme at
    all -- see docs/theme-comparison-methodology-guide.md.

    Parameters
    ----------
    rater_a, rater_b : dict
        Maps a unit identifier (e.g. (participant_id, construct_name)) to
        the value that rater assigned to that unit (e.g. 0/1 for
        evidence_present, or a valence category string). A unit present
        in only one rater's dict is treated as missing (NaN) for the
        other -- not dropped from the report, but Krippendorff's formula
        itself only uses units where both raters provided a value.
    level_of_measurement : str
        "nominal" (categories, e.g. valence labels), "ordinal", "interval",
        or "ratio" (e.g. a 0/1 presence indicator can be treated as
        nominal or interval -- nominal is the safer default absent a
        specific reason to assume order/interval spacing).

    Returns
    -------
    dict:
        alpha                float | None
        n_units_total        int   -- union of both raters' unit sets
        n_units_both_coded   int   -- units both raters actually coded
            (what the alpha calculation is actually based on)
        low_n_warning        bool  -- n_units_both_coded < LOW_N_UNITS_THRESHOLD
        level_of_measurement str
        error                str | None

    Reference
    ---------
    Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a
    standard reliability measure for coding data. Communication Methods
    and Measures, 1(1), 77-89.
    """
    result: Dict[str, Any] = {
        "alpha": None,
        "level_of_measurement": level_of_measurement,
        "error": None,
    }

    try:
        import krippendorff
    except ImportError:
        result["error"] = (
            "The 'krippendorff' package is not installed. "
            "Run: pip install krippendorff"
        )
        return result

    all_units = sorted(
        set(rater_a.keys()) | set(rater_b.keys()), key=lambda u: str(u)
    )
    both_coded = [u for u in all_units if u in rater_a and u in rater_b]

    result["n_units_total"]      = len(all_units)
    result["n_units_both_coded"] = len(both_coded)
    result["low_n_warning"]      = len(both_coded) < LOW_N_UNITS_THRESHOLD

    if len(both_coded) < 2:
        result["error"] = (
            f"Need at least 2 units coded by both raters to compute "
            f"agreement. Found {len(both_coded)}."
        )
        return result

    row_a = [rater_a.get(u, np.nan) for u in all_units]
    row_b = [rater_b.get(u, np.nan) for u in all_units]

    # krippendorff.alpha requires numeric or consistently-typed values;
    # non-numeric categorical labels (e.g. "positive"/"negative") are
    # passed through as strings, which the package handles natively for
    # nominal/ordinal measurement levels.
    reliability_data = [row_a, row_b]

    try:
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level_of_measurement,
        )
        result["alpha"] = round(float(alpha), 4)
    except Exception as e:
        result["error"] = f"krippendorff.alpha() failed: {e}"

    return result


# -----------------------------------------------------------------------
# ROC / Youden's J threshold calibration
# -----------------------------------------------------------------------

def calibrate_threshold_youden(
    scores: List[float],
    true_labels: List[int],
) -> Dict[str, Any]:
    """
    Find the decision threshold that maximizes Youden's J statistic
    (sensitivity + specificity - 1) for a continuous score against
    binary human judgments -- e.g. calibrating theme_comparator.py's
    MATCH_FLOOR against real human match/no-match labels on candidate
    theme pairs, once such labels exist.

    Parameters
    ----------
    scores : list of float
        Continuous prediction scores (e.g. agreement scores for
        candidate theme pairs).
    true_labels : list of int
        Binary ground truth aligned with scores: 1 = a human judged this
        pair a true match, 0 = not a match.

    Returns
    -------
    dict:
        n_pairs, n_true_matches, n_true_nonmatches,
        optimal_threshold, sensitivity, specificity, youden_j, auc,
        low_n_warning, error

    Reference
    ---------
    Youden, W. J. (1950). Index for rating diagnostic tests. Cancer,
    3(1), 32-35.
    """
    result: Dict[str, Any] = {"error": None}

    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(true_labels)

    if len(scores_arr) != len(labels_arr):
        result["error"] = (
            f"scores and true_labels must be the same length. "
            f"Got {len(scores_arr)} and {len(labels_arr)}."
        )
        return result

    if len(scores_arr) < 2:
        result["error"] = f"Need at least 2 labeled pairs. Got {len(scores_arr)}."
        return result

    unique_labels = set(labels_arr.tolist())
    if len(unique_labels) < 2:
        result["error"] = (
            "true_labels must contain both classes (at least one true "
            "match and one true non-match) -- ROC/Youden's J is "
            "undefined when every pair was judged the same way."
        )
        return result

    try:
        from sklearn.metrics import roc_curve, auc as _auc
    except ImportError:
        result["error"] = "scikit-learn is not installed."
        return result

    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    j_stat    = tpr - fpr
    best_idx  = int(np.argmax(j_stat))

    result.update({
        "n_pairs":            len(scores_arr),
        "n_true_matches":     int(labels_arr.sum()),
        "n_true_nonmatches":  int(len(labels_arr) - labels_arr.sum()),
        "optimal_threshold":  round(float(thresholds[best_idx]), 4),
        "sensitivity":        round(float(tpr[best_idx]), 4),
        "specificity":        round(float(1 - fpr[best_idx]), 4),
        "youden_j":           round(float(j_stat[best_idx]), 4),
        "auc":                round(float(_auc(fpr, tpr)), 4),
        "low_n_warning":      len(scores_arr) < LOW_N_UNITS_THRESHOLD,
    })
    return result
