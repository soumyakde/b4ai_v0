# core/analytics/descriptive/normality_checks.py
"""
Normality & Data-Quality Checks
================================
Descriptive-analysis gate before any inferential test: distributional
assumption checks (normality) plus survey-fatigue-adjacent data-quality
flags (straight-lining, missingness, outliers).

Architecture rules (matching score_aggregator.py / inferential_tests.py):
- Pure pandas/scipy — no DB access, no Streamlit, no side effects
- All inputs are DataFrames/arrays; all outputs are DataFrames/dicts
- No instrument classification hardcoded — reuses score_aggregator's
  _is_survey()/_is_assessment()

Public API:
-----------
assess_normality(values, label="")
    Shapiro-Wilk + skewness + kurtosis for one group of per-student values.

assess_group_normality(df, value_col, group_cols)
    Runs assess_normality() once per group (e.g. per instrument, or per
    instrument+construct).

recommend_test_family(verdict, paired=False, n_groups=2, repeated_measures=False)
    Advisory only -- maps a normality verdict to the matching
    inferential_tests.py test family. Does not gate/replace the existing
    "always show both parametric and non-parametric" behavior.

assess_variance_homogeneity(df, value_col, group_col, center="median")
    Levene's test (Brown-Forsythe by default) for equal variances across
    2+ groups -- the ANOVA/independent-t-test's other core assumption.

detect_straight_lining(canonical_df, min_items=3)
    Per (user_id, instrument_key) survey pair, flags identical raw
    response_value across all items -- classic careless-responding signal.

compute_missingness_by_module(canonical_df)
    Participation rate (>=1 response) per Content MCQ module, for
    trend inspection across the 7-module curriculum.

detect_outliers(df, value_col, group_cols, method="iqr")
    Per-group (same grouping as assess_group_normality) outlier flag on
    a per-student score column.

detect_excessive_missingness(canonical_df, instrument_keys, threshold=0.5)
    Per-participant answered-item rate across a specific set of
    instruments, flagged below threshold -- used to exclude participants
    from a specific inferential test, not the module-level trend above.

References
----------
Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for
normality (complete samples). Biometrika, 52(3/4), 591-611.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from core.analytics.descriptive.score_aggregator import _is_survey, _is_assessment

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# Below this n, a Shapiro-Wilk verdict is still computed but flagged as
# exploratory -- mirrors the low-N caution pattern already used in
# cpi_engine.py (LOW_N_THRESHOLD) and irt_runner.py (MIN_N_WARN).
LOW_N_THRESHOLD = 20
MIN_N_FOR_TEST = 3


# -----------------------------------------------------------------------
# Public function 1: assess_normality
# -----------------------------------------------------------------------

def assess_normality(values, label: str = "") -> dict:
    """
    Shapiro-Wilk normality test on a 1-D array of per-student values.

    Parameters
    ----------
    values : array-like -- one value per student (e.g. pct_correct or
             mean_score), NaN-containing entries are dropped first.
    label  : str -- optional identifier carried through to the output
             (e.g. instrument_key, or "instrument | construct").

    Returns
    -------
    dict: label, n, statistic, p_value, skewness, kurtosis, verdict,
          low_n_warning, error
    """
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    n = len(arr)

    result = {
        "label": label, "n": n, "statistic": None, "p_value": None,
        "skewness": None, "kurtosis": None, "verdict": None,
        "low_n_warning": False, "error": None,
    }

    if n < MIN_N_FOR_TEST:
        result["error"] = f"Too few students (n={n}) to test normality."
        result["verdict"] = "Not tested (n<3)"
        return result

    if np.allclose(arr, arr[0]):
        result["error"] = "All values identical -- normality test undefined."
        result["verdict"] = "Not tested (no variance)"
        result["n"] = n
        return result

    try:
        stat, p = stats.shapiro(arr)
    except Exception as e:
        result["error"] = str(e)
        return result

    result["statistic"] = round(float(stat), 4)
    result["p_value"] = round(float(p), 4)
    result["skewness"] = round(float(stats.skew(arr)), 4)
    result["kurtosis"] = round(float(stats.kurtosis(arr)), 4)  # excess (Fisher), normal = 0
    result["low_n_warning"] = n < LOW_N_THRESHOLD

    # Failing to reject H0 doesn't prove normality -- worded to avoid
    # that common overclaim.
    verdict = (
        "Consistent with normality (p ≥ .05)"
        if p >= 0.05 else
        "Deviates from normality (p < .05)"
    )
    if result["low_n_warning"]:
        verdict += " -- exploratory, n<20"
    result["verdict"] = verdict

    return result


# -----------------------------------------------------------------------
# Public function 2: assess_group_normality
# -----------------------------------------------------------------------

def assess_group_normality(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Run assess_normality() once per group.

    Parameters
    ----------
    df         : pd.DataFrame -- e.g. output of compute_assessment_scores()
                 or compute_construct_means()
    value_col  : str -- column to test (e.g. "pct_correct", "mean_score")
    group_cols : list of str -- columns to group by (e.g. ["instrument_key"]
                 or ["instrument_key", "construct"])

    Returns
    -------
    pd.DataFrame -- one row per group, columns match assess_normality()'s
    dict keys plus the group_cols values themselves.
    """
    if df.empty:
        return pd.DataFrame(columns=group_cols + [
            "n", "statistic", "p_value", "skewness", "kurtosis",
            "verdict", "low_n_warning", "error",
        ])

    rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        label = " | ".join(str(k) for k in keys_tuple)
        r = assess_normality(g[value_col], label=label)
        row = dict(zip(group_cols, keys_tuple))
        row.update(r)
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Public function 3: recommend_test_family (advisory only)
# -----------------------------------------------------------------------

def recommend_test_family(
    verdict: str,
    paired: bool = False,
    n_groups: int = 2,
    repeated_measures: bool = False,
) -> str:
    """
    Map a normality verdict to the matching inferential_tests.py test
    family. Advisory only -- inferential_tests.py's existing runners
    already compute parametric AND non-parametric variants side-by-side
    unconditionally; this just annotates that existing dual display
    with an explicit recommendation, it doesn't gate or replace it.
    """
    is_normal = bool(verdict) and verdict.startswith("Consistent with normality")

    if repeated_measures:
        return (
            "RM-ANOVA (run_repeated_measures)" if is_normal
            else "Friedman test (run_repeated_measures, non-parametric)"
        )
    if paired:
        return (
            "Paired t-test (run_paired_comparison)" if is_normal
            else "Wilcoxon signed-rank (run_paired_comparison, non-parametric)"
        )
    if n_groups > 2:
        return (
            "One-way ANOVA (run_between_groups)" if is_normal
            else "Kruskal-Wallis (run_between_groups, non-parametric)"
        )
    return (
        "Independent t-test (run_between_groups)" if is_normal
        else "Mann-Whitney U (run_between_groups, non-parametric)"
    )


# -----------------------------------------------------------------------
# Public function 3b: assess_variance_homogeneity
# -----------------------------------------------------------------------

def assess_variance_homogeneity(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    center: str = "median",
) -> dict:
    """
    Levene's test for homogeneity of variance across 2+ groups -- the
    other core assumption (beyond normality) for ANOVA / independent
    t-test comparisons.

    center="median" (the Brown-Forsythe variant) is the default -- more
    robust to non-normality within the groups themselves than
    center="mean", which matters here since this is often run alongside
    a normality check that may already show a violation.

    Parameters
    ----------
    df        : pd.DataFrame -- one row per student, must contain
                value_col and group_col.
    value_col : str -- column to test (e.g. "score", "pct_correct").
    group_col : str -- grouping column (e.g. "grade", "cohort_id").
    center    : str -- passed to scipy.stats.levene ("median"/"mean"/"trimmed").

    Returns
    -------
    dict: group_col, groups, n_per_group, variances, statistic,
          p_value, verdict, low_n_warning, error
    """
    result = {
        "group_col": group_col, "groups": [], "n_per_group": {},
        "variances": {}, "statistic": None, "p_value": None,
        "verdict": None, "low_n_warning": False, "error": None,
    }

    sub = df[[value_col, group_col]].dropna()
    groups_data = {
        str(g): grp[value_col].astype(float).to_numpy()
        for g, grp in sub.groupby(group_col, sort=False)
    }

    if len(groups_data) < 2:
        result["error"] = "Need at least 2 groups to test variance homogeneity."
        result["verdict"] = "Not tested (fewer than 2 groups)"
        return result

    too_small = [g for g, a in groups_data.items() if len(a) < MIN_N_FOR_TEST]
    if too_small:
        result["error"] = (
            f"Too few students in group(s) {too_small} (n<{MIN_N_FOR_TEST}) "
            "to test variance homogeneity."
        )
        result["verdict"] = "Not tested (n<3 in at least one group)"
        return result

    group_names = sorted(groups_data.keys())
    result["groups"] = group_names
    result["n_per_group"] = {g: int(len(groups_data[g])) for g in group_names}
    result["variances"] = {
        g: round(float(groups_data[g].var(ddof=1)), 4) for g in group_names
    }

    try:
        stat, p = stats.levene(*[groups_data[g] for g in group_names], center=center)
    except Exception as e:
        result["error"] = str(e)
        return result

    result["statistic"] = round(float(stat), 4)
    result["p_value"] = round(float(p), 4)
    result["low_n_warning"] = min(result["n_per_group"].values()) < LOW_N_THRESHOLD

    verdict = (
        "Consistent with equal variances (p ≥ .05)"
        if p >= 0.05 else
        "Deviates from equal variances (p < .05)"
    )
    if result["low_n_warning"]:
        verdict += " -- exploratory, n<20"
    result["verdict"] = verdict

    return result


# -----------------------------------------------------------------------
# Public function 4: detect_straight_lining
# -----------------------------------------------------------------------

def detect_straight_lining(
    canonical_df: pd.DataFrame,
    min_items: int = 3,
) -> pd.DataFrame:
    """
    Flag careless/invariant survey responding.

    Uses raw response_value (NOT item_score, which DatasetBuilder already
    reverse-codes -- using the adjusted score would mask a genuine
    "always pick the same option" pattern once reverse items get flipped).
    Applied to survey instruments only (Cognitive Engagement/SIMS) -- an
    all-correct/all-incorrect MCQ pattern is a legitimate performance
    outcome, not evidence of carelessness, so this check would be
    false-positive-prone if applied there.

    Parameters
    ----------
    canonical_df : pd.DataFrame
    min_items    : int -- minimum items answered before a single-value
                   pattern is meaningful (avoids flagging a 1-2 item
                   partial submission as "straight-lining").

    Returns
    -------
    pd.DataFrame -- one row per (user_id, instrument_key) survey pair:
        user_id, instrument_key, n_items, n_distinct_responses, flagged
    """
    cols = ["user_id", "instrument_key", "n_items", "n_distinct_responses", "flagged"]
    if canonical_df.empty:
        return pd.DataFrame(columns=cols)

    df = canonical_df[canonical_df["instrument_key"].apply(_is_survey)].copy()
    df = df[df["response_value"].notna()]
    df = df[df["response_value"].astype(str).str.strip() != ""]

    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for (uid, inst), g in df.groupby(["user_id", "instrument_key"], sort=False):
        n_items = len(g)
        n_distinct = g["response_value"].nunique()
        rows.append({
            "user_id": uid,
            "instrument_key": inst,
            "n_items": n_items,
            "n_distinct_responses": n_distinct,
            "flagged": bool(n_items >= min_items and n_distinct == 1),
        })

    return pd.DataFrame(rows, columns=cols)


# -----------------------------------------------------------------------
# Public function 5: compute_missingness_by_module
# -----------------------------------------------------------------------

def compute_missingness_by_module(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Participation rate per Content MCQ module (1-7), for a fatigue trend.

    Uses participation (>=1 response row for that module's instrument),
    not item-completion %, since MCQ items are randomized per-student
    draws from a larger bank (compute_assessment_scores()'s own docstring:
    "each student answers a different 20-item subset of 57") -- raw
    n_items_answered isn't comparable module-to-module without knowing
    each module's expected draw size. A declining participation trend
    across modules is the fatigue signal that matters, computable from
    canonical_df alone with no new data dependencies.

    Returns
    -------
    pd.DataFrame: module_id, module_num, n_participants, pct_of_cohort
        sorted by module_num.
    """
    cols = ["module_id", "module_num", "n_participants", "pct_of_cohort"]
    if canonical_df.empty:
        return pd.DataFrame(columns=cols)

    mcq_mask = canonical_df["instrument_key"].str.contains(
        "content_mcq_assessment", na=False
    )
    mcq_df = canonical_df[mcq_mask]
    if mcq_df.empty:
        return pd.DataFrame(columns=cols)

    total_cohort = canonical_df["user_id"].nunique()
    if total_cohort == 0:
        return pd.DataFrame(columns=cols)

    rows = []
    for inst, g in mcq_df.groupby("instrument_key", sort=False):
        n_part = g["user_id"].nunique()
        # instrument_key format: "module{N}_content_mcq_assessment"
        digits = "".join(ch for ch in inst.split("_")[0] if ch.isdigit())
        module_num = int(digits) if digits else 0
        rows.append({
            "module_id": f"module_{module_num}" if module_num else inst,
            "module_num": module_num,
            "n_participants": n_part,
            "pct_of_cohort": round(100.0 * n_part / total_cohort, 1),
        })

    return pd.DataFrame(rows, columns=cols).sort_values("module_num").reset_index(drop=True)


# -----------------------------------------------------------------------
# Public function 6: detect_outliers
# -----------------------------------------------------------------------

def detect_outliers(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    method: str = "iqr",
) -> pd.DataFrame:
    """
    Per-group outlier flag on a per-student score column.

    An outlier is always relative to its own comparison group (e.g. a
    specific instrument, or instrument+construct) -- not a single global
    label -- since "extreme" only means something relative to peers on
    the same measure.

    Parameters
    ----------
    df         : pd.DataFrame -- e.g. output of compute_assessment_scores()
                 or compute_construct_means()
    value_col  : str -- e.g. "pct_correct", "mean_score"
    group_cols : list of str -- e.g. ["instrument_key"]
    method     : "iqr" (Tukey's 1.5xIQR beyond Q1/Q3, the standard
                 convention) or "zscore" (|z| > 3)

    Returns
    -------
    pd.DataFrame -- one row per (group..., user_id): value, flagged
    """
    base_cols = group_cols + ["user_id", value_col, "flagged"]
    if df.empty or "user_id" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        vals = g[value_col].astype(float)
        if len(vals) < MIN_N_FOR_TEST:
            flags = pd.Series(False, index=g.index)
        elif method == "zscore":
            sd = vals.std(ddof=1)
            if sd == 0 or pd.isna(sd):
                flags = pd.Series(False, index=g.index)
            else:
                z = (vals - vals.mean()) / sd
                flags = z.abs() > 3
        else:  # iqr
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            # Standard Tukey fencing: when IQR=0 (e.g. a strong ceiling
            # effect -- the middle 50% all land on one value), the fences
            # collapse to that single point, so anything different from it
            # is correctly flagged. Deliberately NOT special-cased to
            # suppress flagging here -- doing so would silently hide the
            # clearest possible outlier case (one extreme value against an
            # otherwise-unanimous cluster).
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flags = (vals < lo) | (vals > hi)

        for idx, row in g.iterrows():
            out = dict(zip(group_cols, keys_tuple))
            out["user_id"] = row["user_id"]
            out[value_col] = row[value_col]
            out["flagged"] = bool(flags.loc[idx])
            rows.append(out)

    return pd.DataFrame(rows, columns=base_cols)


# -----------------------------------------------------------------------
# Public function 7: detect_excessive_missingness
# -----------------------------------------------------------------------

def detect_excessive_missingness(
    canonical_df: pd.DataFrame,
    instrument_keys: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Per-participant answered-item rate across a SPECIFIC set of
    instruments (the ones involved in the current inferential test),
    flagged below threshold. This is a per-participant exclusion
    candidate, distinct from compute_missingness_by_module()'s
    module-level trend view.

    Parameters
    ----------
    canonical_df    : pd.DataFrame
    instrument_keys : list of str -- the instrument(s) this test uses
                       (e.g. the pre+post pair for a paired comparison)
    threshold       : float -- flag if answered-item rate < threshold

    Returns
    -------
    pd.DataFrame: user_id, n_answered, n_expected, pct_answered, flagged
        n_expected = max items answered by any single participant for
        these instruments (a same-machine ceiling, avoiding a new
        dependency on the YAML question-bank loader).
    """
    cols = ["user_id", "n_answered", "n_expected", "pct_answered", "flagged"]
    if canonical_df.empty or not instrument_keys:
        return pd.DataFrame(columns=cols)

    df = canonical_df[canonical_df["instrument_key"].isin(instrument_keys)]
    df = df[df["item_score"].notna() | df["response_value"].notna()]
    if df.empty:
        return pd.DataFrame(columns=cols)

    counts = df.groupby("user_id").size()
    n_expected = int(counts.max()) if len(counts) else 0
    if n_expected == 0:
        return pd.DataFrame(columns=cols)

    rows = []
    for uid, n_answered in counts.items():
        pct = n_answered / n_expected
        rows.append({
            "user_id": uid,
            "n_answered": int(n_answered),
            "n_expected": n_expected,
            "pct_answered": round(pct, 3),
            "flagged": bool(pct < threshold),
        })

    return pd.DataFrame(rows, columns=cols)
