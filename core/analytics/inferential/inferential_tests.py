# core/analytics/inferential/inferential_tests.py
"""
Inferential Statistics Engine
==============================
Computes inferential tests from the canonical research dataset.

All computation is pure scipy + pandas — no Streamlit, no DB access.
Results are returned as plain dicts for the dashboard to render.

Architecture rules:
- Pure functions — same inputs always produce same outputs
- No global state, no side effects
- Each function validates its inputs and fails fast with clear messages
- Power analysis built analytically from scipy.stats — no extra packages

Public API:
-----------
run_paired_comparison(canonical_df, pre_instrument, post_instrument,
                      alpha=0.05, include_wilcoxon=False)
    Paired t-test (+ optional Wilcoxon) for pre vs post instruments.
    Includes Cohen's d and power analysis.

run_between_groups(canonical_df, instrument_key, group_col,
                   demographics_df=None, alpha=0.05)
    One-way ANOVA + Kruskal-Wallis for group differences.
    Includes eta-squared and power analysis.

run_repeated_measures(canonical_df, instrument_key, alpha=0.05)
    Friedman test across modules (module_id as time variable).
    Includes Kendall's W effect size.

LOW_N_THRESHOLD : int = 30
    Any group with n < LOW_N_THRESHOLD triggers a low_n_warning.
    Results are still computed and displayed — the warning is
    advisory, not a gate. Teachers decide how to interpret small-n
    results.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import math
import warnings

import pandas as pd
import numpy as np
from scipy import stats


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

LOW_N_THRESHOLD = 30      # warn when any group n < this
DEFAULT_ALPHA   = 0.05

# Cohen's d / eta-squared / Kendall's W benchmarks
_D_SMALL,  _D_MEDIUM,  _D_LARGE  = 0.2, 0.5, 0.8
_ETA_SMALL, _ETA_MEDIUM, _ETA_LARGE = 0.01, 0.06, 0.14
_W_SMALL,  _W_MEDIUM,  _W_LARGE  = 0.1, 0.3, 0.5


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _effect_label_d(d: float) -> str:
    d = abs(d)
    if d >= _D_LARGE:  return "large"
    if d >= _D_MEDIUM: return "medium"
    if d >= _D_SMALL:  return "small"
    return "negligible"


def _effect_label_eta(eta2: float) -> str:
    if eta2 >= _ETA_LARGE:  return "large"
    if eta2 >= _ETA_MEDIUM: return "medium"
    if eta2 >= _ETA_SMALL:  return "small"
    return "negligible"


def _effect_label_w(w: float) -> str:
    if w >= _W_LARGE:  return "large"
    if w >= _W_MEDIUM: return "medium"
    if w >= _W_SMALL:  return "small"
    return "negligible"


def _cohens_d_paired(diff: pd.Series) -> float:
    """
    Cohen's d for paired samples.
    d = mean(diff) / std(diff, ddof=1)
    """
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def _eta_squared_oneway(groups: List[pd.Series]) -> float:
    """
    Eta-squared for one-way ANOVA.
    η² = SS_between / SS_total
    """
    all_vals  = pd.concat(groups)
    grand_mean = all_vals.mean()
    ss_total   = ((all_vals - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for g in groups
    )
    return float(ss_between / ss_total)


def _kendalls_w(data: pd.DataFrame) -> float:
    """
    Kendall's W (coefficient of concordance) for repeated measures.

    Parameters
    ----------
    data : pd.DataFrame
        Shape (n_subjects, k_conditions). Each column is a time point.

    Returns
    -------
    float in [0, 1]. W=1 means perfect agreement across time points.
    """
    n, k = data.shape
    if k < 2 or n < 2:
        return 0.0

    # Rank each subject's scores across conditions
    ranked = data.rank(axis=1)
    col_sums = ranked.sum(axis=0)
    mean_col_sum = col_sums.mean()
    ss = ((col_sums - mean_col_sum) ** 2).sum()
    w = 12 * ss / (n ** 2 * (k ** 3 - k))
    return float(np.clip(w, 0.0, 1.0))


# -----------------------------------------------------------------------
# Power analysis (analytical, scipy only)
# -----------------------------------------------------------------------

def _power_paired_ttest(
    n: int,
    cohens_d: float,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """
    Compute achieved power for a paired t-test given n and Cohen's d.

    Uses the non-central t-distribution:
        ncp    = |d| * sqrt(n)
        df     = n - 1
        t_crit = t_{1-alpha/2, df}  (two-tailed)
        power  = P(|T| > t_crit | T ~ t(df, ncp))
    """
    if n < 2 or cohens_d == 0:
        return 0.0
    df     = n - 1
    ncp    = abs(cohens_d) * math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # Power = P(T > t_crit) + P(T < -t_crit) under non-central t
    power  = (
        1 - stats.nct.cdf(t_crit,  df, ncp)
        +   stats.nct.cdf(-t_crit, df, ncp)
    )
    return float(np.clip(power, 0.0, 1.0))


def _n_needed_paired_ttest(
    cohens_d: float,
    target_power: float,
    alpha: float = DEFAULT_ALPHA,
    max_n: int = 500,
) -> int:
    """
    Find minimum n for a paired t-test to achieve target_power.
    Returns max_n if not achievable within that bound.
    """
    if cohens_d == 0:
        return max_n
    for n in range(2, max_n + 1):
        if _power_paired_ttest(n, cohens_d, alpha) >= target_power:
            return n
    return max_n


def _power_oneway_anova(
    n_per_group: int,
    k_groups: int,
    eta_squared: float,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """
    Approximate power for balanced one-way ANOVA.

    f² = eta² / (1 - eta²)
    ncp = f² * N  where N = n_per_group * k_groups
    df_between = k - 1,  df_within = N - k
    """
    if eta_squared <= 0 or eta_squared >= 1:
        return 0.0
    f2  = eta_squared / (1 - eta_squared)
    N   = n_per_group * k_groups
    ncp = f2 * N
    df1 = k_groups - 1
    df2 = N - k_groups
    if df2 <= 0:
        return 0.0
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    power  = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
    return float(np.clip(power, 0.0, 1.0))


def _n_needed_oneway_anova(
    k_groups: int,
    eta_squared: float,
    target_power: float,
    alpha: float = DEFAULT_ALPHA,
    max_n_per_group: int = 500,
) -> int:
    """Find minimum n per group for ANOVA to achieve target_power."""
    if eta_squared <= 0:
        return max_n_per_group
    for n in range(2, max_n_per_group + 1):
        if _power_oneway_anova(n, k_groups, eta_squared, alpha) >= target_power:
            return n
    return max_n_per_group


# -----------------------------------------------------------------------
# Score extraction helpers
# -----------------------------------------------------------------------

def _get_user_scores(
    canonical_df: pd.DataFrame,
    instrument_key: str,
) -> pd.Series:
    """
    Extract per-user total score for a given instrument.

    Handles both:
      - Direct match: instrument_key == df["instrument_key"]
      - Suffix match: df["instrument_key"].endswith("_" + instrument_key)

    Total score = sum(item_score) per user, NaN items excluded.
    Returns a Series indexed by user_id, named instrument_key.
    """
    mask = (
        (canonical_df["instrument_key"] == instrument_key) |
        canonical_df["instrument_key"].str.endswith("_" + instrument_key)
    )
    subset = canonical_df[mask].copy()

    if subset.empty:
        return pd.Series(dtype=float, name=instrument_key)

    scores = (
        subset[subset["item_score"].notna()]
        .groupby("user_id")["item_score"]
        .sum()
        .rename(instrument_key)
    )
    return scores


def _get_user_pct_correct(
    canonical_df: pd.DataFrame,
    instrument_key: str,
) -> pd.Series:
    """
    Extract per-user % correct (0–100) for a binary assessment.
    Denominator = items answered per user (not bank size).
    """
    mask = (
        (canonical_df["instrument_key"] == instrument_key) |
        canonical_df["instrument_key"].str.endswith("_" + instrument_key)
    )
    subset = canonical_df[mask].copy()

    if subset.empty:
        return pd.Series(dtype=float, name=instrument_key)

    valid = subset[subset["item_score"].notna()]
    agg = valid.groupby("user_id")["item_score"].agg(
        raw_score="sum",
        n_items="count",
    )
    pct = (agg["raw_score"] / agg["n_items"] * 100).rename(instrument_key)
    return pct


def _get_construct_means_by_module(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    construct: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract per-user per-module mean scores for a survey instrument.

    Returns a DataFrame with user_id as index and module_id columns.
    If construct is specified, filters to that construct only.
    """
    mask = (
        (canonical_df["instrument_key"] == instrument_key) |
        canonical_df["instrument_key"].str.endswith("_" + instrument_key)
    )
    subset = canonical_df[mask].copy()

    if construct:
        subset = subset[subset["construct"] == construct]

    subset = subset[subset["item_score"].notna()]

    if subset.empty:
        return pd.DataFrame()

    pivot = (
        subset.groupby(["user_id", "module_id"])["item_score"]
        .mean()
        .unstack("module_id")
    )
    return pivot


# -----------------------------------------------------------------------
# Public function 1: run_paired_comparison
# -----------------------------------------------------------------------

def run_paired_comparison(
    canonical_df: pd.DataFrame,
    pre_instrument: str,
    post_instrument: str,
    alpha: float = DEFAULT_ALPHA,
    include_wilcoxon: bool = False,
    use_pct: bool = True,
) -> Dict[str, Any]:
    """
    Paired t-test (+ optional Wilcoxon) for pre vs post instruments.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Output of DatasetBuilder.build().
    pre_instrument : str
        DB instrument_name for the pre measure.
    post_instrument : str
        DB instrument_name for the post measure.
    alpha : float
        Significance threshold. Default 0.05.
    include_wilcoxon : bool
        If True, also compute Wilcoxon signed-rank test.
    use_pct : bool
        If True, use % correct (0–100). If False, use raw sum scores.
        Default True — % correct is comparable across instruments
        with different numbers of items.

    Returns
    -------
    dict with keys:
        pre_instrument, post_instrument, n_pairs, use_pct,
        pre_mean, pre_std, post_mean, post_std, mean_diff,
        t_stat, t_p_value, significant,
        cohens_d, effect_size_label,
        wilcoxon_stat (None if not requested),
        wilcoxon_p    (None if not requested),
        power_achieved, n_needed_80, n_needed_95,
        low_n_warning, alpha,
        error (str | None) — set if computation failed
    """
    result: Dict[str, Any] = {
        "pre_instrument":  pre_instrument,
        "post_instrument": post_instrument,
        "alpha":           alpha,
        "use_pct":         use_pct,
        "error":           None,
    }

    try:
        get_fn = _get_user_pct_correct if use_pct else _get_user_scores
        pre_scores  = get_fn(canonical_df, pre_instrument)
        post_scores = get_fn(canonical_df, post_instrument)

        # Align on common user_ids (inner join)
        paired = pd.DataFrame({"pre": pre_scores, "post": post_scores}).dropna()
        n = len(paired)

        if n < 2:
            result["error"] = (
                f"Insufficient paired observations (n={n}). "
                f"Need at least 2 users with both pre and post scores."
            )
            return result

        diff = paired["post"] - paired["pre"]

        # Descriptive stats
        result["n_pairs"]   = n
        result["pre_mean"]  = round(float(paired["pre"].mean()),  4)
        result["pre_std"]   = round(float(paired["pre"].std(ddof=1)),  4)
        result["post_mean"] = round(float(paired["post"].mean()), 4)
        result["post_std"]  = round(float(paired["post"].std(ddof=1)), 4)
        result["mean_diff"] = round(float(diff.mean()), 4)

        # Paired t-test
        t_stat, t_p = stats.ttest_rel(paired["post"], paired["pre"])
        result["t_stat"]    = round(float(t_stat), 4)
        result["t_p_value"] = round(float(t_p),    6)
        result["significant"] = bool(t_p < alpha)

        # Effect size
        d = _cohens_d_paired(diff)
        result["cohens_d"]          = round(d, 4)
        result["effect_size_label"] = _effect_label_d(d)

        # Wilcoxon (optional)
        if include_wilcoxon:
            if (diff == 0).all():
                result["wilcoxon_stat"] = None
                result["wilcoxon_p"]    = None
                result["wilcoxon_note"] = "All differences are zero."
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w_stat, w_p = stats.wilcoxon(
                        paired["post"], paired["pre"],
                        alternative="two-sided",
                    )
                result["wilcoxon_stat"] = round(float(w_stat), 4)
                result["wilcoxon_p"]    = round(float(w_p),    6)
        else:
            result["wilcoxon_stat"] = None
            result["wilcoxon_p"]    = None

        # Power analysis
        power = _power_paired_ttest(n, d, alpha)
        result["power_achieved"] = round(power, 4)
        result["n_needed_80"]    = _n_needed_paired_ttest(d, 0.80, alpha)
        result["n_needed_95"]    = _n_needed_paired_ttest(d, 0.95, alpha)
        result["low_n_warning"]  = n < LOW_N_THRESHOLD

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Public function 2: run_between_groups
# -----------------------------------------------------------------------

def run_between_groups(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    group_col: str,
    demographics_df: Optional[pd.DataFrame] = None,
    alpha: float = DEFAULT_ALPHA,
    use_pct: bool = True,
) -> Dict[str, Any]:
    """
    One-way ANOVA + Kruskal-Wallis for group differences on an instrument.

    Parameters
    ----------
    canonical_df : pd.DataFrame
    instrument_key : str
        DB instrument_name to analyse.
    group_col : str
        Grouping variable. Must be in canonical_df OR demographics_df.
        Options: "grade", "cohort_id", "gender", "first_language_english"
    demographics_df : pd.DataFrame | None
        Required if group_col is "gender" or "first_language_english".
    alpha : float
    use_pct : bool

    Returns
    -------
    dict with keys:
        instrument_key, group_col, alpha, use_pct,
        groups, n_per_group, group_means, group_stds,
        f_stat, anova_p, significant,
        eta_squared, effect_size_label,
        kruskal_stat, kruskal_p,
        power_achieved, n_needed_80,
        low_n_warning, error
    """
    _DEMO_COLS = {"gender", "first_language_english"}

    result: Dict[str, Any] = {
        "instrument_key": instrument_key,
        "group_col":      group_col,
        "alpha":          alpha,
        "use_pct":        use_pct,
        "error":          None,
    }

    try:
        get_fn = _get_user_pct_correct if use_pct else _get_user_scores
        scores = get_fn(canonical_df, instrument_key).rename("score").reset_index()

        # Attach grouping column
        if group_col in _DEMO_COLS:
            if demographics_df is None:
                raise ValueError(
                    f"group_col='{group_col}' requires demographics_df."
                )
            scores = scores.merge(
                demographics_df[["user_id", group_col]],
                on="user_id", how="left",
            )
        elif "cohort_id" in canonical_df.columns and group_col == "cohort_id":
            cohort_map_df = (
                canonical_df[["user_id", "cohort_id"]]
                .drop_duplicates("user_id")
            )
            scores = scores.merge(cohort_map_df, on="user_id", how="left")
        elif group_col == "grade" and demographics_df is not None:
            scores = scores.merge(
                demographics_df[["user_id", "grade"]],
                on="user_id", how="left",
            )
        else:
            raise ValueError(
                f"group_col='{group_col}' not available. "
                f"Use: grade, cohort_id, gender, first_language_english."
            )

        scores = scores.dropna(subset=["score", group_col])

        if scores.empty:
            result["error"] = "No data after merging group column."
            return result

        # Build per-group lists
        groups_data = {
            str(g): grp["score"].values
            for g, grp in scores.groupby(group_col)
            if len(grp) >= 1
        }

        if len(groups_data) < 2:
            result["error"] = (
                f"Need at least 2 groups with data. "
                f"Found: {list(groups_data.keys())}"
            )
            return result

        group_names  = sorted(groups_data.keys())
        group_arrays = [groups_data[g] for g in group_names]

        result["groups"]      = group_names
        result["n_per_group"] = {g: int(len(groups_data[g])) for g in group_names}
        result["group_means"] = {
            g: round(float(groups_data[g].mean()), 4) for g in group_names
        }
        result["group_stds"] = {
            g: round(float(groups_data[g].std(ddof=1)), 4)
            if len(groups_data[g]) > 1 else 0.0
            for g in group_names
        }

        # One-way ANOVA
        f_stat, anova_p = stats.f_oneway(*group_arrays)
        result["f_stat"]    = round(float(f_stat),  4)
        result["anova_p"]   = round(float(anova_p), 6)
        result["significant"] = bool(anova_p < alpha)

        # Effect size: eta-squared
        eta2 = _eta_squared_oneway([pd.Series(a) for a in group_arrays])
        result["eta_squared"]       = round(eta2, 4)
        result["effect_size_label"] = _effect_label_eta(eta2)

        # Kruskal-Wallis
        kruskal_stat, kruskal_p = stats.kruskal(*group_arrays)
        result["kruskal_stat"] = round(float(kruskal_stat), 4)
        result["kruskal_p"]    = round(float(kruskal_p),    6)

        # Power analysis (balanced: use mean n per group)
        k = len(group_names)
        mean_n = int(round(sum(result["n_per_group"].values()) / k))
        power  = _power_oneway_anova(mean_n, k, eta2, alpha)
        result["power_achieved"] = round(power, 4)
        result["n_needed_80"]    = _n_needed_oneway_anova(k, eta2, 0.80, alpha)

        min_n = min(result["n_per_group"].values())
        result["low_n_warning"] = min_n < LOW_N_THRESHOLD

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Public function 3: run_repeated_measures
# -----------------------------------------------------------------------

def run_repeated_measures(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    construct: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """
    Friedman test across modules (module_id as within-subject time variable).
    Kendall's W as effect size.

    Use for:
    - Survey construct means across modules 1–7
    - MCQ % correct across modules 1–7

    Parameters
    ----------
    canonical_df : pd.DataFrame
    instrument_key : str
        Survey base key (e.g. "b4ai_sims_survey") or MCQ key
        (e.g. "module_content_mcq_assessment" — partial match used).
    construct : str | None
        For surveys: filter to this construct.
        For MCQ: leave None.
    alpha : float

    Returns
    -------
    dict with keys:
        instrument_key, construct, alpha,
        time_points, n_subjects,
        means_by_time, stds_by_time,
        friedman_stat, p_value, significant,
        kendalls_w, effect_size_label,
        low_n_warning, error
    """
    result: Dict[str, Any] = {
        "instrument_key": instrument_key,
        "construct":      construct,
        "alpha":          alpha,
        "error":          None,
    }

    try:
        if construct is not None:
            # Survey construct means per module
            pivot = _get_construct_means_by_module(
                canonical_df, instrument_key, construct
            )
        else:
            # MCQ % correct per module
            mask = (
                canonical_df["instrument_key"].str.contains(
                    instrument_key, regex=False
                )
            )
            subset = canonical_df[mask & canonical_df["item_score"].notna()].copy()

            if subset.empty:
                result["error"] = f"No data for instrument '{instrument_key}'."
                return result

            agg = (
                subset.groupby(["user_id", "module_id"])["item_score"]
                .agg(raw="sum", n="count")
                .assign(pct=lambda d: d["raw"] / d["n"] * 100)
                .reset_index()
            )
            pivot = agg.pivot(index="user_id", columns="module_id", values="pct")

        if pivot.empty or pivot.shape[1] < 2:
            result["error"] = (
                "Need at least 2 time points. "
                f"Found columns: {list(pivot.columns)}"
            )
            return result

        # Sort columns by module number for logical ordering
        def _module_sort_key(col):
            import re
            m = re.search(r"(\d+)", str(col))
            return int(m.group(1)) if m else 0

        pivot = pivot[sorted(pivot.columns, key=_module_sort_key)]

        # Drop users with any missing time point (Friedman requires complete data)
        pivot = pivot.dropna()
        n = len(pivot)

        if n < 2:
            result["error"] = (
                f"Insufficient complete cases for Friedman test (n={n}). "
                f"Need at least 2 subjects with data at all time points."
            )
            return result

        time_points = list(pivot.columns)

        result["time_points"] = time_points
        result["n_subjects"]  = n
        result["means_by_time"] = {
            str(t): round(float(pivot[t].mean()), 4) for t in time_points
        }
        result["stds_by_time"] = {
            str(t): round(float(pivot[t].std(ddof=1)), 4)
            if n > 1 else 0.0
            for t in time_points
        }

        # Friedman test
        friedman_stat, p_value = stats.friedmanchisquare(
            *[pivot[t].values for t in time_points]
        )
        result["friedman_stat"] = round(float(friedman_stat), 4)
        result["p_value"]       = round(float(p_value),       6)
        result["significant"]   = bool(p_value < alpha)

        # Kendall's W
        w = _kendalls_w(pivot)
        result["kendalls_w"]        = round(w, 4)
        result["effect_size_label"] = _effect_label_w(w)

        result["low_n_warning"] = n < LOW_N_THRESHOLD

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Public function 4: run_bland_altman
# -----------------------------------------------------------------------

def run_bland_altman(
    canonical_df: pd.DataFrame,
    pre_instrument: str,
    post_instrument: str,
    use_pct: bool = True,
) -> Dict[str, Any]:
    """
    Bland-Altman method agreement analysis for paired pre/post instruments.

    Computes the mean difference (bias), SD of differences (random error),
    and 95% limits of agreement for each paired observation.

    Statistical method
    ------------------
    For each subject i with pre score A_i and post score B_i:

        diff_i   = A_i - B_i            (pre minus post)
        mean_i   = (A_i + B_i) / 2

    Summary statistics:
        d_bar = mean(diff_i)            <- systematic bias
        s     = SD(diff_i, ddof=1)      <- random error
        LoA   = d_bar +/- 2s            <- ~95% of individual differences

    A Pearson correlation between diff_i and mean_i is computed to test
    for proportional bias. If the correlation p-value < 0.05, the
    measurement error grows with the size of the measurement; a log
    transformation of raw scores is recommended before re-running.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Output of DatasetBuilder.build().
    pre_instrument : str
        DB instrument_name for the pre measure (e.g.
        "precourse_pre_ai_misconceptions_assessment").
    post_instrument : str
        DB instrument_name for the post measure.
    use_pct : bool
        If True, use % correct (0-100) -- recommended for cross-instrument
        comparability. If False, use raw sum scores.

    Returns
    -------
    dict with keys:
        pre_instrument, post_instrument, use_pct,
        n_pairs,
        mean_diff       -- d_bar (systematic bias; +ve = pre > post on average),
        sd_diff         -- s (random error),
        loa_lower       -- d_bar - 2s,
        loa_upper       -- d_bar + 2s,
        proportional_bias_r  -- Pearson r(diff, mean),
        proportional_bias_p  -- p-value for that correlation,
        proportional_bias    -- bool: True if p < 0.05 (consider log transform),
        per_pair_df     -- pd.DataFrame with columns:
                           user_id | pre | post | diff | mean_val
        low_n_warning   -- bool: n < LOW_N_THRESHOLD,
        error           -- str | None

    Reference
    ---------
    Bland, J. M., & Altman, D. G. (1990). A note on the use of the
    intraclass correlation coefficient in the evaluation of agreement
    between two methods of measurement. Computers in Biology and
    Medicine, 20(5), 337-340.
    https://doi.org/10.1016/0010-4825(90)90013-F
    """
    result: Dict[str, Any] = {
        "pre_instrument":  pre_instrument,
        "post_instrument": post_instrument,
        "use_pct":         use_pct,
        "error":           None,
        "per_pair_df":     None,
    }

    try:
        get_fn = _get_user_pct_correct if use_pct else _get_user_scores
        pre_scores  = get_fn(canonical_df, pre_instrument)
        post_scores = get_fn(canonical_df, post_instrument)

        # Inner join -- only subjects with both pre and post scores
        paired = pd.DataFrame({
            "pre":  pre_scores,
            "post": post_scores,
        }).dropna()

        n = len(paired)

        if n < 2:
            result["error"] = (
                f"Insufficient paired observations (n={n}). "
                f"Need at least 2 subjects with both pre and post scores."
            )
            return result

        # Per-pair statistics (Bland-Altman 1990)
        diff     = paired["pre"] - paired["post"]        # A - B
        mean_val = (paired["pre"] + paired["post"]) / 2  # (A + B) / 2

        d_bar = float(diff.mean())
        s     = float(diff.std(ddof=1))

        result["n_pairs"]    = n
        result["mean_diff"]  = round(d_bar, 4)
        result["sd_diff"]    = round(s,     4)
        result["loa_lower"]  = round(d_bar - 2 * s, 4)
        result["loa_upper"]  = round(d_bar + 2 * s, 4)

        # Proportional bias: Pearson r between differences and means.
        # If significant (p < 0.05), error grows with measurement size.
        if n >= 3:
            r_val, r_p = stats.pearsonr(diff, mean_val)
            result["proportional_bias_r"] = round(float(r_val), 4)
            result["proportional_bias_p"] = round(float(r_p),   6)
            result["proportional_bias"]   = bool(r_p < 0.05)
        else:
            result["proportional_bias_r"] = None
            result["proportional_bias_p"] = None
            result["proportional_bias"]   = False

        # Per-pair table for dashboard display
        per_pair = pd.DataFrame({
            "user_id":  paired.index,
            "pre":      paired["pre"].round(2).values,
            "post":     paired["post"].round(2).values,
            "diff":     diff.round(2).values,
            "mean_val": mean_val.round(2).values,
        })
        result["per_pair_df"] = per_pair

        result["low_n_warning"] = n < LOW_N_THRESHOLD

    except Exception as e:
        result["error"] = str(e)

    return result
