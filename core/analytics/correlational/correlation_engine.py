# core/analytics/correlational/correlation_engine.py
"""
Correlational Analysis Engine
==============================
Does engagement/motivation relate to assessment performance? Answers
this in four phases, matching the confirmed research plan for the
Correlations tab (Task L):

Phase 0 — Reliability & redundancy screening (assumption gate before
    any sub-construct is trusted as a predictor).
Phase 1 — Composite construction & RAI (collapse redundant
    sub-constructs, compute the SIMS Relative Autonomy Index).
Phase 2 — Person-mean-centered mixed-effects model (primary analysis:
    within-person vs. between-person effects, module as fixed effect,
    random intercept for student).
Phase 3 — Repeated-measures correlation + FDR (supplementary,
    reporting-friendly companion to Phase 2).

Architecture rules (matching normality_checks.py / inferential_tests.py):
- Pure pandas/scipy/statsmodels/pingouin -- no DB access, no Streamlit
- All inputs are DataFrames/dicts; all outputs are DataFrames/dicts
- pingouin/statsmodels imported lazily inside functions, wrapped in
  try/except ImportError, so this module stays importable even if
  either package is ever missing (mirrors inferential_tests.py's own
  lazy `import pingouin as pg` pattern in run_repeated_measures()).

References
----------
Rotgans, J. I., & Schmidt, H. G. (2011). Cognitive engagement in the
    problem-based learning classroom. Advances in Health Sciences
    Education, 16(4), 465-479.
Heddy, B. C., Taasoobshirazi, G., Chancey, J. B., & Danielson, R. W.
    (2018). Developing and validating a Conceptual Change Cognitive
    Engagement Instrument. Frontiers in Education, 3:43. (Their own EFA,
    N=513, found the 4 "message appraisal" sub-constructs -- Coherency,
    Plausibility, Credibility, Comprehensibility -- collapse into one
    empirical factor; Attention+Culture form a second; Personal_relevance
    a third. This is the a priori basis for DEFAULT_COMPOSITE_MAP below.)
Guay, F., Vallerand, R. J., & Blanchard, C. (2000). On the assessment of
    situational intrinsic and extrinsic motivation: The Situational
    Motivation Scale (SIMS). Motivation and Emotion, 24(3), 175-213.
Ryan, R. M., & Connell, J. P. (1989). Perceived locus of causality and
    internalization: Examining reasons for acting in two domains.
    Journal of Personality and Social Psychology, 57(5), 749-761.
    (Source of the Relative Autonomy Index formula -- NOT Guay et al.
    2000, which is the SIMS source paper but never presents RAI.)
Eisinga, R., Te Grotenhuis, M., & Pelzer, B. (2013). The reliability of
    a two-item scale: Pearson, Cronbach, or Spearman-Brown? International
    Journal of Public Health, 58(4), 637-642.
Enders, C. K., & Tofighi, D. (2007). Centering predictors in multilevel
    regression models: A new look at an old issue. Psychological
    Methods, 12(2), 121-138.
Pinheiro, J. C., & Bates, D. M. (2000). Mixed-Effects Models in S and
    S-PLUS. Springer. (Sec. 2.4 -- REML likelihoods are only comparable
    across models with identical fixed effects; use ML for model
    comparison, refit the chosen model with REML for reporting.)
Bakdash, J. Z., & Marusich, L. R. (2017). Repeated measures correlation.
    Frontiers in Psychology, 8:456.
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate: A practical and powerful approach to multiple testing. Journal
    of the Royal Statistical Society: Series B, 57(1), 289-300.
Curran-Everett, D. (2000). Multiple comparisons: philosophies and
    illustrations. American Journal of Physiology-Regulatory,
    Integrative and Comparative Physiology, 279(1), R1-R8. (A
    physiology-methods paper, not education-specific -- cited here only
    as supporting rationale for FDR over Bonferroni in exploratory
    multiple-testing, alongside the foundational Benjamini & Hochberg
    1995 citation above.)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import re

import numpy as np
import pandas as pd
from scipy import stats

from core.analytics.descriptive.score_aggregator import compute_construct_means
from core.analytics.irt.reliability_analysis import alpha_badge

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

LOW_N_THRESHOLD = 30
MIN_ITEMS_FOR_ALPHA = 3
DEFAULT_ALPHA = 0.05

# Theory-driven default grouping, per the confirmed plan. Construct
# names here are lowercase_snake_case to match canonical_df's own
# convention (confirmed live: compute_construct_means() documents
# "construct: str canonical lowercase_snake_case name", and this was
# directly verified against real pilot data -- the YAML section names
# are TitleCase, e.g. "Engagement_with_task", but DatasetBuilder lowers
# them before they reach canonical_df).
# - Situational Cognitive Engagement: Rotgans & Schmidt (2011)'s own
#   validated 3-facet structure (task engagement, effort/persistence,
#   flow), fit as a single latent construct via CFA (coefficient H =
#   .93 exploration / .78 cross-validation). Attention is deliberately
#   NOT included here -- see Attention_Culture below.
# - Message_appraisal: Heddy et al. (2018)'s own EFA collapses these 4
#   sub-constructs into one empirical factor.
# - Attention_Culture: Heddy et al. (2018)'s own EFA found Attention
#   and Culture load together as a second, distinct empirical factor --
#   this composite mirrors that factor exactly, rather than grouping
#   Attention with the SCES facets (an earlier version of this map did
#   so; corrected 2026-09-01 to match Heddy et al.'s reported structure).
# - Personal_relevance: kept standalone, matching Heddy et al.'s 3rd
#   factor and the confirmed plan.
DEFAULT_SCCCES_COMPOSITE_MAP: Dict[str, List[str]] = {
    "Situational Cognitive Engagement": [
        "engagement_with_task", "effort_and_persistence",
        "experience_of_flow",
    ],
    "Message_appraisal": [
        "coherency_of_messaging", "plausibility_of_messaging",
        "credibility_of_messaging", "comprehensibility_of_messaging",
    ],
    "Attention_Culture": ["attention", "culture"],
    "Personal_relevance": ["personal_relevance"],
}

# SIMS constructs feed compute_rai() directly, or stand alone
# (Amotivation) -- not part of compute_composite_scores()'s map.
SIMS_RAI_CONSTRUCTS = {
    "intrinsic": "intrinsic_motivation",
    "identified": "identified_regulation",
    "external": "external_regulation",
    "amotivation": "amotivation",
}


def _resolve_instrument_keys(canonical_df: pd.DataFrame, base_instrument_key: str) -> List[str]:
    """
    Real instrument_key values in canonical_df are module-prefixed
    (e.g. "module1_b4ai_sccces_survey"), not the bare survey base key
    -- compute_construct_means()'s own instrument_keys= filter is an
    EXACT match with no suffix fallback (unlike inferential_tests.py's
    _get_user_scores()/_get_user_pct_correct(), which do fall back to a
    suffix match). Resolve the bare key to the real, present keys first
    so callers here get the same "just give me the base survey name"
    convenience those other functions already offer.
    """
    all_keys = canonical_df["instrument_key"].unique()
    return [
        k for k in all_keys
        if k == base_instrument_key or str(k).endswith("_" + base_instrument_key)
    ]


def _module_sort_key(col) -> int:
    """Extract the leading integer from a module_id-like string for
    sorting (e.g. 'module_3' -> 3). Mirrors inferential_tests.py's own
    _module_sort_key() closure."""
    m = re.search(r"(\d+)", str(col))
    return int(m.group(1)) if m else 0


# =========================================================================
# Phase 0 -- Reliability & redundancy screening
# =========================================================================

def _spearman_brown(r: float) -> float:
    """
    Spearman-Brown step-up formula for a 2-item scale's reliability.
    r_sb = 2r / (1+r), where r is the Pearson correlation between the
    two items. Eisinga, Te Grotenhuis & Pelzer (2013) -- recommended
    over raw Cronbach's alpha for exactly-2-item scales.
    """
    if r is None or np.isnan(r) or r <= -1:
        return float("nan")
    return (2 * r) / (1 + r)


def spearman_brown_badge(value: float) -> str:
    """Same 0.70/0.80/0.90 convention commonly applied to Spearman-Brown
    reliability as to Cronbach's alpha, but kept as its own labeled
    function rather than silently reusing alpha_badge() for a different
    reliability coefficient."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if value >= 0.90:
        return "🟢 Excellent (≥0.90)"
    if value >= 0.80:
        return "🟢 Good (≥0.80)"
    if value >= 0.70:
        return "🟡 Acceptable (≥0.70)"
    return "🔴 Poor (<0.70)"


def compute_construct_reliability(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    construct: Optional[str] = None,
) -> pd.DataFrame:
    """
    Per-sub-construct reliability: Cronbach's alpha (3+ items),
    Spearman-Brown-corrected reliability (exactly 2 items), or an
    explicit "not estimable" label (1 item) -- never a blank or
    misleading 0.0.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Output of DatasetBuilder.build().
    instrument_key : str
        Survey base key (e.g. "b4ai_sccces_survey").
    construct : str | None
        If given, restrict to this construct only.

    Returns
    -------
    pd.DataFrame: instrument_key, construct, n_items, n_subjects,
        method, value, ci_low, ci_high, badge, low_n_warning, error
    """
    mask = (
        (canonical_df["instrument_key"] == instrument_key) |
        canonical_df["instrument_key"].str.endswith("_" + instrument_key)
    )
    df = canonical_df[mask].copy()
    df = df[df["construct"].notna() & (df["construct"] != "")]
    df = df[df["item_score"].notna()]

    if construct is not None:
        df = df[df["construct"] == construct]

    if df.empty:
        return pd.DataFrame(columns=[
            "instrument_key", "construct", "n_items", "n_subjects",
            "method", "value", "ci_low", "ci_high", "badge",
            "low_n_warning", "error",
        ])

    rows = []
    for cons, g in df.groupby("construct", sort=False):
        n_items = g["question_id"].nunique()
        n_subjects = g["user_id"].nunique()
        row: Dict[str, Any] = {
            "instrument_key": instrument_key, "construct": cons,
            "n_items": n_items, "n_subjects": n_subjects,
            "method": None, "value": None, "ci_low": None, "ci_high": None,
            "badge": "—", "low_n_warning": n_subjects < LOW_N_THRESHOLD,
            "error": None,
        }

        if n_items < 1:
            row["error"] = "No items found."
            rows.append(row)
            continue

        if n_items == 1:
            row["method"] = "single_item_no_reliability"
            row["error"] = "Single-item indicator -- reliability not estimable."
            rows.append(row)
            continue

        # Need a complete-case user x item matrix for both alpha and
        # the 2-item Spearman-Brown path.
        wide = g.pivot_table(
            index="user_id", columns="question_id", values="item_score",
        ).dropna()

        if len(wide) < 3:
            row["method"] = "not_tested"
            row["error"] = f"Too few complete cases (n={len(wide)}) to test reliability."
            rows.append(row)
            continue

        if n_items == 2:
            row["method"] = "spearman_brown"
            cols = list(wide.columns)
            r_val = wide[cols[0]].corr(wide[cols[1]])
            sb = _spearman_brown(r_val)
            row["value"] = round(float(sb), 4) if not np.isnan(sb) else None
            row["badge"] = spearman_brown_badge(sb)
        else:
            try:
                import pingouin as pg
                long_df = wide.reset_index().melt(
                    id_vars="user_id", var_name="question_id", value_name="item_score",
                )
                alpha, ci = pg.cronbach_alpha(
                    data=long_df, items="question_id",
                    scores="item_score", subject="user_id",
                )
                row["method"] = "cronbach_alpha"
                row["value"] = round(float(alpha), 4)
                row["ci_low"] = round(float(ci[0]), 4)
                row["ci_high"] = round(float(ci[1]), 4)
                row["badge"] = alpha_badge(alpha)
            except ImportError:
                row["error"] = "pingouin not installed -- Cronbach's alpha unavailable."
            except Exception as e:
                row["error"] = str(e)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_inter_construct_correlation_matrix(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    method: str = "pearson",
) -> dict:
    """
    Cross-module, per-student construct-mean correlation matrix -- a
    live, dataset-specific decision aid for the Composite Builder
    (Phase 1), alongside Heddy et al. (2018)'s a priori EFA finding.
    Does not gate anything by itself.

    Returns
    -------
    dict: instrument_key, method, matrix (pd.DataFrame | None),
        n_per_pair (dict[(c1,c2)] -> int), error
    """
    result: Dict[str, Any] = {
        "instrument_key": instrument_key, "method": method,
        "matrix": None, "n_per_pair": {}, "error": None,
    }
    try:
        resolved_keys = _resolve_instrument_keys(canonical_df, instrument_key)
        if not resolved_keys:
            result["error"] = f"No survey data for '{instrument_key}'."
            return result
        cm = compute_construct_means(canonical_df, instrument_keys=resolved_keys)
        if cm.empty:
            result["error"] = f"No survey data for '{instrument_key}'."
            return result

        # Trait-level: pool across modules per (user, construct).
        trait = (
            cm.groupby(["user_id", "construct"])["mean_score"]
            .mean()
            .reset_index()
        )
        wide = trait.pivot(index="user_id", columns="construct", values="mean_score")

        if wide.shape[1] < 2:
            result["error"] = "Need at least 2 constructs to build a correlation matrix."
            return result

        result["matrix"] = wide.corr(method=method).round(4)
        cols = list(wide.columns)
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                n = wide[[c1, c2]].dropna().shape[0]
                result["n_per_pair"][(c1, c2)] = n
    except Exception as e:
        result["error"] = str(e)

    return result


# =========================================================================
# Phase 1 -- Composite construction & RAI
# =========================================================================

def compute_composite_scores(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    composite_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Item-weighted average across named sub-constructs into a composite,
    per (user, module) -- preserves the per-module grain Phase 2 needs
    (deliberately NOT built on aggregate_construct_means(), which
    collapses across modules for a different use case).

    Parameters
    ----------
    canonical_df : pd.DataFrame
    instrument_key : str
    composite_map : dict
        composite_name -> list of construct names to combine. A
        construct used standalone is just a one-element list.

    Returns
    -------
    pd.DataFrame: user_id, module_id, composite, n_subconstructs,
        n_items_total, total_score, mean_score, error
    """
    mask = (
        (canonical_df["instrument_key"] == instrument_key) |
        canonical_df["instrument_key"].str.endswith("_" + instrument_key)
    )
    df = canonical_df[mask].copy()
    df = df[df["construct"].notna() & (df["construct"] != "")]
    df = df[df["item_score"].notna()]

    if df.empty:
        return pd.DataFrame(columns=[
            "user_id", "module_id", "composite", "n_subconstructs",
            "n_items_total", "total_score", "mean_score", "error",
        ])

    rows = []
    for composite_name, constructs in composite_map.items():
        sub = df[df["construct"].isin(constructs)]
        if sub.empty:
            continue
        grouped = sub.groupby(["user_id", "module_id"])
        agg = grouped["item_score"].agg(total_score="sum", n_items_total="count").reset_index()
        n_sub = grouped["construct"].nunique().reset_index(name="n_subconstructs")
        agg = agg.merge(n_sub, on=["user_id", "module_id"], how="left")
        agg["composite"] = composite_name
        agg["mean_score"] = agg["total_score"] / agg["n_items_total"]
        agg["error"] = None
        rows.append(agg)

    if not rows:
        return pd.DataFrame(columns=[
            "user_id", "module_id", "composite", "n_subconstructs",
            "n_items_total", "total_score", "mean_score", "error",
        ])

    out = pd.concat(rows, ignore_index=True)
    return out[[
        "user_id", "module_id", "composite", "n_subconstructs",
        "n_items_total", "total_score", "mean_score", "error",
    ]]


def compute_rai(canonical_df: pd.DataFrame, sims_instrument_key: str = "b4ai_sims_survey") -> pd.DataFrame:
    """
    Relative Autonomy Index from the SIMS sub-constructs:
        RAI = 2*Intrinsic + 1*Identified - 1*External - 2*Amotivation
    Ryan & Connell (1989) -- NOT Guay et al. (2000), which is the SIMS
    source paper but never presents the RAI formula itself (verified by
    reading it in full, including references and appendix).

    NaN (not a partial score) when any of the 4 SIMS sub-constructs is
    missing for that user x module.

    Returns
    -------
    pd.DataFrame: user_id, module_id, rai, n_subscales_present, error
    """
    resolved_keys = _resolve_instrument_keys(canonical_df, sims_instrument_key)
    if not resolved_keys:
        return pd.DataFrame(columns=[
            "user_id", "module_id", "rai", "n_subscales_present", "error",
        ])
    cm = compute_construct_means(canonical_df, instrument_keys=resolved_keys)
    if cm.empty:
        return pd.DataFrame(columns=[
            "user_id", "module_id", "rai", "n_subscales_present", "error",
        ])

    wide = cm.pivot_table(
        index=["user_id", "module_id"], columns="construct", values="mean_score",
    ).reset_index()

    needed = list(SIMS_RAI_CONSTRUCTS.values())
    present = [c for c in needed if c in wide.columns]
    wide["n_subscales_present"] = wide[present].notna().sum(axis=1) if present else 0

    def _rai_row(row):
        if row["n_subscales_present"] != 4:
            return np.nan
        return (
            2 * row.get("intrinsic_motivation", np.nan)
            + 1 * row.get("identified_regulation", np.nan)
            - 1 * row.get("external_regulation", np.nan)
            - 2 * row.get("amotivation", np.nan)
        )

    wide["rai"] = wide.apply(_rai_row, axis=1)
    wide["error"] = None
    return wide[["user_id", "module_id", "rai", "n_subscales_present", "error"]]


# =========================================================================
# Phase 2 -- Person-mean-centered mixed-effects model
# =========================================================================

def build_person_module_dataset(
    canonical_df: pd.DataFrame,
    predictor_frames: Dict[str, pd.DataFrame],
    mcq_instrument_substr: str = "content_mcq_assessment",
) -> pd.DataFrame:
    """
    (user_id, module_id) grain long dataset: MCQ % correct joined with
    one or more predictor scores (composites and/or RAI).

    Parameters
    ----------
    canonical_df : pd.DataFrame
    predictor_frames : dict
        predictor_name -> DataFrame with columns [user_id, module_id,
        <value_col>] where value_col is "mean_score" (composites, from
        compute_composite_scores(), pre-filtered to one composite) or
        "rai" (from compute_rai()). Each frame's non-key column is
        renamed to the dict key on join.
    mcq_instrument_substr : str
        Same partial-match convention already used by
        run_repeated_measures()'s MCQ branch in inferential_tests.py.

    Returns
    -------
    pd.DataFrame: user_id, module_id, module_num, pct_correct,
        <predictor columns...>
    """
    mask = canonical_df["instrument_key"].str.contains(mcq_instrument_substr, regex=False)
    subset = canonical_df[mask & canonical_df["item_score"].notna()].copy()
    if subset.empty:
        return pd.DataFrame(columns=["user_id", "module_id", "module_num", "pct_correct"])

    outcome = (
        subset.groupby(["user_id", "module_id"])["item_score"]
        .agg(raw="sum", n="count")
        .assign(pct_correct=lambda d: d["raw"] / d["n"] * 100)
        .reset_index()[["user_id", "module_id", "pct_correct"]]
    )

    merged = outcome
    for name, frame in predictor_frames.items():
        value_col = "mean_score" if "mean_score" in frame.columns else "rai"
        f = frame[["user_id", "module_id", value_col]].rename(columns={value_col: name})
        merged = merged.merge(f, on=["user_id", "module_id"], how="left")

    merged["module_num"] = merged["module_id"].apply(_module_sort_key)
    return merged


def person_mean_center(
    df: pd.DataFrame,
    predictor_cols: List[str],
    id_col: str = "user_id",
) -> pd.DataFrame:
    """
    For each predictor column, adds <col>_within (person-mean-centered,
    the within-person effect) and <col>_between (that person's own
    cross-module mean, the between-person effect). Enders & Tofighi
    (2007).
    """
    out = df.copy()
    for col in predictor_cols:
        person_mean = out.groupby(id_col)[col].transform("mean")
        out[f"{col}_between"] = person_mean
        out[f"{col}_within"] = out[col] - person_mean
    return out


def run_mixed_model(
    long_df: pd.DataFrame,
    outcome_col: str,
    within_cols: List[str],
    between_cols: List[str],
    id_col: str = "user_id",
    alpha: float = DEFAULT_ALPHA,
    try_random_slope: bool = True,
) -> dict:
    """
    Person-mean-centered mixed-effects model, built in blocks and
    compared via AIC/BIC/LRT:
        M0  intercept + (1|user_id)
        M1  + module_num (fixed)
        M2  + within-person predictor terms
        M3  + between-person predictor terms

    All nested-block comparisons use reml=False (ML) -- REML
    likelihoods are only comparable across models with identical fixed
    effects (Pinheiro & Bates 2000, sec 2.4). Both the best-by-BIC block
    (the more conservative criterion given n~75-110) and the best-by-AIC
    block (when different) are refit with reml=True for reporting-
    quality SEs -- both are shown to the researcher as primary output,
    not just BIC's pick, so both need accurate SEs, not just the ML fit
    used for model selection.

    A random-slope variant (module_num | user_id) is attempted behind
    its own try/except when try_random_slope=True, falling back to
    random-intercept-only on non-convergence -- surfaced via
    `.converged`, never silently hidden.

    STANDING CAVEAT: statsmodels' MixedLM has no Kenward-Roger /
    Satterthwaite small-sample SE correction. At n~75-110 students,
    coefficients use asymptotic (z-based) SEs -- treat these p-values
    as more exploratory than the paired-t/ANOVA results elsewhere in
    this app. This module always populates `small_sample_caveat` so the
    UI layer can surface it without re-deriving the wording.

    Returns
    -------
    dict: blocks (dict: name -> {aic, bic, llf, converged, n_obs,
        n_groups, formula, params: {term: {estimate, se, z, p}}, error}),
        lrt (list of {from_block, to_block, lr_stat, df, p_value}),
        best_block_by_aic, best_block_by_bic, vif (dict | None),
        small_sample_caveat, low_n_warning, error
    """
    result: Dict[str, Any] = {
        "blocks": {}, "lrt": [], "best_block_by_aic": None,
        "best_block_by_bic": None, "vif": None,
        "small_sample_caveat": (
            "statsmodels' MixedLM has no small-sample (Kenward-Roger / "
            "Satterthwaite) standard-error correction. At this sample "
            "size, treat these p-values as more exploratory than the "
            "paired-t/ANOVA results elsewhere in this app."
        ),
        "low_n_warning": False, "error": None,
    }

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        result["error"] = "statsmodels not installed -- mixed-effects model unavailable."
        return result

    needed_cols = [outcome_col, "module_num", id_col] + within_cols + between_cols
    df = long_df.dropna(subset=needed_cols).copy()
    if df.empty:
        result["error"] = "No complete cases across the outcome, module, and selected predictors."
        return result

    n_groups = df[id_col].nunique()
    result["low_n_warning"] = n_groups < LOW_N_THRESHOLD

    def _fit_block(name, formula, reml=False, re_formula=None, display_formula=None):
        try:
            kwargs = {"re_formula": re_formula} if re_formula else {}
            model = smf.mixedlm(formula, df, groups=df[id_col], **kwargs)
            fit = model.fit(reml=reml)
            if not fit.converged:
                return {
                    "aic": None, "bic": None, "llf": None, "converged": False,
                    "n_obs": None, "n_groups": n_groups,
                    "formula": display_formula or formula, "params": {},
                    "error": f"{name} did not converge.",
                }
            params = {
                term: {
                    "estimate": round(float(fit.params[term]), 4),
                    "se": round(float(fit.bse[term]), 4),
                    "z": round(float(fit.tvalues[term]), 4) if term in fit.tvalues else None,
                    "p": round(float(fit.pvalues[term]), 6) if term in fit.pvalues else None,
                }
                for term in fit.params.index
            }
            return {
                "aic": round(float(fit.aic), 3), "bic": round(float(fit.bic), 3),
                "llf": round(float(fit.llf), 3), "converged": bool(fit.converged),
                "n_obs": int(fit.nobs), "n_groups": n_groups,
                "formula": display_formula or formula, "params": params, "error": None,
            }
        except Exception as e:
            return {
                "aic": None, "bic": None, "llf": None, "converged": False,
                "n_obs": None, "n_groups": n_groups,
                "formula": display_formula or formula,
                "params": {}, "error": str(e),
            }

    within_terms = " + ".join(f"{c}_within" for c in within_cols) if within_cols else None
    between_terms = " + ".join(f"{c}_between" for c in between_cols) if between_cols else None

    formulas = {
        "M0_null":      f"{outcome_col} ~ 1",
        "M1_module":    f"{outcome_col} ~ module_num",
    }
    if within_terms:
        formulas["M2_within"] = f"{outcome_col} ~ module_num + {within_terms}"
    if between_terms:
        rhs = "module_num"
        if within_terms:
            rhs += f" + {within_terms}"
        rhs += f" + {between_terms}"
        formulas["M3_full"] = f"{outcome_col} ~ {rhs}"

    # Track each block's (formula, re_formula) so the REML reporting
    # refit below can reuse the exact same specification, including for
    # the random-slope block -- both the BIC-best and AIC-best blocks
    # get this refit, not just whichever one BIC happens to prefer.
    block_specs: Dict[str, tuple] = {}

    for name, formula in formulas.items():
        result["blocks"][name] = _fit_block(name, formula, reml=False)
        block_specs[name] = (formula, None)

    if try_random_slope and "M3_full" in formulas:
        rs_formula = formulas["M3_full"]
        rs_display = rs_formula + " + (module_num | user_id)"
        result["blocks"]["M3b_random_slope"] = _fit_block(
            "M3b_random_slope", rs_formula, reml=False,
            re_formula="~module_num", display_formula=rs_display,
        )
        block_specs["M3b_random_slope"] = (rs_formula, "~module_num")

    # LRT between successively nested blocks (same random-effects
    # structure, growing fixed effects) -- 2*(llf_bigger - llf_smaller),
    # df = difference in number of fixed-effect params.
    ordered = [n for n in ("M0_null", "M1_module", "M2_within", "M3_full") if n in result["blocks"]]
    for a, b in zip(ordered, ordered[1:]):
        ba, bb = result["blocks"][a], result["blocks"][b]
        if ba.get("llf") is None or bb.get("llf") is None:
            continue
        lr_stat = 2 * (bb["llf"] - ba["llf"])
        df_diff = len(bb["params"]) - len(ba["params"])
        if df_diff <= 0:
            continue
        p_value = float(stats.chi2.sf(lr_stat, df_diff))
        result["lrt"].append({
            "from_block": a, "to_block": b,
            "lr_stat": round(lr_stat, 4), "df": df_diff,
            "p_value": round(p_value, 6),
        })

    valid_blocks = {k: v for k, v in result["blocks"].items() if v.get("aic") is not None}
    if valid_blocks:
        best_aic_name = min(valid_blocks, key=lambda k: valid_blocks[k]["aic"])
        best_bic_name = min(valid_blocks, key=lambda k: valid_blocks[k]["bic"])
        result["best_block_by_aic"] = best_aic_name
        result["best_block_by_bic"] = best_bic_name

        # Refit whichever block(s) are selected as "best" with reml=True
        # for reporting-quality SEs -- both BIC's and AIC's picks, since
        # both are now shown to the researcher as primary output, not
        # just BIC's. Skipped (falls back to the ML fit already in
        # `blocks`) only if the model's own (formula, re_formula) spec
        # somehow wasn't tracked.
        for best_name in {best_bic_name, best_aic_name}:
            spec = block_specs.get(best_name)
            if spec is None:
                continue
            formula, re_formula = spec
            reporting = _fit_block(
                best_name + "_reml", formula, reml=True,
                re_formula=re_formula,
                display_formula=result["blocks"][best_name]["formula"],
            )
            result["blocks"][best_name + "_reml"] = reporting

    # VIF on the richest converged fixed-effects design (M3_full or
    # whichever full-predictor block converged), intercept excluded.
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_cols = [f"{c}_within" for c in within_cols] + [f"{c}_between" for c in between_cols] + ["module_num"]
        vif_cols = [c for c in vif_cols if c in df.columns]
        if len(vif_cols) >= 2:
            X = df[vif_cols].assign(const=1.0)
            result["vif"] = {
                c: round(float(variance_inflation_factor(X.values, i)), 3)
                for i, c in enumerate(X.columns) if c != "const"
            }
    except Exception:
        pass

    return result


# =========================================================================
# Phase 3 -- Repeated-measures correlation + FDR
# =========================================================================

def r_effect_label(r: Optional[float]) -> str:
    """
    Cohen's (1988) conventional benchmarks for a correlation
    coefficient's magnitude -- |r| < .10 negligible, .10-.30 small,
    .30-.50 medium, >=.50 large. Cohen, J. (1988). Statistical Power
    Analysis for the Behavioral Sciences (2nd ed.). Hillsdale, NJ:
    Lawrence Erlbaum Associates. Cohen himself, and the field since,
    caution these are rough, domain-independent benchmarks of last
    resort, not a substitute for context-specific judgment -- surfaced
    as a label alongside the raw r, not in place of it.
    """
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "—"
    ar = abs(r)
    if ar >= 0.50:
        return "Large (≥.50)"
    if ar >= 0.30:
        return "Medium (≥.30)"
    if ar >= 0.10:
        return "Small (≥.10)"
    return "Negligible (<.10)"


def run_repeated_measures_correlations(
    long_df: pd.DataFrame,
    outcome_col: str,
    predictor_cols: List[str],
    id_col: str = "user_id",
    alpha: float = DEFAULT_ALPHA,
    min_modules_per_student: int = 3,
) -> dict:
    """
    pg.rm_corr() per predictor vs. outcome (Bakdash & Marusich 2017),
    then Benjamini-Hochberg FDR correction across the set (Benjamini &
    Hochberg 1995; see also Curran-Everett 2000 for FDR-over-Bonferroni
    rationale in exploratory multiple-testing -- a physiology-methods
    paper, cited only as supporting rationale, not as an
    education-specific source).

    Returns
    -------
    dict: results (list of {predictor, r, dof, p_unc, ci95, power,
        n_subjects, effect_size_label, error}), fdr (same rows +
        p_fdr, reject_fdr), alpha, error
    """
    result: Dict[str, Any] = {"results": [], "fdr": [], "alpha": alpha, "error": None}

    try:
        import pingouin as pg
    except ImportError:
        result["error"] = "pingouin not installed -- repeated-measures correlation unavailable."
        return result

    counts = long_df.groupby(id_col)["module_id"].nunique() if "module_id" in long_df.columns else long_df.groupby(id_col).size()
    eligible_ids = counts[counts >= min_modules_per_student].index
    df = long_df[long_df[id_col].isin(eligible_ids)]

    rows = []
    for predictor in predictor_cols:
        sub = df[[id_col, predictor, outcome_col]].dropna()
        n_subjects = sub[id_col].nunique()
        row: Dict[str, Any] = {
            "predictor": predictor, "r": None, "dof": None, "p_unc": None,
            "ci95": None, "power": None, "n_subjects": n_subjects,
            "effect_size_label": "—", "error": None,
        }
        if n_subjects < 3 or len(sub) < 6:
            row["error"] = f"Too few complete cases (subjects={n_subjects}) for rm_corr."
            rows.append(row)
            continue
        try:
            r = pg.rm_corr(data=sub, x=predictor, y=outcome_col, subject=id_col)
            row["r"] = round(float(r["r"].iloc[0]), 4)
            row["dof"] = int(r["dof"].iloc[0])
            row["p_unc"] = round(float(r["pval"].iloc[0]), 6)
            row["ci95"] = list(r["CI95"].iloc[0])
            row["power"] = round(float(r["power"].iloc[0]), 4)
            row["effect_size_label"] = r_effect_label(row["r"])
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    result["results"] = rows

    testable = [r for r in rows if r["p_unc"] is not None]
    if testable:
        try:
            from statsmodels.stats.multitest import multipletests
            pvals = [r["p_unc"] for r in testable]
            reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
            for r, rej, p in zip(testable, reject, p_adj):
                fdr_row = dict(r)
                fdr_row["p_fdr"] = round(float(p), 6)
                fdr_row["reject_fdr"] = bool(rej)
                result["fdr"].append(fdr_row)
        except ImportError:
            result["error"] = "statsmodels not installed -- FDR correction unavailable."

    return result
