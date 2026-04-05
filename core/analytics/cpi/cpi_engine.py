# core/analytics/cpi/cpi_engine.py
"""
CPI Computation Engine
======================
Pure functions — no Streamlit, no DB access, no side effects.
All results are plain dicts or DataFrames.

CPI_quant (Quantitative Task Performance) — two methods:

  CTT (Classical Test Theory):
    cpi_quant_ctt = sum(item_score) / count(items)  [0, 1]
    Optionally weighted by Bloom's Taxonomy level.
    Reference: Crocker & Algina (1986). Introduction to Classical and
    Modern Test Theory. Holt, Rinehart and Winston.

  IRT (Item Response Theory):
    cpi_quant_irt = sigmoid(theta) = 1 / (1 + exp(-theta))  [0, 1]
    theta = EAP person ability estimate from Rasch or 2PL model.
    Reference: Baker, F. B. (1985). The Basics of Item Response Theory.
    Heinemann.

CPI_qual (Qualitative Reflection Quality):
  cpi_qual = sum(dimension_scores) / max_possible_score  [0, 1]
  LLM-as-judge using cpi_rubric.py dimensions.
  Reference: Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench
  and Chatbot Arena. arXiv. https://doi.org/10.48550/ARXIV.2306.05685

CPI+ (Combined):
  cpi_plus = w1 * cpi_quant + w2 * cpi_qual  [0, 1]
  Default: w1 = w2 = 0.5

Architecture rules:
  - Pure functions — same inputs always produce same outputs
  - No global state, no side effects
  - Validates inputs, fails fast with clear error messages
  - Uses irt_runner and cpi_rubric internally
  - call_fn (call_model from llm_clients) injected, not imported
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd

from core.analytics.cpi.cpi_rubric import (
    CPI_QUAL_DIMENSIONS,
    CPI_QUAL_MAX_PER_QUESTION,
    CPI_QUAL_SYSTEM_PROMPT,
    build_cpi_qual_prompt,
    parse_cpi_qual_response,
)


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

LOW_N_THRESHOLD = 30


# -----------------------------------------------------------------------
# Helper: sigmoid (theta → probability scale)
# -----------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Logistic sigmoid. Maps IRT logit-scale theta to [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-float(x))))


# -----------------------------------------------------------------------
# Public function 1: compute_cpi_quant_ctt
# -----------------------------------------------------------------------

def compute_cpi_quant_ctt(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    bloom_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    CTT-based CPI_quant: proportion correct per student on a binary MCQ.

    If bloom_weights is provided, it maps question_id → weight (float).
    Unweighted questions default to 1.0. Final score is
    weighted_sum / max_weighted_sum. Without weights, this equals
    simple % correct.

    Parameters
    ----------
    canonical_df  : pd.DataFrame — canonical research dataset
    instrument_key: str          — exact DB instrument_name key
    bloom_weights : dict | None  — {question_id: float}. None = uniform.

    Returns
    -------
    pd.DataFrame — columns: user_id | cpi_quant_ctt | n_items | method
    """
    mask = (
        (canonical_df["instrument_key"] == instrument_key) |
        canonical_df["instrument_key"].str.endswith("_" + instrument_key)
    )
    subset = canonical_df[mask & canonical_df["item_score"].isin([0.0, 1.0])].copy()

    if subset.empty:
        return pd.DataFrame(columns=["user_id", "cpi_quant_ctt", "n_items", "method"])

    if bloom_weights:
        subset["_weight"] = subset["question_id"].map(bloom_weights).fillna(1.0)
        agg = (
            subset.groupby("user_id")
            .apply(lambda g: pd.Series({
                "cpi_quant_ctt": (
                    (g["item_score"] * g["_weight"]).sum() /
                    g["_weight"].sum()
                ),
                "n_items": len(g),
            }))
            .reset_index()
        )
    else:
        agg = (
            subset.groupby("user_id")
            .agg(
                cpi_quant_ctt=("item_score", "mean"),
                n_items=("item_score", "count"),
            )
            .reset_index()
        )

    agg["cpi_quant_ctt"] = agg["cpi_quant_ctt"].round(4)
    agg["method"]        = "CTT" + (" (Bloom-weighted)" if bloom_weights else "")
    return agg[["user_id", "cpi_quant_ctt", "n_items", "method"]]


# -----------------------------------------------------------------------
# Public function 2: compute_cpi_quant_irt
# -----------------------------------------------------------------------

def compute_cpi_quant_irt(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    irt_model: str = "rasch",
) -> Dict[str, Any]:
    """
    IRT-based CPI_quant: sigmoid-normalized EAP theta per student.

    Normalization: cpi_quant_irt = sigmoid(theta) = 1 / (1 + exp(-theta))
    This maps the logit-scale theta to [0, 1], with theta = 0 → 0.5
    (median ability), theta = +2 → 0.88, theta = -2 → 0.12.

    Parameters
    ----------
    canonical_df  : pd.DataFrame
    instrument_key: str          — exact DB instrument_name key
    irt_model     : str          — "rasch" | "2pl"

    Returns
    -------
    dict with keys:
        model_used       str
        person_df        pd.DataFrame — user_id | theta | theta_se
                                        | cpi_quant_irt | method
        low_n_warning    bool
        n_persons        int
        n_items          int
        dropped_items    list | None
        error            str | None
    """
    result: Dict[str, Any] = {
        "model_used":    irt_model,
        "person_df":     pd.DataFrame(),
        "low_n_warning": False,
        "n_persons":     0,
        "n_items":       0,
        "dropped_items": None,
        "error":         None,
    }

    try:
        from core.analytics.irt.irt_runner import (
            build_binary_response_matrix,
            run_rasch_model,
            run_2pl_model,
            MIN_N_2PL,
            MIN_N_WARN,
        )

        matrix, item_ids = build_binary_response_matrix(
            canonical_df, instrument_key
        )

        n = len(matrix)
        result["n_persons"]     = n
        result["n_items"]       = len(item_ids)
        result["low_n_warning"] = n < MIN_N_WARN

        if irt_model == "2pl":
            irt_result = run_2pl_model(matrix, item_ids)
        else:
            irt_result = run_rasch_model(matrix, item_ids)

        if irt_result.get("error"):
            result["error"] = irt_result["error"]
            return result

        result["dropped_items"] = irt_result.get("dropped_items")

        person_params = irt_result.get("person_params", pd.DataFrame())
        if person_params.empty or "theta" not in person_params.columns:
            result["error"] = "IRT model returned no person parameters."
            return result

        person_df = person_params[
            ["user_id", "theta"] +
            (["theta_se"] if "theta_se" in person_params.columns else [])
        ].copy()

        # Normalize theta → [0, 1] via sigmoid
        person_df["cpi_quant_irt"] = person_df["theta"].apply(_sigmoid).round(4)
        person_df["method"]        = irt_model.upper()

        result["person_df"] = person_df

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Public function 3: get_reflection_texts
# -----------------------------------------------------------------------

def get_reflection_texts(
    canonical_df: pd.DataFrame,
    module_id: Optional[str] = None,
    question_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract reflection response texts from canonical_df.

    Reflections are stored with:
        instrument_key containing "module_reflections"
        item_score = NaN (no scoring)
        response_value = raw student text

    Parameters
    ----------
    canonical_df : pd.DataFrame
    module_id    : str | None — filter to this module (e.g. "module_1").
                               None = all modules.
    question_ids : list | None — filter to these question IDs.
                                None = all questions.

    Returns
    -------
    pd.DataFrame — columns: user_id | module_id | question_id | text
        Empty rows (blank responses) are excluded.
    """
    mask = canonical_df["instrument_key"].str.contains(
        "module_reflections", regex=False
    )
    subset = canonical_df[mask].copy()

    if module_id:
        subset = subset[subset["module_id"] == module_id]

    if question_ids:
        subset = subset[subset["question_id"].isin(question_ids)]

    # response_value holds the raw text
    subset = subset[subset["response_value"].notna()].copy()
    subset = subset[subset["response_value"].astype(str).str.strip() != ""].copy()

    return (
        subset[["user_id", "module_id", "question_id", "response_value"]]
        .rename(columns={"response_value": "text"})
        .drop_duplicates(subset=["user_id", "question_id"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------
# Public function 4: score_reflection_llm
# -----------------------------------------------------------------------

def score_reflection_llm(
    participant_id: str,
    question_id: str,
    text: str,
    module_id: str,
    model: str,
    call_fn: Callable,
    module_title: str = "",
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> Dict[str, Any]:
    """
    Score a single reflection response using the LLM-as-judge rubric.

    Parameters
    ----------
    participant_id : str
    question_id    : str    — "conceptual_change" | "module_takeaway"
    text           : str    — raw student response
    module_id      : str
    model          : str    — "claude" | "gemini" | "gpt" | "groq"
    call_fn        : callable — call_model function from llm_clients
    module_title   : str    — optional, improves context
    temperature    : float  — 0.0 for deterministic scoring
    max_tokens     : int

    Returns
    -------
    dict with keys:
        participant_id, question_id, module_id,
        scores       (dict: dim_id -> int, 1-4),
        justification (str),
        cpi_qual_q   (float [0,1]: this question's contribution),
        raw_prompt   (str),
        raw_response (str | None),
        model        (str),
        tokens_used  (int),
        error        (str | None)
    """
    result: Dict[str, Any] = {
        "participant_id": participant_id,
        "question_id":    question_id,
        "module_id":      module_id,
        "scores":         {dim["id"]: 1 for dim in CPI_QUAL_DIMENSIONS},
        "justification":  "",
        "cpi_qual_q":     0.0,
        "raw_prompt":     "",
        "raw_response":   None,
        "model":          model,
        "tokens_used":    0,
        "error":          None,
    }

    if not text or not text.strip():
        result["error"] = "Empty reflection text."
        return result

    prompt = build_cpi_qual_prompt(
        participant_id=participant_id,
        question_id=question_id,
        text=text,
        module_id=module_id,
        module_title=module_title,
    )
    result["raw_prompt"] = prompt

    llm_response = call_fn(
        model=model,
        prompt=prompt,
        system=CPI_QUAL_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    result["tokens_used"] = llm_response.get("tokens_used", 0)

    if llm_response.get("error"):
        result["error"] = llm_response["error"]
        return result

    raw_text             = llm_response.get("text", "")
    result["raw_response"] = raw_text

    parsed = parse_cpi_qual_response(raw_text, participant_id, question_id)
    if parsed.get("parse_error"):
        result["error"] = parsed["parse_error"]
        return result

    result["scores"]      = parsed["scores"]
    result["justification"] = parsed["justification"]

    # CPI_qual contribution for this question: sum / max
    total     = sum(parsed["scores"].values())
    result["cpi_qual_q"] = round(total / CPI_QUAL_MAX_PER_QUESTION, 4)

    return result


# -----------------------------------------------------------------------
# Public function 5: compute_cpi_qual_from_scores
# -----------------------------------------------------------------------

def compute_cpi_qual_from_scores(
    scores_list: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Aggregate per-question LLM dimension scores into CPI_qual per participant.

    CPI_qual = sum(all dimension scores across all questions)
               / (max_per_question * n_questions_answered)

    Participants with no scored questions receive NaN.

    Parameters
    ----------
    scores_list : list of dicts from score_reflection_llm()

    Returns
    -------
    pd.DataFrame — columns: user_id | cpi_qual | n_questions_scored
    """
    if not scores_list:
        return pd.DataFrame(columns=["user_id", "cpi_qual", "n_questions_scored"])

    records = []
    for s in scores_list:
        if s.get("error"):
            continue
        dim_sum = sum(s["scores"].values())
        records.append({
            "user_id":    s["participant_id"],
            "dim_sum":    dim_sum,
            "max_for_q":  CPI_QUAL_MAX_PER_QUESTION,
        })

    if not records:
        return pd.DataFrame(columns=["user_id", "cpi_qual", "n_questions_scored"])

    df = pd.DataFrame(records)
    agg = (
        df.groupby("user_id")
        .agg(
            total_dim_sum=("dim_sum", "sum"),
            total_max=("max_for_q", "sum"),
            n_questions_scored=("dim_sum", "count"),
        )
        .reset_index()
    )
    agg["cpi_qual"] = (agg["total_dim_sum"] / agg["total_max"]).round(4)
    return agg[["user_id", "cpi_qual", "n_questions_scored"]]


# -----------------------------------------------------------------------
# Public function 6: compute_cpi_combined
# -----------------------------------------------------------------------

def compute_cpi_combined(
    cpi_quant_df: pd.DataFrame,
    cpi_qual_df: pd.DataFrame,
    quant_col: str = "cpi_quant_ctt",
    w1: float = 0.5,
    w2: float = 0.5,
) -> pd.DataFrame:
    """
    Compute CPI+ = w1 * CPI_quant + w2 * CPI_qual per participant.

    Participants with only one component receive that component's
    weighted contribution (i.e. missing values are treated as absent,
    not zero, and the result is NaN unless both are present).

    Parameters
    ----------
    cpi_quant_df : pd.DataFrame — must contain user_id and quant_col
    cpi_qual_df  : pd.DataFrame — must contain user_id and cpi_qual
    quant_col    : str          — column to use for CPI_quant
                                  ("cpi_quant_ctt" or "cpi_quant_irt")
    w1           : float        — weight for CPI_quant (default 0.5)
    w2           : float        — weight for CPI_qual  (default 0.5)

    Returns
    -------
    pd.DataFrame — columns:
        user_id | cpi_quant | cpi_qual | cpi_plus | w1 | w2
    """
    assert abs(w1 + w2 - 1.0) < 1e-6, "w1 + w2 must equal 1.0"

    left  = cpi_quant_df[["user_id", quant_col]].rename(
        columns={quant_col: "cpi_quant"}
    )
    right = cpi_qual_df[["user_id", "cpi_qual"]]

    merged = left.merge(right, on="user_id", how="outer")
    merged["cpi_plus"] = (
        w1 * merged["cpi_quant"] + w2 * merged["cpi_qual"]
    ).round(4)
    merged["w1"] = w1
    merged["w2"] = w2

    return merged[["user_id", "cpi_quant", "cpi_qual", "cpi_plus", "w1", "w2"]]


# -----------------------------------------------------------------------
# Public function 7: cpi_summary_stats
# -----------------------------------------------------------------------

def cpi_summary_stats(cpi_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute cohort-level summary statistics for CPI display.

    Parameters
    ----------
    cpi_df : pd.DataFrame — output of compute_cpi_combined()

    Returns
    -------
    dict: {column: {mean, median, sd, min, max, n}} for each CPI column
    """
    cols = ["cpi_quant", "cpi_qual", "cpi_plus"]
    stats = {}
    for col in cols:
        if col not in cpi_df.columns:
            continue
        series = cpi_df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean":   round(float(series.mean()),   3),
            "median": round(float(series.median()), 3),
            "sd":     round(float(series.std(ddof=1)), 3) if len(series) > 1 else 0.0,
            "min":    round(float(series.min()),    3),
            "max":    round(float(series.max()),    3),
            "n":      int(series.count()),
        }
    return stats

# ═══════════════════════════════════════════════════════════════════════
# THREE-COMPONENT CPI+ MODEL  (Pellegrino & Hilton 2012; Hake 1998)
# CPI+ = w1·CPI_outcome + w2·CPI_process + w3·CPI_qual
# ═══════════════════════════════════════════════════════════════════════

# ── Default weights (equal — justified for small N, Dawes 1979) ───────────────
DEFAULT_W       = dict(w_outcome=1/3, w_process=1/3, w_qual=1/3)
DEFAULT_OUTCOME = dict(alpha=1/3, beta=1/3, gamma=1/3)   # MCQ, g_misc, g_AICI
DEFAULT_PROCESS = dict(delta=0.5, epsilon=0.5)            # SCCCES, SIMS


def hake_gain(pre: float, post: float, max_score: float) -> float:
    """
    Normalised gain <g> = (post - pre) / (max - pre).
    Returns NaN if pre == max (ceiling). Clipped to [-1, 1].
    Reference: Hake (1998) Am J Phys 66(1).
    """
    if pre is None or post is None or max_score is None:
        return float("nan")
    if max_score <= pre:
        return float("nan")
    g = (post - pre) / (max_score - pre)
    return float(np.clip(g, -1.0, 1.0))


def compute_cpi_outcome(
    canonical_df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute CPI_outcome per student.
    CPI_outcome = α·CPI_MCQ + β·g_misconceptions + γ·g_AICI

    Returns DataFrame: user_id, cpi_mcq, g_misconceptions, g_aici,
                       cpi_outcome, n_modules_mcq
    """
    w = {**DEFAULT_OUTCOME, **(weights or {})}

    from core.analytics.descriptive.score_aggregator import compute_assessment_scores

    asc = compute_assessment_scores(canonical_df)
    if asc.empty:
        return pd.DataFrame(columns=["user_id", "cpi_mcq", "g_misconceptions",
                                     "g_aici", "cpi_outcome"])

    results = []
    for uid in asc["user_id"].unique():
        udf = asc[asc["user_id"] == uid]

        mcq_rows = udf[udf["instrument_key"].str.contains(
            "content_mcq_assessment", na=False
        )]
        cpi_mcq = float(mcq_rows["pct_correct"].mean() / 100) \
            if not mcq_rows.empty else float("nan")

        pre_misc  = udf[udf["instrument_key"].str.contains("pre_ai_misconceptions",  na=False)]
        post_misc = udf[udf["instrument_key"].str.contains("post_ai_misconceptions", na=False)]
        g_misc = float("nan")
        if not pre_misc.empty and not post_misc.empty:
            g_misc = hake_gain(pre_misc["pct_correct"].mean(),
                               post_misc["pct_correct"].mean(), 100.0)

        pre_aici  = udf[udf["instrument_key"].str.contains("pre_aici",  na=False)]
        post_aici = udf[udf["instrument_key"].str.contains("post_aici", na=False)]
        g_aici = float("nan")
        if not pre_aici.empty and not post_aici.empty:
            g_aici = hake_gain(pre_aici["pct_correct"].mean(),
                               post_aici["pct_correct"].mean(), 100.0)

        components = {
            "cpi_mcq":          (cpi_mcq, w["alpha"]),
            "g_misconceptions": (g_misc,  w["beta"]),
            "g_aici":           (g_aici,  w["gamma"]),
        }
        valid = [(v, wt) for v, wt in components.values() if not np.isnan(v)]
        if valid:
            total_w = sum(wt for _, wt in valid)
            cpi_out = sum(v * wt for v, wt in valid) / total_w
        else:
            cpi_out = float("nan")

        results.append({
            "user_id":          uid,
            "cpi_mcq":          round(cpi_mcq, 4) if not np.isnan(cpi_mcq) else None,
            "g_misconceptions": round(g_misc,  4) if not np.isnan(g_misc)  else None,
            "g_aici":           round(g_aici,  4) if not np.isnan(g_aici)  else None,
            "cpi_outcome":      round(cpi_out, 4) if not np.isnan(cpi_out) else None,
            "n_modules_mcq":    int(len(mcq_rows)),
        })
    return pd.DataFrame(results)


def compute_cpi_process(
    canonical_df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute CPI_process per student from SCCCES and SIMS survey means.
    Normalises Likert 1-4 → 0-1. Reverse-codes external_regulation
    and amotivation before aggregation.

    Returns DataFrame: user_id, sccces_mean, sims_mean, cpi_process
    """
    w = {**DEFAULT_PROCESS, **(weights or {})}

    from core.analytics.descriptive.score_aggregator import compute_construct_means

    cm = compute_construct_means(canonical_df)
    if cm.empty:
        return pd.DataFrame(columns=["user_id", "sccces_mean", "sims_mean", "cpi_process"])

    results = []
    for uid in cm["user_id"].unique():
        udf = cm[cm["user_id"] == uid]

        sccces_rows = udf[udf["instrument_key"].apply(lambda k: "sccces" in k.lower())]
        sccces_mean = float("nan")
        if not sccces_rows.empty:
            sccces_mean = (sccces_rows["mean_score"].mean() - 1) / 3

        sims_rows = udf[udf["instrument_key"].apply(lambda k: "sims" in k.lower())]
        sims_mean = float("nan")
        if not sims_rows.empty:
            reverse = ["external_regulation", "amotivation"]
            sims_rows = sims_rows.copy()
            sims_rows["adj"] = sims_rows.apply(
                lambda r: ((5 - r["mean_score"]) - 1) / 3
                if r.get("construct", "") in reverse
                else (r["mean_score"] - 1) / 3,
                axis=1,
            )
            sims_mean = float(sims_rows["adj"].mean())

        components = [(sccces_mean, w["delta"]), (sims_mean, w["epsilon"])]
        valid = [(v, wt) for v, wt in components if not np.isnan(v)]
        cpi_proc = (sum(v * wt for v, wt in valid) / sum(wt for _, wt in valid)
                    if valid else float("nan"))

        results.append({
            "user_id":     uid,
            "sccces_mean": round(sccces_mean, 4) if not np.isnan(sccces_mean) else None,
            "sims_mean":   round(sims_mean,   4) if not np.isnan(sims_mean)   else None,
            "cpi_process": round(cpi_proc,    4) if not np.isnan(cpi_proc)    else None,
        })
    return pd.DataFrame(results)


def compute_cpi_qual_dta(
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute CPI_qual per student from DTA evidence density.
    Alias: use this when CPI_qual comes from DTA (not LLM reflection scoring).

    Returns DataFrame: user_id, evidence_total, constructs_found, cpi_qual
    """
    try:
        from core.analytics.llm.dta_pipeline import load_dta_results, list_dta_runs
    except ImportError:
        return pd.DataFrame(columns=["user_id", "evidence_total",
                                     "constructs_found", "cpi_qual"])

    if run_id is None:
        runs = [r for r in list_dta_runs()
                if r.get("construct_groups", "[]") != '["learning_objectives"]']
        if not runs:
            return pd.DataFrame(columns=["user_id", "evidence_total",
                                         "constructs_found", "cpi_qual"])
        run_id = runs[0]["run_id"]

    df = load_dta_results(run_id)
    if df.empty:
        return pd.DataFrame(columns=["user_id", "evidence_total",
                                     "constructs_found", "cpi_qual"])

    n_constructs = df["construct_name"].nunique()
    results = []
    for uid, udf in df.groupby("participant_id"):
        ev_total = int(udf["evidence_count"].sum())
        found    = int((udf["evidence_count"] > 0).sum())
        if n_constructs > 0 and found > 0:
            mean_ev = udf[udf["evidence_count"] > 0]["evidence_count"].mean()
            density = (found / n_constructs) * min(mean_ev / 5, 1.0)
        else:
            density = 0.0
        results.append({
            "user_id":          uid,
            "evidence_total":   ev_total,
            "constructs_found": found,
            "cpi_qual":         round(float(density), 4),
        })
    return pd.DataFrame(results)


def compute_cpi_plus(
    canonical_df: pd.DataFrame,
    dta_run_id: Optional[str] = None,
    top_weights: dict = None,
    outcome_weights: dict = None,
    process_weights: dict = None,
) -> pd.DataFrame:
    """
    Compute three-component CPI+ per student.
    CPI+ = w1·CPI_outcome + w2·CPI_process + w3·CPI_qual

    CPI_qual is sourced from DTA evidence density if a DTA run exists,
    otherwise returns NaN for that component.
    """
    w = {**DEFAULT_W, **(top_weights or {})}

    df_out  = compute_cpi_outcome(canonical_df, outcome_weights)
    df_proc = compute_cpi_process(canonical_df, process_weights)
    df_qual = compute_cpi_qual_dta(dta_run_id)

    merged = df_out.copy()
    if not df_proc.empty:
        merged = merged.merge(df_proc, on="user_id", how="outer")
    else:
        merged["sccces_mean"] = float("nan")
        merged["sims_mean"]   = float("nan")
        merged["cpi_process"] = float("nan")

    if not df_qual.empty:
        merged = merged.merge(df_qual, on="user_id", how="outer")
    else:
        merged["cpi_qual"] = float("nan")

    def _cpi_plus(row):
        components = [
            (row.get("cpi_outcome"), w["w_outcome"]),
            (row.get("cpi_process"), w["w_process"]),
            (row.get("cpi_qual"),    w["w_qual"]),
        ]
        valid = [(v, wt) for v, wt in components
                 if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not valid:
            return None
        total_w = sum(wt for _, wt in valid)
        return round(sum(v * wt for v, wt in valid) / total_w, 4)

    merged["cpi_plus"] = merged.apply(_cpi_plus, axis=1)
    return merged


def cpi_band(cpi: Optional[float]) -> str:
    """Return descriptive band label for a CPI+ score."""
    if cpi is None or (isinstance(cpi, float) and np.isnan(cpi)):
        return "Insufficient data"
    if cpi >= 0.75:
        return "🟢 High (≥0.75)"
    if cpi >= 0.50:
        return "🟡 Moderate (0.50–0.74)"
    if cpi >= 0.25:
        return "🟠 Developing (0.25–0.49)"
    return "🔴 Emerging (<0.25)"
