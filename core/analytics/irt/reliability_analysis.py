"""
core/analytics/irt/reliability_analysis.py
==========================================
Construct Reliability (CR) and supplementary reliability measures
for Basics4AI questionnaires.

PRIMARY METHOD:
  Construct Reliability (CR) — Rosli et al. (2021)
    CR = (Σλᵢ)² / [(Σλᵢ)² + Σ(1−λᵢ²)]
    where λᵢ = standardised factor loading for item i.

  Conversion from IRT discrimination (aᵢ) to factor loading:
    λᵢ = aᵢ / √(1 + aᵢ²)    [UIRT–FA equivalence, McDonald 1999]

  For Rasch (1PL): all aᵢ = 1.0 by model constraint → λᵢ = 0.707

SECONDARY MEASURES (shown alongside CR):
  Cronbach's α  — Cronbach (1951); noted as baseline / legacy
  McDonald's ω  — McDonald (1999); Dunn et al. (2014)
                  ω = λ²_h / (λ²_h + Σδᵢ)  where λ_h = Σλᵢ (hierarchical)
  Average Variance Extracted (AVE) — Fornell & Larcker (1981)
                  AVE = Σλᵢ² / n

INTERPRETATION THRESHOLDS (Rosli et al. 2021; Hair et al. 2014):
  CR   ≥ 0.70 = acceptable reliability
  CR   ≥ 0.80 = good reliability
  AVE  ≥ 0.50 = adequate convergent validity
  α    ≥ 0.70 = acceptable (Hair et al.)

CITATIONS:
  Rosli, M. S., Saleh, N. S., Alshammari, S. H., Ibrahim, M. M.,
    Atan, A. S., & Atan, N. A. (2021). Improving Questionnaire
    Reliability using Construct Reliability for Researches in
    Educational Technology. iJIM, 15(04), 109.
    https://doi.org/10.3991/ijim.v15i04.20199

  Dunn, T. J., Baguley, T., & Brunsden, V. (2014). From alpha to
    omega. British Journal of Psychology, 105(3), 399–412.

  Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation
    models. Journal of Marketing Research, 18(1), 39–50.

  Hair, J. F., Ringle, C. M., & Sarstedt, M. (2014). PLS-SEM.
    Journal of Marketing Theory and Practice, 19(2), 139–152.

  Libasin, Z., Ahmad, N., & Umar, N. (2025). Beyond Cronbach's Alpha:
    Rethinking Reliability in the Age of Digital Learning.
    SIG e-Learning@CS. e-ISBN 978-629-98755-7-4.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# ── Thresholds ────────────────────────────────────────────────────────────────
CR_ACCEPTABLE  = 0.70
CR_GOOD        = 0.80
AVE_ACCEPTABLE = 0.50
ALPHA_ACCEPTABLE = 0.70


# ── IRT → factor loading conversion ─────────────────────────────────────────

def irt_discrimination_to_loading(a: float) -> float:
    """
    Convert IRT 2PL/GRM discrimination parameter (a) to standardised
    factor loading (λ) using the UIRT–FA equivalence:
        λ = a / √(1 + a²)
    Reference: McDonald (1999); Reeve & Fayers (2005).
    """
    return float(a / np.sqrt(1.0 + a ** 2))


def rasch_loading() -> float:
    """
    In the Rasch (1PL) model all discriminations are constrained to 1.0.
    λ = 1 / √(1 + 1²) = 1/√2 ≈ 0.707
    """
    return irt_discrimination_to_loading(1.0)


# ── Core reliability metrics ──────────────────────────────────────────────────

def compute_cr(loadings: List[float]) -> float:
    """
    Construct Reliability (CR) — Rosli et al. (2021).

    CR = (Σλᵢ)² / [(Σλᵢ)² + Σ(1−λᵢ²)]

    Parameters
    ----------
    loadings : list of float — standardised factor loadings (0 < λ < 1)

    Returns
    -------
    float — CR value [0, 1]
    """
    lam = np.array([float(l) for l in loadings])
    sum_lam   = np.sum(lam)
    sum_error = np.sum(1.0 - lam ** 2)
    denom = sum_lam ** 2 + sum_error
    return float(sum_lam ** 2 / denom) if denom > 0 else float("nan")


def compute_ave(loadings: List[float]) -> float:
    """
    Average Variance Extracted (AVE) — Fornell & Larcker (1981).

    AVE = Σλᵢ² / n

    Assesses convergent validity: AVE ≥ 0.50 is acceptable.
    """
    lam = np.array([float(l) for l in loadings])
    return float(np.mean(lam ** 2))


def compute_omega(loadings: List[float]) -> float:
    """
    McDonald's ω (hierarchical) — McDonald (1999); Dunn et al. (2014).

    ω = (Σλᵢ)² / [(Σλᵢ)² + Σ(1−λᵢ²)]

    Note: for unidimensional scales ω = CR.
    The function is kept separate for conceptual clarity and future
    multidimensional extension.
    """
    return compute_cr(loadings)   # identical for unidimensional case


def compute_cronbach_alpha(response_matrix: pd.DataFrame) -> float:
    """
    Cronbach's α from raw item responses.

    α = (n/(n-1)) × (1 − Σs²ᵢ/s²_total)

    Included as legacy baseline; CR preferred (Rosli et al. 2021;
    Libasin et al. 2025).
    """
    n_items = response_matrix.shape[1]
    if n_items < 2:
        return float("nan")
    item_vars = response_matrix.var(axis=0, ddof=1)
    total_var = response_matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return float(np.clip(alpha, -1.0, 1.0))


# ── Interpretation helpers ────────────────────────────────────────────────────

def cr_badge(cr: float) -> str:
    if cr is None or np.isnan(cr):
        return "—"
    if cr >= CR_GOOD:
        return f"🟢 Good (≥{CR_GOOD})"
    if cr >= CR_ACCEPTABLE:
        return f"🟡 Acceptable (≥{CR_ACCEPTABLE})"
    return f"🔴 Poor (<{CR_ACCEPTABLE})"


def alpha_badge(alpha: float) -> str:
    if alpha is None or np.isnan(alpha):
        return "—"
    if alpha >= 0.90:
        return "🟢 Excellent (≥0.90)"
    if alpha >= 0.80:
        return "🟢 Good (≥0.80)"
    if alpha >= ALPHA_ACCEPTABLE:
        return f"🟡 Acceptable (≥{ALPHA_ACCEPTABLE})"
    return f"🔴 Poor (<{ALPHA_ACCEPTABLE})"


def ave_badge(ave: float) -> str:
    if ave is None or np.isnan(ave):
        return "—"
    if ave >= AVE_ACCEPTABLE:
        return f"🟢 Adequate (≥{AVE_ACCEPTABLE})"
    return f"🔴 Inadequate (<{AVE_ACCEPTABLE})"


# ── High-level entry point ────────────────────────────────────────────────────

def compute_reliability_report(
    irt_result: dict,
    response_matrix: pd.DataFrame,
    construct_name: str = "",
    model_type: str = "Rasch",
) -> dict:
    """
    Produce a reliability report for one instrument / construct.

    Parameters
    ----------
    irt_result       : dict — output of run_rasch_model / run_2pl_model / run_grm_model
    response_matrix  : pd.DataFrame — raw item responses (persons × items)
    construct_name   : str — label for display
    model_type       : str — "Rasch" | "2PL" | "GRM"

    Returns
    -------
    dict with keys:
        construct, n_items, n_persons, model_type,
        loadings,
        cr, cr_badge,
        ave, ave_badge,
        omega,
        cronbach_alpha, alpha_badge,
        error
    """
    report = {
        "construct":      construct_name,
        "n_items":        0,
        "n_persons":      0,
        "model_type":     model_type,
        "loadings":       [],
        "cr":             float("nan"),
        "cr_badge":       "—",
        "ave":            float("nan"),
        "ave_badge":      "—",
        "omega":          float("nan"),
        "cronbach_alpha": float("nan"),
        "alpha_badge":    "—",
        "error":          None,
    }

    try:
        if irt_result.get("error"):
            report["error"] = irt_result["error"]
            return report

        params = irt_result.get("item_params", pd.DataFrame())
        report["n_persons"] = irt_result.get("n_persons", 0)

        if params.empty:
            report["error"] = "No item parameters in IRT result."
            return report

        report["n_items"] = len(params)

        # ── Extract factor loadings ───────────────────────────────────────
        if "a" in params.columns:
            # 2PL or GRM — use discrimination parameter
            loadings = [irt_discrimination_to_loading(a)
                        for a in params["a"].astype(float)]
        else:
            # Rasch — all discriminations = 1
            loadings = [rasch_loading()] * len(params)

        report["loadings"] = [round(l, 4) for l in loadings]

        # ── Reliability metrics ───────────────────────────────────────────
        cr  = compute_cr(loadings)
        ave = compute_ave(loadings)
        om  = compute_omega(loadings)

        report["cr"]     = round(cr,  4) if not np.isnan(cr)  else float("nan")
        report["ave"]    = round(ave, 4) if not np.isnan(ave) else float("nan")
        report["omega"]  = round(om,  4) if not np.isnan(om)  else float("nan")
        report["cr_badge"]  = cr_badge(cr)
        report["ave_badge"] = ave_badge(ave)

        # ── Cronbach's α from raw responses ──────────────────────────────
        if not response_matrix.empty:
            alpha = compute_cronbach_alpha(
                response_matrix.select_dtypes(include="number")
            )
            report["cronbach_alpha"] = round(alpha, 4) if not np.isnan(alpha) else float("nan")
            report["alpha_badge"]    = alpha_badge(alpha)

    except Exception as e:
        report["error"] = str(e)

    return report


# ── Construct-level summary table ─────────────────────────────────────────────

def build_reliability_summary_df(reports: List[dict]) -> pd.DataFrame:
    """
    Convert a list of reliability reports into a display DataFrame.
    """
    rows = []
    for r in reports:
        rows.append({
            "Construct":        r.get("construct", "—"),
            "N Items":          r.get("n_items", "—"),
            "N Students":       r.get("n_persons", "—"),
            "Model":            r.get("model_type", "—"),
            "CR":               f"{r['cr']:.3f}" if not (isinstance(r['cr'], float) and np.isnan(r['cr'])) else "—",
            "CR status":        r.get("cr_badge", "—"),
            "AVE":              f"{r['ave']:.3f}" if not (isinstance(r['ave'], float) and np.isnan(r['ave'])) else "—",
            "AVE status":       r.get("ave_badge", "—"),
            "Cronbach α":       f"{r['cronbach_alpha']:.3f}" if not (isinstance(r['cronbach_alpha'], float) and np.isnan(r['cronbach_alpha'])) else "—",
            "α status":         r.get("alpha_badge", "—"),
        })
    return pd.DataFrame(rows)
