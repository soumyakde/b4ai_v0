# core/analytics/irt/irt_runner.py
"""
IRT Analysis Engine
===================
Item Response Theory analysis using the pure-Python girth library.
No R, no rpy2, no threading issues.

Dependencies: girth (pip install girth), numpy, scipy, pandas

Supported models:
    Rasch (1PL)  — binary items, item difficulty only
    2PL          — binary items, difficulty + discrimination (n>=50)
    GRM          — polytomous Likert items (n>=50)

Public API:
-----------
build_binary_response_matrix(canonical_df, instrument_key)
build_likert_response_matrix(canonical_df, instrument_key, construct)
run_rasch_model(response_matrix, item_ids)
run_2pl_model(response_matrix, item_ids)
run_grm_model(response_matrix, item_ids)
get_icc_data(model_result, item_id=None, n_points=100)
get_wright_map_data(model_result)

Constants:
    MIN_N_2PL  = 50
    MIN_N_GRM  = 50
    MIN_N_WARN = 100
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------
# girth imports
# -----------------------------------------------------------------------
try:
    from girth import (
        rasch_mml,
        twopl_mml,
        grm_mml,
        ability_eap,
    )
    _GIRTH_AVAILABLE = True
except ImportError:
    _GIRTH_AVAILABLE = False

# Keep rpy2 flag for backward compat — always False now
_RPY2_AVAILABLE = False

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
MIN_N_2PL  = 50
MIN_N_GRM  = 50
MIN_N_WARN = 100


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _check_girth():
    if not _GIRTH_AVAILABLE:
        raise RuntimeError(
            "girth is not installed. Run: pip install girth"
        )


def _natural_sort_key(col):
    import re
    parts = re.split(r"(\d+)", str(col))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _compute_aic_bic(log_likelihood: float, n_params: int, n: int) -> Dict:
    """Compute AIC and BIC from log-likelihood."""
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n)
    return {"aic": round(float(aic), 2), "bic": round(float(bic), 2)}


def _rasch_log_likelihood(
    difficulties: np.ndarray,
    thetas: np.ndarray,
    response_matrix: np.ndarray,
) -> float:
    """Approximate log-likelihood for Rasch model."""
    ll = 0.0
    n_persons, n_items = response_matrix.shape
    for i in range(n_persons):
        for j in range(n_items):
            if np.isnan(response_matrix[i, j]):
                continue
            p = 1 / (1 + np.exp(-(thetas[i] - difficulties[j])))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll += response_matrix[i, j] * np.log(p) + \
                  (1 - response_matrix[i, j]) * np.log(1 - p)
    return ll


# -----------------------------------------------------------------------
# Response matrix builders
# -----------------------------------------------------------------------

def build_binary_response_matrix(
    canonical_df: pd.DataFrame,
    instrument_key: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a binary person x item response matrix for IRT.

    Parameters
    ----------
    canonical_df : pd.DataFrame
    instrument_key : str
        DB instrument_name (exact or suffix match).

    Returns
    -------
    response_matrix : pd.DataFrame
        Rows=users, columns=item IDs. Values: 1.0/0.0/NaN.
    item_ids : list of str
    """
    mask = (
        (canonical_df["instrument_key"] == instrument_key) |
        canonical_df["instrument_key"].str.endswith("_" + instrument_key)
    )
    subset = canonical_df[mask].copy()

    if subset.empty:
        raise ValueError(
            f"No data found for instrument '{instrument_key}'."
        )

    valid = subset[subset["item_score"].isin([0.0, 1.0])].copy()
    if valid.empty:
        raise ValueError(
            f"No binary-scored items for '{instrument_key}'. "
            f"Ensure scoring has been applied via DatasetBuilder."
        )

    matrix = valid.pivot_table(
        index="user_id",
        columns="question_id",
        values="item_score",
        aggfunc="first",
    )
    sorted_cols = sorted(matrix.columns, key=_natural_sort_key)
    matrix = matrix[sorted_cols]
    return matrix, list(matrix.columns)


def build_likert_response_matrix(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    construct: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a polytomous person x item response matrix for GRM.

    Values: integers 1-4 (Likert).
    """
    mask = (
        (canonical_df["instrument_key"] == instrument_key) |
        canonical_df["instrument_key"].str.endswith("_" + instrument_key)
    )
    subset = canonical_df[
        mask & (canonical_df["construct"] == construct)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No data for instrument='{instrument_key}', "
            f"construct='{construct}'."
        )

    valid = subset[subset["item_score"].notna()].copy()
    valid["item_score"] = valid["item_score"].round().astype(int)

    matrix = valid.pivot_table(
        index="user_id",
        columns="question_id",
        values="item_score",
        aggfunc="mean",
    ).round()

    sorted_cols = sorted(matrix.columns, key=_natural_sort_key)
    matrix = matrix[sorted_cols]
    return matrix.astype(float), list(matrix.columns)


# -----------------------------------------------------------------------
# Public function 1: run_rasch_model
# -----------------------------------------------------------------------

def run_rasch_model(
    response_matrix: pd.DataFrame,
    item_ids: List[str],
) -> Dict[str, Any]:
    """
    Fit a 1PL Rasch model using girth.rasch_mml.

    Returns
    -------
    dict with keys:
        model_type, n_persons, n_items, low_n_warning,
        item_params (DataFrame: item_id, b),
        person_params (DataFrame: user_id, theta, theta_se),
        aic, bic, error, _difficulties, _thetas
    """
    result: Dict[str, Any] = {
        "model_type":    "Rasch",
        "n_persons":     len(response_matrix),
        "n_items":       len(item_ids),
        "low_n_warning": len(response_matrix) < MIN_N_WARN,
        "error":         None,
    }

    try:
        _check_girth()
        user_ids = list(response_matrix.index)

        # Drop zero-variance items
        variance   = response_matrix.var()
        zero_var   = variance[variance == 0].index.tolist()
        if zero_var:
            response_matrix = response_matrix.drop(columns=zero_var)
            item_ids = [i for i in item_ids if i not in zero_var]
            result["dropped_items"] = zero_var

        if response_matrix.shape[1] < 2:
            result["error"] = (
                "Fewer than 2 items with variance. Cannot fit IRT model."
            )
            return result

        # girth expects items x persons (transposed), integer 0/1
        data = response_matrix.T.values.astype(float)
        # Replace NaN with 0 (treat as incorrect — note this for low-n)
        data = np.nan_to_num(data, nan=0.0).astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rasch_result = rasch_mml(data)

        difficulties = rasch_result["Difficulty"]
        result["_difficulties"] = difficulties

        # Item parameters table
        item_params = pd.DataFrame({
            "item_id": item_ids[:len(difficulties)],
            "b":       [round(float(d), 4) for d in difficulties],
        })
        result["item_params"] = item_params

        # Item fit — infit/outfit approximation
        result["item_fit"] = _compute_item_fit(
            data, difficulties, item_ids
        )

        # Person ability estimates via EAP
        thetas, theta_se = _estimate_thetas(data, difficulties)
        result["_thetas"] = thetas

        person_params = pd.DataFrame({
            "user_id":  user_ids[:len(thetas)],
            "theta":    [round(float(t), 4) for t in thetas],
            "theta_se": [round(float(s), 4) for s in theta_se],
        })
        result["person_params"] = person_params

        # AIC / BIC
        ll = _rasch_log_likelihood(difficulties, thetas, response_matrix.values)
        result.update(_compute_aic_bic(ll, len(difficulties), len(user_ids)))

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Public function 2: run_2pl_model
# -----------------------------------------------------------------------

def run_2pl_model(
    response_matrix: pd.DataFrame,
    item_ids: List[str],
) -> Dict[str, Any]:
    """
    Fit a 2PL IRT model using girth.twopl_mml. Requires n >= MIN_N_2PL.
    """
    n = len(response_matrix)
    result: Dict[str, Any] = {
        "model_type":    "2PL",
        "n_persons":     n,
        "n_items":       len(item_ids),
        "low_n_warning": n < MIN_N_WARN,
        "error":         None,
    }

    if n < MIN_N_2PL:
        result["error"] = (
            f"2PL model requires n >= {MIN_N_2PL} persons. "
            f"Current n = {n}. "
            f"This model will be available after the July/August cohort "
            f"(expected n ~90)."
        )
        return result

    try:
        _check_girth()
        user_ids = list(response_matrix.index)

        variance = response_matrix.var()
        zero_var = variance[variance == 0].index.tolist()
        if zero_var:
            response_matrix = response_matrix.drop(columns=zero_var)
            item_ids = [i for i in item_ids if i not in zero_var]
            result["dropped_items"] = zero_var

        data = np.nan_to_num(
            response_matrix.T.values.astype(float), nan=0.0
        ).astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tpl_result = twopl_mml(data)

        discriminations = tpl_result["Discrimination"]
        difficulties    = tpl_result["Difficulty"]
        result["_difficulties"]    = difficulties
        result["_discriminations"] = discriminations

        item_params = pd.DataFrame({
            "item_id": item_ids[:len(difficulties)],
            "a":       [round(float(a), 4) for a in discriminations],
            "b":       [round(float(d), 4) for d in difficulties],
        })
        result["item_params"] = item_params

        thetas, theta_se = _estimate_thetas(data, difficulties,
                                             discriminations)
        result["_thetas"] = thetas

        person_params = pd.DataFrame({
            "user_id":  user_ids[:len(thetas)],
            "theta":    [round(float(t), 4) for t in thetas],
            "theta_se": [round(float(s), 4) for s in theta_se],
        })
        result["person_params"] = person_params
        result["item_fit"]      = pd.DataFrame({"item_id": item_ids})

        ll = _rasch_log_likelihood(difficulties, thetas,
                                    response_matrix.values)
        result.update(_compute_aic_bic(ll, 2 * len(difficulties),
                                        len(user_ids)))

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Public function 3: run_grm_model
# -----------------------------------------------------------------------

def run_grm_model(
    response_matrix: pd.DataFrame,
    item_ids: List[str],
) -> Dict[str, Any]:
    """
    Fit a Graded Response Model using girth.grm_mml. Requires n >= MIN_N_GRM.
    """
    n = len(response_matrix)
    result: Dict[str, Any] = {
        "model_type":    "GRM",
        "n_persons":     n,
        "n_items":       len(item_ids),
        "low_n_warning": n < MIN_N_WARN,
        "error":         None,
    }

    if n < MIN_N_GRM:
        result["error"] = (
            f"GRM requires n >= {MIN_N_GRM} persons. "
            f"Current n = {n}. "
            f"This model will be available after the July/August cohort."
        )
        return result

    try:
        _check_girth()
        user_ids = list(response_matrix.index)

        # GRM expects 0-indexed categories: convert 1-4 -> 0-3
        data = np.nan_to_num(
            response_matrix.T.values.astype(float), nan=1.0
        ).astype(int) - 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grm_result = grm_mml(data)

        discriminations = grm_result["Discrimination"]
        difficulties    = grm_result["Difficulty"]

        # Build item params table
        rows = []
        for i, iid in enumerate(item_ids[:len(discriminations)]):
            row = {"item_id": iid, "a": round(float(discriminations[i]), 4)}
            if hasattr(difficulties[i], "__len__"):
                for j, d in enumerate(difficulties[i]):
                    row[f"b{j+1}"] = round(float(d), 4)
            else:
                row["b"] = round(float(difficulties[i]), 4)
            rows.append(row)

        result["item_params"]   = pd.DataFrame(rows)
        result["item_fit"]      = pd.DataFrame({"item_id": item_ids})

        thetas, theta_se = _estimate_thetas_grm(data, discriminations, difficulties)
        result["_thetas"] = thetas

        result["person_params"] = pd.DataFrame({
            "user_id":  user_ids[:len(thetas)],
            "theta":    [round(float(t), 4) for t in thetas],
            "theta_se": [round(float(s), 4) for s in theta_se],
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Internal: person ability estimation
# -----------------------------------------------------------------------

def _estimate_thetas(
    data: np.ndarray,
    difficulties: np.ndarray,
    discriminations: Optional[np.ndarray] = None,
    n_points: int = 41,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EAP (Expected A Posteriori) person ability estimates.

    Parameters
    ----------
    data : np.ndarray
        Items x Persons matrix of binary responses.
    difficulties : np.ndarray
        Item difficulty parameters.
    discriminations : np.ndarray or None
        Item discrimination parameters (2PL). None = Rasch (all 1.0).
    n_points : int
        Quadrature points for numerical integration.
    """
    if discriminations is None:
        discriminations = np.ones(len(difficulties))

    theta_grid = np.linspace(-4, 4, n_points)
    prior      = stats.norm.pdf(theta_grid)
    prior     /= prior.sum()

    n_persons = data.shape[1]
    thetas    = np.zeros(n_persons)
    theta_se  = np.zeros(n_persons)

    for p in range(n_persons):
        log_likelihood = np.zeros(n_points)
        for i in range(len(difficulties)):
            if np.isnan(data[i, p]):
                continue
            # 2PL probability
            prob = 1 / (
                1 + np.exp(-discriminations[i] * (theta_grid - difficulties[i]))
            )
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            if data[i, p] == 1:
                log_likelihood += np.log(prob)
            else:
                log_likelihood += np.log(1 - prob)

        # Normalize to get posterior
        log_likelihood -= log_likelihood.max()
        posterior = np.exp(log_likelihood) * prior
        total     = posterior.sum()

        if total > 0:
            posterior /= total
            thetas[p]   = (theta_grid * posterior).sum()
            theta_se[p] = np.sqrt(
                ((theta_grid - thetas[p]) ** 2 * posterior).sum()
            )
        else:
            thetas[p]   = 0.0
            theta_se[p] = 1.0

    return thetas, theta_se


def _estimate_thetas_grm(
    data: np.ndarray,
    discriminations: np.ndarray,
    difficulties: List[np.ndarray],
    n_points: int = 41,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EAP (Expected A Posteriori) person ability estimates for the
    Graded Response Model (polytomous items).

    Parameters
    ----------
    data : np.ndarray
        Items x Persons matrix of 0-indexed category responses
        (category 0 .. K-1).
    discriminations : np.ndarray
        Item discrimination parameters (one per item).
    difficulties : list of np.ndarray
        Per-item category-threshold parameters. Each entry is a scalar
        (2-category item) or a 1d array of K-1 thresholds for a
        K-category item; NaN padding (from ragged category counts) is
        dropped.
    n_points : int
        Quadrature points for numerical integration.
    """
    theta_grid = np.linspace(-4, 4, n_points)
    prior      = stats.norm.pdf(theta_grid)
    prior     /= prior.sum()

    n_items, n_persons = data.shape
    thetas   = np.zeros(n_persons)
    theta_se = np.zeros(n_persons)

    # Precompute each item's category-response-probability curves
    # over the theta grid via the standard cumulative-logit GRM formula:
    # P*(score >= k) = 1 / (1 + exp(-a*(theta - b_k))), P*(>=0)=1, P*(>=K)=0
    item_cat_probs = []
    for i in range(n_items):
        thr = np.atleast_1d(difficulties[i]).astype(float)
        thr = thr[~np.isnan(thr)]
        a   = discriminations[i]
        cum_above = np.vstack(
            [np.ones(n_points)]
            + [1 / (1 + np.exp(-a * (theta_grid - t))) for t in thr]
            + [np.zeros(n_points)]
        )
        cat_probs = np.clip(cum_above[:-1] - cum_above[1:], 1e-10, 1.0)
        item_cat_probs.append(cat_probs)

    for p in range(n_persons):
        log_likelihood = np.zeros(n_points)
        for i in range(n_items):
            k = int(data[i, p])
            cat_probs = item_cat_probs[i]
            if k < 0 or k >= cat_probs.shape[0]:
                continue
            log_likelihood += np.log(cat_probs[k])

        log_likelihood -= log_likelihood.max()
        posterior = np.exp(log_likelihood) * prior
        total     = posterior.sum()

        if total > 0:
            posterior /= total
            thetas[p]   = (theta_grid * posterior).sum()
            theta_se[p] = np.sqrt(
                ((theta_grid - thetas[p]) ** 2 * posterior).sum()
            )
        else:
            thetas[p]   = 0.0
            theta_se[p] = 1.0

    return thetas, theta_se


# -----------------------------------------------------------------------
# Internal: item fit statistics
# -----------------------------------------------------------------------

def _compute_item_fit(
    data: np.ndarray,
    difficulties: np.ndarray,
    item_ids: List[str],
) -> pd.DataFrame:
    """
    Compute Rasch infit and outfit mean-square statistics.
    """
    discriminations = np.ones(len(difficulties))
    thetas, _ = _estimate_thetas(data, difficulties)

    n_items, n_persons = data.shape
    rows = []

    for j in range(n_items):
        residuals_sq    = []
        weighted_res_sq = []
        weights         = []

        for p in range(n_persons):
            if np.isnan(data[j, p]):
                continue
            prob = 1 / (1 + np.exp(-(thetas[p] - difficulties[j])))
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            w    = prob * (1 - prob)
            r    = data[j, p] - prob
            residuals_sq.append(r ** 2)
            weighted_res_sq.append(w * r ** 2 / (w + 1e-10))
            weights.append(w)

        if not weights:
            rows.append({"item_id": item_ids[j], "infit": None, "outfit": None})
            continue

        w_arr = np.array(weights)
        outfit = np.mean(residuals_sq)
        infit  = np.sum(w_arr * np.array(residuals_sq)) / np.sum(w_arr)

        rows.append({
            "item_id": item_ids[j],
            "infit":   round(float(infit),  3),
            "outfit":  round(float(outfit), 3),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Public function 4: get_icc_data
# -----------------------------------------------------------------------

def get_icc_data(
    model_result: Dict[str, Any],
    item_id: Optional[str] = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Compute ICC trace line data for Plotly rendering.

    Returns
    -------
    pd.DataFrame
        Columns: theta, item_id, category, probability
    """
    if model_result.get("error"):
        return pd.DataFrame(columns=["theta", "item_id",
                                      "category", "probability"])

    params = model_result.get("item_params", pd.DataFrame())
    if params.empty:
        return pd.DataFrame(columns=["theta", "item_id",
                                      "category", "probability"])

    theta_grid = np.linspace(-4, 4, n_points)
    rows       = []

    items_to_plot = params if item_id is None else \
        params[params["item_id"] == item_id]

    model_type = model_result.get("model_type", "Rasch")

    for _, row in items_to_plot.iterrows():
        iid = row["item_id"]
        b   = float(row.get("b", row.get("b1", 0.0)))
        a   = float(row.get("a", 1.0))

        if model_type in ("Rasch", "2PL"):
            prob = 1 / (1 + np.exp(-a * (theta_grid - b)))
            for theta_val, p in zip(theta_grid, prob):
                rows.append({
                    "theta":       round(float(theta_val), 4),
                    "item_id":     iid,
                    "category":    "P(correct)",
                    "probability": round(float(p), 6),
                })

        elif model_type == "GRM":
            # Category boundary curves
            b_cols = [c for c in row.index if c.startswith("b") and c != "b"]
            thresholds = [float(row[c]) for c in sorted(b_cols)]

            # Category response functions
            n_cats = len(thresholds) + 1
            for cat in range(n_cats):
                for theta_val in theta_grid:
                    if cat == 0:
                        p = 1 - 1/(1+np.exp(-a*(theta_val - thresholds[0]))) \
                            if thresholds else 0.5
                    elif cat == n_cats - 1:
                        p = 1/(1+np.exp(-a*(theta_val - thresholds[-1])))
                    else:
                        p = 1/(1+np.exp(-a*(theta_val - thresholds[cat-1]))) \
                          - 1/(1+np.exp(-a*(theta_val - thresholds[cat])))
                    rows.append({
                        "theta":       round(float(theta_val), 4),
                        "item_id":     iid,
                        "category":    f"Cat {cat+1}",
                        "probability": round(float(max(0.0, p)), 6),
                    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Public function 5: get_wright_map_data
# -----------------------------------------------------------------------

def get_wright_map_data(
    model_result: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Extract person and item parameter data for Wright Map rendering.

    Returns
    -------
    dict:
        "persons" : DataFrame — columns: user_id, theta
        "items"   : DataFrame — columns: item_id, b
    """
    empty = {
        "persons": pd.DataFrame(columns=["user_id", "theta"]),
        "items":   pd.DataFrame(columns=["item_id", "b"]),
    }

    if model_result.get("error"):
        return empty

    persons = model_result.get("person_params", pd.DataFrame())
    items   = model_result.get("item_params",   pd.DataFrame())

    if persons.empty or items.empty:
        return empty

    persons_out = persons[["user_id", "theta"]].copy() \
        if "theta" in persons.columns else empty["persons"]

    b_col = next(
        (c for c in items.columns if c.lower() in ("b", "d", "difficulty")),
        None
    )
    items_out = items[["item_id", b_col]].rename(columns={b_col: "b"}) \
        if b_col else empty["items"]

    return {"persons": persons_out, "items": items_out}
