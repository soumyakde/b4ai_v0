"""
agreement.py
-------------------------------------------------------
Qualitative Inter-Rater Reliability Module

Supports:
- Cohen's kappa (pairwise)
- Weighted kappa (ordinal scale)
- ICC(2,1) and ICC(2,k)
- Krippendorff's alpha (ordinal)
- LLM stability checks

Works with qualitative_ratings table from storage.py

NO UI LOGIC
-------------------------------------------------------
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import sqlite3
from itertools import combinations


# =====================================================
# DATA LOADER
# =====================================================

def load_ratings_long(
    conn: sqlite3.Connection,
    module_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads long-format ratings:

    student_id | module_id | question_id | construct |
    rating | model | prompt_hash
    """

    query = """
    SELECT
        student_id,
        module_id,
        question_id,
        construct,
        rating,
        model,
        prompt_hash
    FROM qualitative_ratings
    """

    params = []

    if module_id:
        query += " WHERE module_id=?"
        params.append(module_id)

    df = pd.read_sql_query(query, conn, params=params)

    return df


# =====================================================
# HELPER: PIVOT FOR AGREEMENT
# =====================================================

def pivot_for_raters(
    df: pd.DataFrame,
    rater_column: str = "model",
) -> pd.DataFrame:
    """
    Returns matrix:

    subject_id | rater1 | rater2 | ...
    """

    df["subject"] = (
        df["student_id"].astype(str)
        + "_"
        + df["question_id"].astype(str)
        + "_"
        + df["construct"].astype(str)
    )

    pivot = df.pivot_table(
        index="subject",
        columns=rater_column,
        values="rating",
        aggfunc="first",
    )

    return pivot.dropna()


# =====================================================
# COHEN'S KAPPA
# =====================================================

def cohens_kappa(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Standard Cohen's kappa.
    """

    assert len(r1) == len(r2)

    observed = np.mean(r1 == r2)

    categories = np.union1d(r1, r2)

    p1 = [np.mean(r1 == c) for c in categories]
    p2 = [np.mean(r2 == c) for c in categories]

    expected = np.sum(np.array(p1) * np.array(p2))

    if expected == 1:
        return 1.0

    return (observed - expected) / (1 - expected)


# =====================================================
# WEIGHTED KAPPA (ORDINAL)
# =====================================================

def weighted_kappa(
    r1: np.ndarray,
    r2: np.ndarray,
    max_rating: int = 3,
) -> float:
    """
    Quadratic weighted kappa.
    """

    n = len(r1)
    categories = np.arange(0, max_rating + 1)

    O = np.zeros((len(categories), len(categories)))
    for i, j in zip(r1, r2):
        O[int(i), int(j)] += 1

    O = O / n

    w = np.zeros_like(O)
    for i in categories:
        for j in categories:
            w[i, j] = ((i - j) ** 2) / (max_rating ** 2)

    row_marginals = O.sum(axis=1)
    col_marginals = O.sum(axis=0)

    E = np.outer(row_marginals, col_marginals)

    numerator = np.sum(w * O)
    denominator = np.sum(w * E)

    if denominator == 0:
        return 1.0

    return 1 - (numerator / denominator)


# =====================================================
# ICC(2,1) and ICC(2,k)
# =====================================================

def icc_two_way_random(data: np.ndarray) -> Tuple[float, float]:
    """
    Computes ICC(2,1) and ICC(2,k).

    data shape:
    subjects x raters
    """

    n, k = data.shape

    mean_subject = np.mean(data, axis=1)
    mean_rater = np.mean(data, axis=0)
    grand_mean = np.mean(data)

    ss_subject = k * np.sum((mean_subject - grand_mean) ** 2)
    ss_rater = n * np.sum((mean_rater - grand_mean) ** 2)
    ss_error = np.sum((data - mean_subject[:, None] - mean_rater + grand_mean) ** 2)

    df_subject = n - 1
    df_rater = k - 1
    df_error = (n - 1) * (k - 1)

    ms_subject = ss_subject / df_subject
    ms_rater = ss_rater / df_rater
    ms_error = ss_error / df_error

    icc21 = (
        (ms_subject - ms_error)
        /
        (ms_subject + (k - 1) * ms_error + k * (ms_rater - ms_error) / n)
    )

    icc2k = (
        (ms_subject - ms_error)
        /
        (ms_subject + (ms_rater - ms_error) / n)
    )

    return icc21, icc2k


# =====================================================
# KRIPPENDORFF'S ALPHA (ORDINAL)
# =====================================================

def krippendorff_alpha_ordinal(data: np.ndarray) -> float:
    """
    Simplified ordinal alpha.
    data shape:
    subjects x raters
    """

    n_subjects, n_raters = data.shape

    disagreements = 0
    total = 0

    for row in data:
        valid = row[~np.isnan(row)]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                disagreements += (valid[i] - valid[j]) ** 2
                total += 1

    if total == 0:
        return 1.0

    Do = disagreements / total

    flat = data.flatten()
    flat = flat[~np.isnan(flat)]

    De = np.var(flat)

    if De == 0:
        return 1.0

    return 1 - (Do / De)


# =====================================================
# MASTER AGREEMENT REPORT
# =====================================================

def compute_agreement_report(
    conn: sqlite3.Connection,
    module_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns agreement metrics across all rater pairs.
    """

    df = load_ratings_long(conn, module_id)

    pivot = pivot_for_raters(df)

    if pivot.shape[1] < 2:
        return pd.DataFrame()

    results = []

    for r1, r2 in combinations(pivot.columns, 2):

        sub = pivot[[r1, r2]].dropna()

        if len(sub) < 5:
            continue

        arr1 = sub[r1].values
        arr2 = sub[r2].values

        kappa = cohens_kappa(arr1, arr2)
        wkappa = weighted_kappa(arr1, arr2)

        data = sub.values
        icc21, icc2k = icc_two_way_random(data)

        alpha = krippendorff_alpha_ordinal(data)

        results.append({
            "rater_1": r1,
            "rater_2": r2,
            "n_subjects": len(sub),
            "cohens_kappa": round(kappa, 4),
            "weighted_kappa": round(wkappa, 4),
            "icc_2_1": round(icc21, 4),
            "icc_2_k": round(icc2k, 4),
            "krippendorff_alpha": round(alpha, 4),
        })

    return pd.DataFrame(results)