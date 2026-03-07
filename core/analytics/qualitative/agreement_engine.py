"""
agreement_engine.py
-------------------------------------------------------
Phase 1 — Agreement Analytics

Computes agreement between:
    Human raters ↔ LLM raters

Outputs construct-level reliability statistics.

NO UI
NO STREAMLIT
-------------------------------------------------------
"""

from typing import Optional, Dict
import sqlite3
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("agreement_engine")
logger.setLevel(logging.INFO)


# =====================================================
# ICC IMPLEMENTATION (ICC(2,1))
# =====================================================

def icc2_1(data: pd.DataFrame) -> float:
    """
    Computes ICC(2,1) — two-way random absolute agreement.

    data columns:
        target_id | rater | score
    """

    pivot = data.pivot(index="target_id", columns="rater", values="score")

    if pivot.shape[1] < 2:
        return np.nan

    Y = pivot.to_numpy()
    n, k = Y.shape

    mean_targets = np.mean(Y, axis=1, keepdims=True)
    mean_raters = np.mean(Y, axis=0, keepdims=True)
    grand_mean = np.mean(Y)

    MSR = k * np.var(mean_targets, ddof=1)
    MSC = n * np.var(mean_raters, ddof=1)
    MSE = np.var(Y - mean_targets - mean_raters + grand_mean, ddof=1)

    icc = (MSR - MSE) / (
        MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    )

    return float(icc)


# =====================================================
# AGREEMENT ENGINE
# =====================================================

class AgreementEngine:
    """
    Computes human ↔ LLM agreement statistics.
    """

    def __init__(
        self,
        responses_db: str = "responses.db",
        qualitative_db: str = "qualitative_ratings.db",
    ):
        self.responses_db = responses_db
        self.qualitative_db = qualitative_db

    # -------------------------------------------------

    def _load_llm_ratings(self) -> pd.DataFrame:

        with sqlite3.connect(self.qualitative_db) as conn:
            df = pd.read_sql(
                """
                SELECT
                    student_id,
                    module_id,
                    question_id,
                    construct,
                    rating
                FROM llm_ratings
                """,
                conn,
            )

        df["rater"] = "LLM"
        return df

    # -------------------------------------------------

    def _load_human_ratings(self) -> pd.DataFrame:
        """
        Adjust table name if needed.
        """

        with sqlite3.connect(self.responses_db) as conn:
            df = pd.read_sql(
                """
                SELECT
                    student_id,
                    module_id,
                    question_id,
                    construct,
                    rating
                FROM human_ratings
                """,
                conn,
            )

        df["rater"] = "Human"
        return df

    # -------------------------------------------------

    def build_alignment(self) -> pd.DataFrame:
        """
        Aligns human + LLM ratings.
        """

        human = self._load_human_ratings()
        llm = self._load_llm_ratings()

        df = pd.concat([human, llm], ignore_index=True)

        df["target_id"] = (
            df["student_id"]
            + "|"
            + df["module_id"]
            + "|"
            + df["question_id"]
            + "|"
            + df["construct"]
        )

        return df

    # -------------------------------------------------

    def compute_agreement(self) -> pd.DataFrame:
        """
        Returns construct-level agreement metrics.
        """

        df = self.build_alignment()

        results = []

        for construct, g in df.groupby("construct"):

            pivot = g.pivot(
                index="target_id",
                columns="rater",
                values="rating",
            ).dropna()

            if pivot.shape[0] < 5:
                continue

            icc = icc2_1(
                pivot.reset_index()
                .melt(id_vars="target_id",
                      var_name="rater",
                      value_name="score")
            )

            bias = (pivot["LLM"] - pivot["Human"]).mean()

            corr = pivot["LLM"].corr(pivot["Human"])

            results.append(
                {
                    "construct": construct,
                    "n": len(pivot),
                    "ICC2_1": round(icc, 3),
                    "pearson_r": round(corr, 3),
                    "mean_bias": round(bias, 3),
                }
            )

        return pd.DataFrame(results).sort_values(
            "ICC2_1", ascending=False
        )