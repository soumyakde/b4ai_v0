"""
statistics_engine.py
Strict statistical engine operating on canonical dataset only.

Responsibilities:
- Instrument-level statistics
- Module-level statistics
- Student-level summaries
- Reliability (Cronbach's alpha)
- Descriptive item statistics

Non-responsibilities:
- No data loading
- No schema normalization
- No Streamlit usage
- No filtering logic
- No mutation of input dataframe
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List


# ==========================================================
# REQUIRED CANONICAL SCHEMA
# ==========================================================

REQUIRED_COLUMNS = {
    "user_id",
    "instrument_key",
    "module_id",
    "question_id",
    "item_score",
}


# ==========================================================
# STATISTICS ENGINE
# ==========================================================

class StatisticsEngine:
    """
    Strict statistics engine.

    Accepts only canonical dataset.
    Raises immediately if schema invalid.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=True)

        self._validate_schema()
        self._validate_numeric()

    # ------------------------------------------------------
    # SCHEMA VALIDATION
    # ------------------------------------------------------
    def _validate_schema(self):
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Canonical dataset missing required columns: {missing}"
            )

    def _validate_numeric(self):
        if not pd.api.types.is_numeric_dtype(self.df["item_score"]):
            raise TypeError(
                "item_score must be numeric in canonical dataset."
            )

    # ======================================================
    # INSTRUMENT-LEVEL METRICS
    # ======================================================
    def instrument_level_metrics(self) -> pd.DataFrame:
        """
        Returns:
            instrument_key
            mean_score
            std_score
            response_count
            unique_students
        """

        grouped = self.df.groupby("instrument_key")

        result = grouped["item_score"].agg(
            mean_score="mean",
            std_score="std",
            response_count="count"
        ).reset_index()

        students = grouped["user_id"].nunique().reset_index(
            name="unique_students"
        )

        result = result.merge(students, on="instrument_key")

        return result.sort_values("mean_score", ascending=False)

    # ======================================================
    # MODULE-LEVEL METRICS
    # ======================================================
    def module_level_metrics(self) -> pd.DataFrame:
        """
        Returns:
            module_id
            mean_score
            std_score
            response_count
        """

        result = (
            self.df
            .groupby("module_id")["item_score"]
            .agg(
                mean_score="mean",
                std_score="std",
                response_count="count"
            )
            .reset_index()
        )

        return result.sort_values("mean_score", ascending=False)

    # ======================================================
    # STUDENT ACTIVITY
    # ======================================================
    def student_activity(self) -> pd.DataFrame:
        """
        Returns:
            user_id
            total_responses
            instruments_attempted
            mean_score
        """

        grouped = self.df.groupby("user_id")

        result = grouped["item_score"].agg(
            total_responses="count",
            mean_score="mean"
        ).reset_index()

        instruments = grouped["instrument_key"].nunique().reset_index(
            name="instruments_attempted"
        )

        result = result.merge(instruments, on="user_id")

        return result.sort_values("total_responses", ascending=False)

    # ======================================================
    # ITEM-LEVEL DESCRIPTIVES
    # ======================================================
    def item_statistics(self) -> pd.DataFrame:
        """
        Returns:
            instrument_key
            item_id
            mean_score
            std_score
            response_count
        """

        result = (
            self.df
            .groupby(["instrument_key", "item_id"])["item_score"]
            .agg(
                mean_score="mean",
                std_score="std",
                response_count="count"
            )
            .reset_index()
        )

        return result

    # ======================================================
    # CRONBACH'S ALPHA
    # ======================================================
    def reliability_per_instrument(self) -> pd.DataFrame:
        """
        Computes Cronbach's alpha per instrument.

        Requires:
            At least 2 items
            Numeric data
        """

        results = []

        for instrument, group in self.df.groupby("instrument_key"):

            matrix = self._pivot_matrix(group)

            if matrix.shape[1] < 2:
                continue

            alpha = self._cronbach_alpha(matrix)

            results.append({
                "instrument_key": instrument,
                "cronbach_alpha": alpha,
                "n_items": matrix.shape[1],
                "n_students": matrix.shape[0],
            })

        return pd.DataFrame(results)

    # ======================================================
    # INTERNAL HELPERS
    # ======================================================
    def _pivot_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates user x item matrix.
        """

        matrix = df.pivot_table(
            index="user_id",
            columns="item_id",
            values="item_score",
            aggfunc="mean"
        )

        matrix = matrix.dropna(axis=0, how="all")
        matrix = matrix.dropna(axis=1, how="all")

        return matrix

    def _cronbach_alpha(self, matrix: pd.DataFrame) -> float:
        """
        Classical Cronbach's alpha formula.
        """

        item_vars = matrix.var(axis=0, ddof=1)
        total_var = matrix.sum(axis=1).var(ddof=1)

        if total_var == 0:
            return np.nan

        n_items = matrix.shape[1]

        alpha = (
            n_items / (n_items - 1)
        ) * (
            1 - (item_vars.sum() / total_var)
        )

        return float(alpha)