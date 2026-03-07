# core/analysis/quantitative/irt_engine.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


REQUIRED_COLUMNS = {
    "user_id",
    "instrument_key",
    "question_id",
    "item_score",
}


class IRTEngine:
    """
    Deterministic 1PL (Rasch-style) approximation.
    Operates strictly on canonical dataset.
    """

    def __init__(self, df: pd.DataFrame):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        self.df = df.copy(deep=True)
        self.dataset_hash = df.attrs.get("dataset_hash")

        self._validate_schema()
        self._validate_numeric()

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------
    def _validate_schema(self):
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Canonical dataset missing required columns: {missing}"
            )

    def _validate_numeric(self):
        if not pd.api.types.is_numeric_dtype(self.df["item_score"]):
            raise TypeError("item_score must be numeric for IRT.")

    # --------------------------------------------------
    # ELIGIBLE INSTRUMENTS
    # --------------------------------------------------
    def get_eligible_instruments(
        self,
        min_items: int = 2,
        min_students: int = 2,
    ) -> List[str]:

        eligible = []

        for instrument, group in self.df.groupby("instrument_key"):

            n_items = group["question_id"].nunique()
            n_students = group["user_id"].nunique()

            if n_items >= min_items and n_students >= min_students:
                eligible.append(instrument)

        return sorted(eligible)

    # --------------------------------------------------
    # MAIN IRT ROUTINE
    # --------------------------------------------------
    def run(self, instrument_key: str) -> Optional[Dict[str, pd.DataFrame]]:

        group = self.df[self.df["instrument_key"] == instrument_key]

        if group.empty:
            return None

        matrix = self._build_matrix(group)

        if matrix.shape[1] < 2 or matrix.shape[0] < 2:
            return None

        total_scores = matrix.sum(axis=1)

        if total_scores.std(ddof=1) == 0:
            return None

        theta = (
            total_scores - total_scores.mean()
        ) / total_scores.std(ddof=1)

        person_df = pd.DataFrame({
            "user_id": matrix.index,
            "theta_estimate": theta.values,
            "raw_total_score": total_scores.values
        }).sort_values("theta_estimate", ascending=False)

        p_values = matrix.mean(axis=0).clip(0.01, 0.99)
        difficulty = -np.log(p_values / (1 - p_values))

        discrimination = {}

        for item in matrix.columns:

            item_vector = matrix[item]

            if item_vector.nunique() <= 1:
                discrimination[item] = np.nan
                continue

            discrimination[item] = item_vector.corr(total_scores)

        item_df = pd.DataFrame({
            "question_id": matrix.columns,
            "difficulty_b": difficulty.values,
            "discrimination_r": [
                discrimination[i] for i in matrix.columns
            ],
            "mean_score": p_values.values
        }).sort_values("difficulty_b")

        for df_out in [person_df, item_df, matrix]:
            df_out.attrs["dataset_hash"] = self.dataset_hash

        return {
            "item_parameters": item_df.reset_index(drop=True),
            "person_parameters": person_df.reset_index(drop=True),
            "matrix": matrix,
        }

    # --------------------------------------------------
    # MATRIX BUILDER
    # --------------------------------------------------
    def _build_matrix(self, df: pd.DataFrame) -> pd.DataFrame:

        matrix = df.pivot_table(
            index="user_id",
            columns="question_id",
            values="item_score",
            aggfunc="mean"
        )

        matrix = matrix.dropna(axis=0, how="all")
        matrix = matrix.dropna(axis=1, how="all")

        variances = matrix.var(axis=0)
        valid_cols = variances[variances > 0].index
        matrix = matrix[valid_cols]

        matrix = matrix.sort_index().sort_index(axis=1)
        matrix.attrs["dataset_hash"] = self.dataset_hash

        return matrix