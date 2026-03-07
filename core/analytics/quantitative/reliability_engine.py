from __future__ import annotations
from typing import Optional, Dict
import pandas as pd
import numpy as np


class ReliabilityEngine:
    """
    Strict reliability computation layer.

    Assumptions:
    - Input is canonical dataset from DatasetBuilder
    - No schema mutation
    - No scoring mutation
    """

    REQUIRED_COLUMNS = {
        "user_id",
        "instrument_key",
        "question_id",
        "item_score",
    }

    def __init__(self, canonical_df: pd.DataFrame):

        if not isinstance(canonical_df, pd.DataFrame):
            raise TypeError("canonical_df must be DataFrame")

        self.df = canonical_df.copy(deep=True)

        self._validate_schema()

        self.dataset_hash = canonical_df.attrs.get("dataset_hash")

    # --------------------------------------------
    # Schema Validation
    # --------------------------------------------
    def _validate_schema(self):

        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Canonical dataset missing required columns: {missing}"
            )

        if not pd.api.types.is_numeric_dtype(self.df["item_score"]):
            raise TypeError("item_score must be numeric")

    # --------------------------------------------
    # Cronbach Alpha
    # --------------------------------------------
    @staticmethod
    def cronbach_alpha(matrix: pd.DataFrame) -> Optional[float]:

        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return None

        matrix = matrix.dropna(axis=0, how="all")
        matrix = matrix.dropna(axis=1, how="all")

        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return None

        item_variances = matrix.var(axis=0, ddof=1)
        total_score = matrix.sum(axis=1)
        total_variance = total_score.var(ddof=1)

        if total_variance <= 0 or not np.isfinite(total_variance):
            return None

        k = matrix.shape[1]

        alpha = (k / (k - 1)) * (
            1 - (item_variances.sum() / total_variance)
        )

        if not np.isfinite(alpha):
            return None

        return round(float(alpha), 4)

    # --------------------------------------------
    # Reliability Per Instrument
    # --------------------------------------------
    def reliability_per_instrument(self) -> pd.DataFrame:

        results = []

        for instr in sorted(self.df["instrument_key"].dropna().unique()):

            inst_df = self.df[
                (self.df["instrument_key"] == instr)
                & self.df["item_score"].notna()
            ]

            if inst_df["question_id"].nunique() < 2:
                continue

            pivot = inst_df.pivot_table(
                index="user_id",
                columns="question_id",
                values="item_score",
                aggfunc="mean",
            )

            alpha = self.cronbach_alpha(pivot)

            if alpha is not None:
                results.append({
                    "instrument_key": instr,
                    "cronbach_alpha": alpha,
                })

        result = pd.DataFrame(results)

        if not result.empty:
            result = (
                result.sort_values("cronbach_alpha", ascending=False)
                .reset_index(drop=True)
            )

        result.attrs["dataset_hash"] = self.dataset_hash

        return result