# core/analysis/quantitative/aggregation_engine.py
from __future__ import annotations

from typing import Optional, Dict, List
import pandas as pd


class DatasetAggregator:
    """
    Deterministic descriptive aggregation layer.

    Assumptions:
    - Input is canonical long-format dataset from DatasetBuilder
    - One row per user_id × question_id × instrument_key × module_id

    Guarantees:
    - NEVER mutates schema
    - NEVER alters scoring
    - NEVER performs reliability
    - NEVER performs IRT
    - All outputs carry dataset_hash
    """

    REQUIRED_COLUMNS = {
        "user_id",
        "module_id",
        "instrument_key",
        "question_id",
        "item_score",
    }

    # ---------------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------------
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_hash: Optional[str] = None,
        enforce_numeric: bool = True,
    ):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        self.df = df.copy(deep=True)
        self.dataset_hash = dataset_hash or df.attrs.get("dataset_hash")

        self._validate_schema()

        if enforce_numeric:
            self._validate_numeric_item_score()

        self._sort_canonical()

    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------
    def _validate_schema(self):
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Canonical dataset missing required columns: {missing}"
            )

    def _validate_numeric_item_score(self):
        if not pd.api.types.is_numeric_dtype(self.df["item_score"]):
            raise TypeError("item_score must be numeric dtype")

    def _sort_canonical(self):
        self.df = self.df.sort_values(
            ["instrument_key", "module_id", "user_id", "question_id"]
        ).reset_index(drop=True)

    # ---------------------------------------------------
    # SAFE GROUPBY CORE
    # ---------------------------------------------------
    def _safe_groupby(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        value_col: str,
        agg_dict: Dict[str, object],
    ) -> pd.DataFrame:

        if not all(col in df.columns for col in group_cols):
            raise ValueError("Invalid group columns")

        if value_col not in df.columns:
            raise ValueError("Invalid value column")

        if df.empty:
            result = pd.DataFrame(columns=group_cols + list(agg_dict.keys()))
        else:
            result = (
                df.groupby(group_cols, dropna=False)[value_col]
                .agg(**agg_dict)
                .reset_index()
            )

        result = result.sort_values(group_cols).reset_index(drop=True)
        result.attrs["dataset_hash"] = self.dataset_hash
        return result

    # ---------------------------------------------------
    # ITEM LEVEL
    # ---------------------------------------------------
    def item_level_metrics(self) -> pd.DataFrame:
        return self._safe_groupby(
            self.df,
            group_cols=["instrument_key", "question_id"],
            value_col="item_score",
            agg_dict={
                "mean_score": "mean",
                "std_score": "std",
                "n_responses": lambda x: x.notna().sum(),
                "missing_count": lambda x: x.isna().sum(),
            },
        )

    # ---------------------------------------------------
    # CONSTRUCT LEVEL
    # ---------------------------------------------------
    def construct_level_metrics(self) -> pd.DataFrame:

        if "construct" not in self.df.columns:
            empty = pd.DataFrame(
                columns=[
                    "instrument_key",
                    "construct",
                    "mean_score",
                    "std_score",
                    "n_responses",
                    "missing_count",
                ]
            )
            empty.attrs["dataset_hash"] = self.dataset_hash
            return empty

        df = self.df.dropna(subset=["construct"])

        return self._safe_groupby(
            df,
            group_cols=["instrument_key", "construct"],
            value_col="item_score",
            agg_dict={
                "mean_score": "mean",
                "std_score": "std",
                "n_responses": lambda x: x.notna().sum(),
                "missing_count": lambda x: x.isna().sum(),
            },
        )

    # ---------------------------------------------------
    # INSTRUMENT LEVEL
    # ---------------------------------------------------
    def instrument_level_metrics(self) -> pd.DataFrame:
        return self._safe_groupby(
            self.df,
            group_cols=["instrument_key"],
            value_col="item_score",
            agg_dict={
                "mean_score": "mean",
                "std_score": "std",
                "n_responses": lambda x: x.notna().sum(),
                "missing_count": lambda x: x.isna().sum(),
            },
        )

    # ---------------------------------------------------
    # MODULE LEVEL
    # ---------------------------------------------------
    def module_level_metrics(self) -> pd.DataFrame:
        return self._safe_groupby(
            self.df,
            group_cols=["module_id"],
            value_col="item_score",
            agg_dict={
                "mean_score": "mean",
                "std_score": "std",
                "n_responses": lambda x: x.notna().sum(),
                "missing_count": lambda x: x.isna().sum(),
            },
        )

    # ---------------------------------------------------
    # STUDENT LEVEL
    # ---------------------------------------------------
    def student_level_metrics(self) -> pd.DataFrame:
        return self._safe_groupby(
            self.df,
            group_cols=["user_id"],
            value_col="item_score",
            agg_dict={
                "mean_score": "mean",
                "total_score": "sum",
                "n_responses": lambda x: x.notna().sum(),
                "missing_count": lambda x: x.isna().sum(),
            },
        )

    # ---------------------------------------------------
    # STUDENT ACTIVITY
    # ---------------------------------------------------
    def student_activity(self) -> pd.DataFrame:

        if self.df.empty:
            result = pd.DataFrame(
                columns=["user_id", "total_submissions"]
            )
        else:
            result = (
                self.df.groupby("user_id", dropna=False)
                .size()
                .reset_index(name="total_submissions")
                .sort_values("total_submissions", ascending=False)
                .reset_index(drop=True)
            )

        result.attrs["dataset_hash"] = self.dataset_hash
        return result