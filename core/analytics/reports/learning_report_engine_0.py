# core/analytics/reports/learning_report_engine.py

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class LearningReport:
    cohort_summary: pd.DataFrame
    competency_summary: pd.DataFrame
    modality_alignment: pd.DataFrame
    diagnostics: dict


class LearningReportEngine:
    """
    Generates analytic learning reports from CPI and Hybrid CPI data.
    Deterministic, transparent, publication-safe.
    """

    def __init__(self, cpi_df: pd.DataFrame, hybrid_df: pd.DataFrame):
        self.cpi_df = cpi_df.copy()
        self.hybrid_df = hybrid_df.copy()

    # --------------------------------------------------
    # DEBUG CHECKS
    # --------------------------------------------------
    def _debug_inputs(self):

        diagnostics = {}

        diagnostics["cpi_rows"] = len(self.cpi_df)
        diagnostics["hybrid_rows"] = len(self.hybrid_df)

        diagnostics["missing_quant"] = (
            self.hybrid_df["cpi_quant"].isna().sum()
            if "cpi_quant" in self.hybrid_df
            else None
        )

        diagnostics["missing_qual"] = (
            self.hybrid_df["cpi_qual"].isna().sum()
            if "cpi_qual" in self.hybrid_df
            else None
        )

        diagnostics["missing_hybrid"] = (
            self.hybrid_df["cpi_hybrid"].isna().sum()
            if "cpi_hybrid" in self.hybrid_df
            else None
        )

        return diagnostics

    # --------------------------------------------------
    # COHORT SUMMARY
    # --------------------------------------------------
    def cohort_summary(self):

        if self.hybrid_df.empty:
            return pd.DataFrame()

        return (
            self.hybrid_df.groupby("competency")[
                ["cpi_quant", "cpi_qual", "cpi_hybrid"]
            ]
            .agg(["mean", "std", "count"])
            .round(3)
        )

    # --------------------------------------------------
    # COMPETENCY RANKING
    # --------------------------------------------------
    def competency_summary(self):

        if self.hybrid_df.empty:
            return pd.DataFrame()

        df = (
            self.hybrid_df.groupby("competency")["cpi_hybrid"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        df["rank"] = np.arange(1, len(df) + 1)
        return df

    # --------------------------------------------------
    # MODALITY ALIGNMENT
    # --------------------------------------------------
    def modality_alignment(self):

        if self.hybrid_df.empty:
            return pd.DataFrame()

        valid = self.hybrid_df.dropna(
            subset=["cpi_quant", "cpi_qual"]
        )

        if len(valid) < 3:
            return pd.DataFrame()

        corr = (
            valid.groupby("competency")
            .apply(
                lambda x: x["cpi_quant"].corr(x["cpi_qual"])
            )
            .reset_index(name="quant_qual_correlation")
        )

        return corr.round(3)

    # --------------------------------------------------
    # BUILD REPORT
    # --------------------------------------------------
    def build(self):

        diagnostics = self._debug_inputs()

        return LearningReport(
            cohort_summary=self.cohort_summary(),
            competency_summary=self.competency_summary(),
            modality_alignment=self.modality_alignment(),
            diagnostics=diagnostics,
        )