import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path


class HybridCompetencyEngine:
    """
    Integrates quantitative CPI with qualitative ratings
    to produce ICC-weighted Hybrid Competency Index (CPI+).

    CPI+ = w_quant * z(CPI_quant) + w_qual * ICC * z(CPI_qual)

    If ICC values are unavailable, the engine gracefully falls
    back to the standard weighted fusion.
    """

    def __init__(
        self,
        quant_cpi_df: pd.DataFrame,
        qualitative_db: Path = Path("qualitative_ratings.db"),
        agreement_db: Path = Path("qualitative_ratings.db"),
        w_quant: float = 0.6,
        w_qual: float = 0.4,
    ):
        self.quant_cpi_df = quant_cpi_df.copy()
        self.qualitative_db = qualitative_db
        self.agreement_db = agreement_db
        self.w_quant = w_quant
        self.w_qual = w_qual

    # --------------------------------------------------
    # LOAD QUALITATIVE RATINGS
    # --------------------------------------------------
    def _load_qualitative(self):

        if not self.qualitative_db.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(self.qualitative_db)

        df = pd.read_sql_query(
            """
            SELECT
                student_id AS user_id,
                construct,
                score
            FROM qualitative_ratings
            """,
            conn,
        )

        conn.close()
        return df

    # --------------------------------------------------
    # LOAD ICC AGREEMENT VALUES
    # --------------------------------------------------
    def _load_icc(self):

        if not self.agreement_db.exists():
            return pd.DataFrame()

        try:
            conn = sqlite3.connect(self.agreement_db)

            icc_df = pd.read_sql_query(
                """
                SELECT
                    construct,
                    icc
                FROM agreement_metrics
                """,
                conn,
            )

            conn.close()

            if icc_df.empty:
                return pd.DataFrame()

            return icc_df

        except Exception:
            # If agreement table does not exist yet
            return pd.DataFrame()

    # --------------------------------------------------
    # BUILD QUAL CPI
    # --------------------------------------------------
    def qualitative_cpi(self):

        qual = self._load_qualitative()

        if qual.empty:
            return pd.DataFrame(
                columns=["user_id", "competency", "cpi_qual"]
            )

        qual_cpi = (
            qual.groupby(["user_id", "construct"])["score"]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "construct": "competency",
                    "score": "cpi_qual",
                }
            )
        )

        return qual_cpi

    # --------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------
    @staticmethod
    def _zscore(series: pd.Series):

        if series.std(ddof=0) == 0 or series.isna().all():
            return pd.Series(np.zeros(len(series)), index=series.index)

        return (series - series.mean()) / series.std(ddof=0)

    # --------------------------------------------------
    # HYBRID CPI
    # --------------------------------------------------
    def hybrid_cpi(self):

        quant = self.quant_cpi_df.copy()

        # ----- schema adapter -----
        if "cpi" in quant.columns:
            quant = quant.rename(columns={"cpi": "cpi_quant"})
        elif "mean_score" in quant.columns:
            quant = quant.rename(columns={"mean_score": "cpi_quant"})
        else:
            raise ValueError(
                "HybridEngine: expected 'cpi' or 'mean_score' column."
            )

        required_cols = {"user_id", "competency", "cpi_quant"}
        missing = required_cols - set(quant.columns)

        if missing:
            raise ValueError(f"HybridEngine missing columns: {missing}")

        qual = self.qualitative_cpi()

        merged = pd.merge(
            quant,
            qual,
            on=["user_id", "competency"],
            how="outer",
        )

        # --------------------------------------------------
        # Z-SCORE QUANT
        # --------------------------------------------------
        quant_series = merged["cpi_quant"]
        quant_mean = quant_series.mean()
        if pd.isna(quant_mean):
            quant_mean = 0.0

        merged["z_quant"] = self._zscore(
            quant_series.fillna(quant_mean)
        )

        # --------------------------------------------------
        # Z-SCORE QUAL
        # --------------------------------------------------
        qual_series = merged["cpi_qual"]
        qual_mean = qual_series.mean()
        if pd.isna(qual_mean):
            qual_mean = 0.0

        merged["z_qual"] = self._zscore(
            qual_series.fillna(qual_mean)
        )

        # --------------------------------------------------
        # LOAD ICC VALUES
        # --------------------------------------------------
        icc_df = self._load_icc()

        if not icc_df.empty:

            merged = pd.merge(
                merged,
                icc_df,
                left_on="competency",
                right_on="construct",
                how="left",
            )

            merged["icc"] = merged["icc"].fillna(0.0)

        else:
            merged["icc"] = 1.0

        # --------------------------------------------------
        # ICC-WEIGHTED QUALITATIVE SIGNAL
        # --------------------------------------------------
        merged["z_qual_weighted"] = merged["z_qual"] * merged["icc"]

        # --------------------------------------------------
        # RELIABILITY-WEIGHTED NORMALIZED HYBRID CPI
        # --------------------------------------------------

        numerator = (
            self.w_quant * merged["z_quant"]
            + self.w_qual * merged["z_qual_weighted"]
        )

        denominator = (
            self.w_quant
            + self.w_qual * merged["icc"]
        )

        # Avoid divide-by-zero
        denominator = denominator.replace(0, np.nan)

        merged["cpi_hybrid"] = numerator / denominator

        # --------------------------------------------------
        # DEBUG (SAFE)
        # --------------------------------------------------
        print("\n[HYBRID DEBUG]")
        print("Quant rows:", len(quant))
        print("Qual rows:", len(qual))
        print("Merged rows:", len(merged))
        print("ICC present:", not icc_df.empty)
        print("Null counts:\n", merged.isna().sum())

        return merged[
            [
                "user_id",
                "competency",
                "cpi_quant",
                "cpi_qual",
                "icc",
                "cpi_hybrid",
            ]
        ]