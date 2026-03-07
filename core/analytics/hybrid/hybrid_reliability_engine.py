import pandas as pd
import numpy as np


class HybridReliabilityEngine:

    @staticmethod
    def run_debug(fusion_df: pd.DataFrame) -> pd.DataFrame:
        """
        DEBUG hybrid reliability diagnostics.

        Expected columns:
            user_id
            construct
            performance_score
            articulation_score
        """

        required = {
            "user_id",
            "construct",
            "performance_score",
            "articulation_score",
        }

        missing = required - set(fusion_df.columns)
        if missing:
            raise ValueError(f"Hybrid reliability missing columns: {missing}")

        results = []

        for construct, g in fusion_df.groupby("construct"):

            perf = g["performance_score"].astype(float)
            qual = g["articulation_score"].astype(float)

            n = len(g)

            if n < 3:
                continue

            var_quant = perf.var()
            var_qual = qual.var()

            corr = perf.corr(qual)

            total_score = perf + qual

            var_total = total_score.var()
            var_true = perf.var() + qual.var()

            composite_rel = var_true / var_total if var_total != 0 else np.nan

            discrepancy = perf - qual
            discrepancy_var = discrepancy.var()

            results.append(
                {
                    "construct": construct,
                    "n": n,
                    "quant_variance": var_quant,
                    "qual_variance": var_qual,
                    "cross_modal_corr": corr,
                    "composite_reliability": composite_rel,
                    "discrepancy_variance": discrepancy_var,
                }
            )

        df = pd.DataFrame(results)

        print("\n=== HYBRID RELIABILITY DEBUG ===")
        print(df)

        return df