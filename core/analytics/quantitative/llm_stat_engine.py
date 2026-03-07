"""
llm_stat_engine.py

Strict interpretation layer for statistical and IRT outputs.

Responsibilities:
- Accept precomputed statistical outputs
- Accept precomputed IRT outputs
- Generate structured narrative summaries
- Optionally call an injected LLM client

Non-responsibilities:
- No dataset building
- No filtering
- No statistical computation
- No IRT computation
- No direct Streamlit usage
"""

from __future__ import annotations

from typing import Dict, Optional, Any
import pandas as pd
import json
import hashlib


# ==========================================================
# LLM STAT ENGINE
# ==========================================================

class LLMStatEngine:
    """
    Strict interpretation layer.

    Works in two modes:
        1. Deterministic template mode (no LLM)
        2. LLM-assisted mode (if client injected)

    Does NOT compute statistics.
    """

    # ------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        temperature: float = 0.0
    ):
        self.llm_client = llm_client
        self.temperature = temperature

    # ======================================================
    # PUBLIC INTERFACE
    # ======================================================

    def summarize_statistics(
        self,
        instrument_key: str,
        stats_output: Dict[str, Any]
    ) -> str:
        """
        Accepts output from statistics_engine.
        Returns narrative summary.
        """

        self._validate_stats_output(stats_output)

        if self.llm_client:
            return self._llm_summarize_stats(
                instrument_key,
                stats_output
            )

        return self._deterministic_stats_summary(
            instrument_key,
            stats_output
        )

    def summarize_irt(
        self,
        instrument_key: str,
        irt_output: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Accepts output from irt_engine.
        Returns narrative summary.
        """

        self._validate_irt_output(irt_output)

        if self.llm_client:
            return self._llm_summarize_irt(
                instrument_key,
                irt_output
            )

        return self._deterministic_irt_summary(
            instrument_key,
            irt_output
        )

    # ======================================================
    # VALIDATION
    # ======================================================

    def _validate_stats_output(self, stats_output: Dict[str, Any]):
        required = {"summary", "per_item"}
        missing = required - set(stats_output.keys())
        if missing:
            raise ValueError(
                f"Statistics output missing required keys: {missing}"
            )

    def _validate_irt_output(self, irt_output: Dict[str, pd.DataFrame]):
        required = {"item_parameters", "person_parameters"}
        missing = required - set(irt_output.keys())
        if missing:
            raise ValueError(
                f"IRT output missing required keys: {missing}"
            )

    # ======================================================
    # DETERMINISTIC STAT SUMMARY
    # ======================================================

    def _deterministic_stats_summary(
        self,
        instrument_key: str,
        stats_output: Dict[str, Any]
    ) -> str:

        summary = stats_output["summary"]
        per_item = stats_output["per_item"]

        n_students = summary.get("n_students", "N/A")
        mean_total = summary.get("mean_total_score", "N/A")
        std_total = summary.get("std_total_score", "N/A")

        hardest_item = per_item.sort_values(
            "mean_score"
        ).iloc[0]["item_id"]

        easiest_item = per_item.sort_values(
            "mean_score",
            ascending=False
        ).iloc[0]["item_id"]

        return (
            f"Instrument '{instrument_key}' includes {n_students} students. "
            f"The mean total score is {mean_total:.2f} "
            f"(SD = {std_total:.2f}). "
            f"The most difficult item based on mean score is '{hardest_item}', "
            f"while the easiest item is '{easiest_item}'."
        )

    # ======================================================
    # DETERMINISTIC IRT SUMMARY
    # ======================================================

    def _deterministic_irt_summary(
        self,
        instrument_key: str,
        irt_output: Dict[str, pd.DataFrame]
    ) -> str:

        item_df = irt_output["item_parameters"]
        person_df = irt_output["person_parameters"]

        hardest = item_df.sort_values(
            "difficulty_b",
            ascending=False
        ).iloc[0]

        easiest = item_df.sort_values(
            "difficulty_b"
        ).iloc[0]

        best_discriminating = item_df.sort_values(
            "discrimination_r",
            ascending=False
        ).iloc[0]

        mean_theta = person_df["theta_estimate"].mean()

        return (
            f"IRT analysis for '{instrument_key}' shows a mean ability "
            f"estimate (theta) of {mean_theta:.2f}. "
            f"The most difficult item is '{hardest['item_id']}' "
            f"(b = {hardest['difficulty_b']:.2f}). "
            f"The easiest item is '{easiest['item_id']}' "
            f"(b = {easiest['difficulty_b']:.2f}). "
            f"The highest discriminating item is "
            f"'{best_discriminating['item_id']}' "
            f"(r = {best_discriminating['discrimination_r']:.2f})."
        )

    # ======================================================
    # LLM MODE
    # ======================================================

    def _llm_summarize_stats(
        self,
        instrument_key: str,
        stats_output: Dict[str, Any]
    ) -> str:

        prompt = {
            "task": "Summarize statistical output for educators.",
            "instrument": instrument_key,
            "data": stats_output
        }

        return self._call_llm(prompt)

    def _llm_summarize_irt(
        self,
        instrument_key: str,
        irt_output: Dict[str, pd.DataFrame]
    ) -> str:

        serialized = {
            "item_parameters": irt_output["item_parameters"].to_dict(),
            "person_parameters": irt_output["person_parameters"].to_dict()
        }

        prompt = {
            "task": "Interpret IRT results for teachers.",
            "instrument": instrument_key,
            "data": serialized
        }

        return self._call_llm(prompt)

    # ======================================================
    # LLM CALL WRAPPER
    # ======================================================

    def _call_llm(self, payload: Dict[str, Any]) -> str:

        if not self.llm_client:
            raise RuntimeError("No LLM client configured.")

        response = self.llm_client.generate(
            prompt=json.dumps(payload),
            temperature=self.temperature
        )

        return response