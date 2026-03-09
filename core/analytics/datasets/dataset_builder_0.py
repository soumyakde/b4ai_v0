# core/analytics/datasets/dataset_builder.py
from typing import Dict, Optional, Any, List
import pandas as pd
import hashlib
import json

from core.analytics.filters.filter_spec import FilterSpec


CANONICAL_COLUMNS: List[str] = [
    "user_id",
    "module_id",
    "instrument_key",
    "question_id",
    "response_value",
    "item_score",
    "construct",
    "grade",
    "submitted_at",
    "completed_at",
]


class DatasetBuilder:
    """
    Deterministic Canonical Research Dataset Constructor.

    Guarantees:
    - Strict schema enforcement
    - Deterministic scoring
    - Deterministic hashing
    - No hidden mutations
    - No file IO
    - No silent failures
    """

    REQUIRED_RAW_COLUMNS = {
        "user_id",
        "instrument_key",
        "question_id",
        "response_value",
    }

    # --------------------------------------------------
    # INIT
    # --------------------------------------------------

    def __init__(
        self,
        responses_df: pd.DataFrame,
        instruments_dict: Dict[str, Any],
        scoring_dict: Dict[str, Any],
        demographics_df: Optional[pd.DataFrame] = None,
        filter_spec: Optional[FilterSpec] = None,
    ):
        if responses_df is None or responses_df.empty:
            raise ValueError("responses_df cannot be empty.")

        self.responses_df = responses_df.copy()
        self.instruments_dict = instruments_dict or {}
        self.scoring_dict = scoring_dict or {}
        self.demographics_df = demographics_df
        self.filter_spec = filter_spec

    # --------------------------------------------------
    # RAW SCHEMA NORMALIZATION
    # --------------------------------------------------

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:

        rename_map = {
            "instrument_name": "instrument_key",
            "response_raw": "response_value",
            "response_score": "item_score",
            "timestamp": "submitted_at",
        }

        df = df.rename(columns=rename_map)

        missing = self.REQUIRED_RAW_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required raw columns: {missing}")

        return df

    # --------------------------------------------------
    # VECTORISED SCORING (STRICT)
    # --------------------------------------------------

    def _apply_scoring(self, df: pd.DataFrame) -> pd.Series:

        # Always initialize full-length numeric series
        item_scores = pd.Series(index=df.index, dtype=float)

        for instrument_key, scoring_info in self.scoring_dict.items():

            mask = df["instrument_key"] == instrument_key
            if not mask.any():
                continue

            scoring_type = scoring_info.get("scoring_type")

            # -------------------------
            # LIKERT
            # -------------------------
            if scoring_type == "likert":

                default_scale = scoring_info.get("default_scale")
                reverse_scale = scoring_info.get("reverse_scale")
                reverse_questions = set(scoring_info.get("reverse_questions", []))

                if not isinstance(default_scale, dict):
                    raise ValueError(
                        f"{instrument_key} missing valid default_scale."
                    )

                # Default mapping
                mapped = (
                    df.loc[mask, "response_value"]
                    .map(default_scale)
                )

                item_scores.loc[mask] = pd.to_numeric(
                    mapped,
                    errors="coerce"
                )

                # Reverse override
                if reverse_scale:
                    rev_mask = mask & df["question_id"].isin(reverse_questions)

                    reversed_mapped = (
                        df.loc[rev_mask, "response_value"]
                        .map(reverse_scale)
                    )

                    item_scores.loc[rev_mask] = pd.to_numeric(
                        reversed_mapped,
                        errors="coerce"
                    )

            # -------------------------
            # BINARY
            # -------------------------
            elif scoring_type == "binary":

                correct_answers = scoring_info.get("correct_answers")
                if not isinstance(correct_answers, dict):
                    raise ValueError(
                        f"{instrument_key} missing correct_answers mapping."
                    )

                def normalize_response(x):
                    if pd.isna(x):
                        return x
                    x = str(x).strip()
                    if ":" in x:
                        return x.split(":")[0].strip()
                    return x

                responses = df.loc[mask, "response_value"].apply(normalize_response)
                correct_map = df.loc[mask, "question_id"].map(correct_answers)

                item_scores.loc[mask] = (
                    responses == correct_map
                ).astype(float)

            # -------------------------
            # UNKNOWN SCORING TYPE
            # -------------------------
            else:
                raise ValueError(
                    f"Unsupported scoring_type '{scoring_type}' "
                    f"for instrument '{instrument_key}'."
                )

        # FINAL STRICT NUMERIC ENFORCEMENT
        item_scores = pd.to_numeric(item_scores, errors="coerce")

        if item_scores.dtype.kind not in "fi":
            raise TypeError("item_score must be numeric dtype")

        return item_scores

    # --------------------------------------------------
    # ATTACH INSTRUMENT METADATA
    # --------------------------------------------------
    def _attach_metadata(self, df: pd.DataFrame) -> pd.DataFrame:

        print("Loaded instrument keys:", list(self.instruments_dict.keys()))
        print("Instrument keys in DB:", df["instrument_key"].unique())

        # ---- NORMALIZE QUESTION ID FORMAT ----
        df["question_id"] = df["question_id"].astype(str).str.strip()
        df["instrument_key"] = df["instrument_key"].astype(str).str.strip()

        meta_rows = []

        unique_instruments = df["instrument_key"].unique()

        for instrument_key in unique_instruments:

            instrument_df = df[df["instrument_key"] == instrument_key]
            question_ids = instrument_df["question_id"].unique()

            # -------------------------
            # YAML-defined instruments
            # -------------------------
            if instrument_key in self.instruments_dict:

                instrument_meta = self.instruments_dict[instrument_key]
                module_id = instrument_meta.get("module_id") or instrument_meta.get("scope")

                if not module_id:
                    raise ValueError(f"Instrument '{instrument_key}' missing module_id/scope.")

                # Build mapping from YAML
                yaml_qids = {}

                if "questions" in instrument_meta:
                    for block in instrument_meta["questions"]:
                        construct = block.get("name")
                        for question in block.get("questions", []):
                            qid = str(question.get("id")).strip()
                            yaml_qids[qid] = construct

                if "question_bank" in instrument_meta:
                    for question in instrument_meta["question_bank"]:
                        qid = str(question.get("id")).strip()
                        yaml_qids[qid] = None

                # Only generate metadata for question_ids that actually exist in DB
                for qid in question_ids:
                    construct = yaml_qids.get(qid)
                    meta_rows.append({
                        "instrument_key": instrument_key,
                        "question_id": qid,
                        "module_id": module_id,
                        "construct": construct,
                    })

            # -------------------------
            # Auto-infer everything else
            # -------------------------
            else:

                # Infer module_id heuristically
                if instrument_key.startswith("module") and "_content_mcq" in instrument_key:
                    module_id = instrument_key.split("_")[0]
                elif instrument_key == "module_reflections":
                    module_id = "reflection"
                elif instrument_key == "demographics_survey":
                    module_id = "demographics"
                else:
                    module_id = "survey"

                for qid in question_ids:
                    meta_rows.append({
                        "instrument_key": instrument_key,
                        "question_id": qid,
                        "module_id": module_id,
                        "construct": None,
                    })

        meta_df = pd.DataFrame(meta_rows)

        if meta_df.empty:
            raise ValueError("No instrument metadata generated.")

        # ---- MERGE ----
        df = df.merge(
            meta_df,
            on=["instrument_key", "question_id"],
            how="left",
        )

        # ---- STRICT VALIDATION ----
        if df["module_id"].isna().any():
            failing = df[df["module_id"].isna()]
            raise ValueError(
                f"Metadata merge failed for rows:\n"
                f"{failing[['instrument_key','question_id']].drop_duplicates()}"
            )

        return df

    # --------------------------------------------------
    # HASH (DETERMINISTIC)
    # --------------------------------------------------

    def _compute_hash(self, canonical_df: pd.DataFrame) -> str:

        df_sorted = canonical_df.sort_values(
            by=[
                "user_id",
                "module_id",
                "instrument_key",
                "question_id",
            ]
        )

        data_bytes = df_sorted.to_csv(index=False).encode("utf-8")

        meta_payload = json.dumps(
            {
                "instruments_dict": self.instruments_dict,
                "scoring_dict": self.scoring_dict,
                "filter_hash": (
                    self.filter_spec.filter_hash()
                    if self.filter_spec else None
                ),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")

        combined = data_bytes + meta_payload

        return hashlib.sha256(combined).hexdigest()

    # --------------------------------------------------
    # BUILD CANONICAL DATASET
    # --------------------------------------------------

    def build(self) -> pd.DataFrame:

        df = self._normalize_schema(self.responses_df.copy())

        # Attach metadata first
        df = self._attach_metadata(df)

        # Deterministic scoring
        df["item_score"] = self._apply_scoring(df)

        # Attach demographics
        if self.demographics_df is not None:
            if "user_id" not in self.demographics_df.columns:
                raise ValueError("demographics_df missing user_id column.")

            df = df.merge(
                self.demographics_df[["user_id", "grade"]],
                on="user_id",
                how="left",
            )
        else:
            df["grade"] = None

        # Ensure completed_at
        if "completed_at" not in df.columns:
            df["completed_at"] = None

        # Apply FilterSpec AFTER canonicalization
        if self.filter_spec:
            if not self.filter_spec.validate():
                raise ValueError("Invalid FilterSpec provided.")

            mask = self.filter_spec.as_mask(df)
            df = df[mask].copy()

        # Enforce canonical column ordering
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        canonical_df = df[CANONICAL_COLUMNS].copy()

        # Deterministic dataset hash
        dataset_hash = self._compute_hash(canonical_df)
        canonical_df.attrs["dataset_hash"] = dataset_hash

        if self.filter_spec:
            canonical_df.attrs["filter_hash"] = (
                self.filter_spec.filter_hash()
            )

        return canonical_df