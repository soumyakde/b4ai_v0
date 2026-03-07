# core/analytics/filters/filter_spec.py

from typing import List, Optional, Dict, Any
import hashlib
import json
import pandas as pd


class FilterSpec:
    """
    Deterministic filter contract for canonical dataset.

    Operates ONLY on canonical schema:

        user_id
        module_id
        instrument_key
        question_id
        response_value
        item_score
        construct
        grade
        submitted_at
        completed_at

    No raw-schema coupling.
    No free-form filters.
    Fully reproducible.
    """

    # -----------------------------------------------------
    # Constructor
    # -----------------------------------------------------
    def __init__(
        self,
        module_ids: Optional[List[str]] = None,
        instrument_keys: Optional[List[str]] = None,
        question_ids: Optional[List[str]] = None,
        user_ids: Optional[List[str]] = None,
        grades: Optional[List[str]] = None,
        constructs: Optional[List[str]] = None,
        submitted_after: Optional[str] = None,
        submitted_before: Optional[str] = None,
        completed_after: Optional[str] = None,
        completed_before: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ):
        self.module_ids = module_ids
        self.instrument_keys = instrument_keys
        self.question_ids = question_ids
        self.user_ids = user_ids
        self.grades = grades
        self.constructs = constructs
        self.submitted_after = submitted_after
        self.submitted_before = submitted_before
        self.completed_after = completed_after
        self.completed_before = completed_before
        self.min_score = min_score
        self.max_score = max_score

    # -----------------------------------------------------
    # Internal normalization for hashing
    # -----------------------------------------------------
    def _sorted(self, value):
        if value is None:
            return None
        return sorted(value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Canonical serialized representation.
        Lists are sorted to guarantee stable hashing.
        """
        return {
            "module_ids": self._sorted(self.module_ids),
            "instrument_keys": self._sorted(self.instrument_keys),
            "question_ids": self._sorted(self.question_ids),
            "user_ids": self._sorted(self.user_ids),
            "grades": self._sorted(self.grades),
            "constructs": self._sorted(self.constructs),
            "submitted_after": self.submitted_after,
            "submitted_before": self.submitted_before,
            "completed_after": self.completed_after,
            "completed_before": self.completed_before,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------
    def validate(self) -> bool:

        list_fields = [
            "module_ids",
            "instrument_keys",
            "question_ids",
            "user_ids",
            "grades",
            "constructs",
        ]

        for field in list_fields:
            value = getattr(self, field)
            if value is not None:
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    return False

        numeric_fields = ["min_score", "max_score"]
        for field in numeric_fields:
            value = getattr(self, field)
            if value is not None and not isinstance(value, (int, float)):
                return False

        return True

    # -----------------------------------------------------
    # Deterministic Hash
    # -----------------------------------------------------
    def filter_hash(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -----------------------------------------------------
    # Vectorized Mask
    # -----------------------------------------------------
    def as_mask(self, df: pd.DataFrame) -> pd.Series:

        required_columns = [
            "user_id",
            "module_id",
            "instrument_key",
            "question_id",
            "item_score",
            "construct",
            "grade",
            "submitted_at",
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Canonical dataset missing columns: {missing}")

        mask = pd.Series(True, index=df.index)

        if self.module_ids:
            mask &= df["module_id"].isin(self.module_ids)

        if self.instrument_keys:
            mask &= df["instrument_key"].isin(self.instrument_keys)

        if self.question_ids:
            mask &= df["question_id"].isin(self.question_ids)

        if self.user_ids:
            mask &= df["user_id"].isin(self.user_ids)

        if self.grades:
            mask &= df["grade"].isin(self.grades)

        if self.constructs:
            mask &= df["construct"].isin(self.constructs)

        # -----------------------
        # Date handling (typed)
        # -----------------------
        if self.submitted_after or self.submitted_before:
            submitted = pd.to_datetime(df["submitted_at"], errors="coerce")

            if self.submitted_after:
                after = pd.to_datetime(self.submitted_after)
                mask &= submitted > after

            if self.submitted_before:
                before = pd.to_datetime(self.submitted_before)
                mask &= submitted < before

        if "completed_at" in df.columns and (
            self.completed_after or self.completed_before
        ):
            completed = pd.to_datetime(df["completed_at"], errors="coerce")

            if self.completed_after:
                after = pd.to_datetime(self.completed_after)
                mask &= completed > after

            if self.completed_before:
                before = pd.to_datetime(self.completed_before)
                mask &= completed < before

        # -----------------------
        # Score filters
        # -----------------------
        if self.min_score is not None:
            mask &= df["item_score"].notna() & (df["item_score"] >= self.min_score)

        if self.max_score is not None:
            mask &= df["item_score"].notna() & (df["item_score"] <= self.max_score)

        return mask