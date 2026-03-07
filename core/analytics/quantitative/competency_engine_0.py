# core/analytics/quantitative/competency_engine.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.yaml_loader import load_yaml


class CompetencyEngine:
    """
    Cross-module competency projection + progression layer.

    Responsibilities
    ----------------
    1. Project module performance → competencies
    2. Aggregate student and cohort competency scores
    3. Compute Competency Progression Index (CPI)

    Guarantees
    ----------
    - NEVER mutates input dataframe
    - NEVER alters scoring
    - Deterministic outputs
    - Dataset hash propagation
    """

    REQUIRED_COLUMNS = {
        "user_id",
        "module_id",
        "item_score",
    }


    # ---------------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------------
    def __init__(
        self,
        df: pd.DataFrame,
        module_column: str = "module_id",  # <- new argument
        ontology_path: Optional[str | Path] = None,
        dataset_hash: Optional[str] = None,
    ):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be pandas DataFrame")

        self.df = df.copy(deep=True)
        self.module_column = module_column  # <- store column name
        self.dataset_hash = dataset_hash or df.attrs.get("dataset_hash")

        self._validate_schema()

        self.ontology_path = (
            Path(ontology_path)
            if ontology_path
            else Path(__file__).parent / "ai_literacy_ontology.yaml"
        )

        self.ontology = load_yaml(self.ontology_path)

        self._parse_ontology()
        self._load_progression_levels()
        self._sort_canonical()

    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------
    def _validate_schema(self):
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Dataset missing required columns: {missing}"
            )

    def _sort_canonical(self):
        self.df = (
            self.df.sort_values(["module_id", "user_id"])
            .reset_index(drop=True)
        )

    # ---------------------------------------------------
    # ONTOLOGY PARSING
    # ---------------------------------------------------
    def _parse_ontology(self):

        try:
            alignment = self.ontology["module_alignment"]
            competencies = self.ontology["competencies"]
        except KeyError as e:
            raise ValueError(f"Invalid ontology structure: missing {e}")

        self.module_to_competencies: Dict[str, List[str]] = {}

        for module, spec in alignment.items():
            self.module_to_competencies[module] = spec.get(
                "competencies", []
            )

        self.competency_dimension = {
            name: meta["dimension"]
            for name, meta in competencies.items()
        }

    # ---------------------------------------------------
    # CPI LEVEL LOADING (NEW)
    # ---------------------------------------------------
    def _load_progression_levels(self):
        """
        Loads semantic progression levels from ontology
        and attaches deterministic CPI thresholds.

        Ontology defines meaning.
        Code defines measurement policy.
        """

        try:
            levels = self.ontology["progression_levels"]
        except KeyError:
            raise ValueError("Ontology missing progression_levels")

        ordered = list(levels.items())

        # CPI POLICY THRESHOLDS
        thresholds = [
            (0.00, 0.39),
            (0.40, 0.59),
            (0.60, 0.79),
            (0.80, 1.00),
        ]

        if len(ordered) != len(thresholds):
            raise ValueError(
                "Progression levels must match CPI thresholds"
            )

        self.progression_levels: List[
            Tuple[str, str, float, float, str]
        ] = []

        for (key, spec), (min_s, max_s) in zip(
            ordered, thresholds
        ):
            self.progression_levels.append(
                (
                    key,
                    spec.get("label", key),
                    min_s,
                    max_s,
                    spec.get("description", ""),
                )
            )

    # ---------------------------------------------------
    # INTERNAL: MODULE MEANS PER STUDENT
    # ---------------------------------------------------
    def _student_module_scores(self) -> pd.DataFrame:

        if self.df.empty:
            return pd.DataFrame(
                columns=["user_id", self.module_column, "module_mean"]
            )

        result = (
            self.df.groupby(
                ["user_id", self.module_column], dropna=False
            )["item_score"]
            .mean()
            .reset_index(name="module_mean")
        )

        return result.sort_values(
            ["user_id", self.module_column]
        ).reset_index(drop=True)

    # ---------------------------------------------------
    # STUDENT × COMPETENCY METRICS
    # ---------------------------------------------------
    def student_competency_metrics(self) -> pd.DataFrame:

        module_scores = self._student_module_scores()

        rows = []

        for _, row in module_scores.iterrows():
            competencies = self.module_to_competencies.get(
                row[self.module_column], []  # <- use normalized column
            )
            for comp in competencies:
                rows.append(
                    {
                        "user_id": row["user_id"],
                        "competency": comp,
                        "module_mean": row["module_mean"],
                    }
                )

        if not rows:
            result = pd.DataFrame(
                columns=[
                    "user_id",
                    "competency",
                    "mean_score",
                    "evidence_count",
                ]
            )
        else:
            temp = pd.DataFrame(rows)
            result = (
                temp.groupby(
                    ["user_id", "competency"], dropna=False
                )["module_mean"]
                .agg(
                    mean_score="mean",
                    evidence_count="count",
                )
                .reset_index()
                .sort_values(["user_id", "competency"])
                .reset_index(drop=True)
            )

        result.attrs["dataset_hash"] = self.dataset_hash
        return result


    # ---------------------------------------------------
    # CPI LEVEL ASSIGNMENT (NEW)
    # ---------------------------------------------------
    def _assign_level(self, score: float):

        if pd.isna(score):
            return None, None

        for _, label, min_s, max_s, desc in self.progression_levels:
            if min_s <= score <= max_s:
                return label, desc

        return "unclassified", None

    # ---------------------------------------------------
    # COMPETENCY PROGRESSION INDEX (NEW PUBLIC API)
    # ---------------------------------------------------
    def competency_progression_index(self) -> pd.DataFrame:
        """
        Converts competency scores into developmental levels.
        """

        metrics = self.student_competency_metrics().copy()

        if metrics.empty:
            return metrics

        levels = metrics["mean_score"].apply(
            lambda s: self._assign_level(s)
        )

        metrics["progression_level"] = levels.apply(lambda x: x[0])
        metrics["level_description"] = levels.apply(lambda x: x[1])

        metrics.attrs["dataset_hash"] = self.dataset_hash
        return metrics

    # ---------------------------------------------------
    # COHORT COMPETENCY METRICS
    # ---------------------------------------------------
    def cohort_competency_metrics(self) -> pd.DataFrame:

        student_metrics = self.student_competency_metrics()

        if student_metrics.empty:
            result = pd.DataFrame(
                columns=[
                    "competency",
                    "mean_score",
                    "std_score",
                    "n_students",
                ]
            )
        else:
            result = (
                student_metrics.groupby(
                    "competency", dropna=False
                )["mean_score"]
                .agg(
                    mean_score="mean",
                    std_score="std",
                    n_students="count",
                )
                .reset_index()
                .sort_values("competency")
                .reset_index(drop=True)
            )

        result.attrs["dataset_hash"] = self.dataset_hash
        return result