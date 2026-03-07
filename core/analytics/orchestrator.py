# core/analytics/orchestrator.py

import yaml
from pathlib import Path
from core.analytics.qualitative.engine import QualitativeLLMEngine
from core.analytics.qualitative.storage import QualitativeStore
from core.analytics.qualitative.contracts import LLMPromptContract, safe_parse_llm_json
from core.analytics.datasets.dataset_builder import DatasetBuilder
from core.analytics.quantitative.aggregation_engine import AggregationEngine
from core.analytics.quantitative.competency_engine import CompetencyEngine

QUAL_DB_PATH = Path("qualitative_ratings.db")

def load_prompt_contract(yaml_path: Path, instrument_key: str) -> LLMPromptContract:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data.get("instrument_key") != instrument_key:
        raise ValueError(f"No YAML config for instrument {instrument_key}")

    return LLMPromptContract(
        theory_block=data["theory_block"],
        constructs=data["constructs"],
        model_name=data["model_name"]
    )


class AnalyticsOrchestrator:
    """
    Central execution coordinator.

    Streamlit MUST only talk to this class.
    """

    def __init__(self, filter_spec, llm_client=None):
        self.filter_spec = filter_spec
        self.llm_client = llm_client

    # --------------------------------------------------
    # BUILD CANONICAL DATASET
    # --------------------------------------------------
    def build_dataset(self):
        builder = DatasetBuilder(self.filter_spec)
        canonical_df = builder.build()
        return canonical_df

    # --------------------------------------------------
    # QUANT PIPELINE
    # --------------------------------------------------
    def run_quantitative(self, canonical_df):

        agg = AggregationEngine(canonical_df)
        module_scores = agg.module_scores()

        comp = CompetencyEngine(canonical_df)
        competency_scores = comp.student_competency_metrics()

        cpi = comp.competency_progression_index()

        return {
            "module_scores": module_scores,
            "competency_scores": competency_scores,
            "cpi": cpi,
        }

    # --------------------------------------------------
    # QUAL PIPELINE (OPTIONAL EXECUTION)
    # --------------------------------------------------
    def run_qualitative(self, reflection_df):

        if self.llm_client is None:
            raise RuntimeError("LLM client required for qualitative coding.")

        engine = QualitativeCodingEngine(self.llm_client)
        ratings = engine.rate_batch(reflection_df)

        store = QualitativeStore()
        store.save(ratings)

        return ratings
    
    #------------for running qualitative 
    def run_qualitative(self, reflection_df: pd.DataFrame, instrument_key: str = "module_reflections"):
        if self.llm_client is None:
            raise RuntimeError("LLM client required for qualitative coding.")

        yaml_path = Path("streamlit_app/surveys/qualitative_prompts.yaml")
        contract = load_prompt_contract(yaml_path, instrument_key)

        engine = QualitativeLLMEngine(
            api_key=self.llm_client.api_key,  # or pass OpenAI client
            contract=contract
        )

        # Prepare rows in required dict format
        rows = []
        for _, row in reflection_df.iterrows():
            rows.append({
                "student_id": str(row["user_id"]),
                "module_id": str(row["module_id"]),
                "question_id": str(row["question_id"]),
                "question_text": row.get("question_text", ""),
                "response_text": row.get("response_text", "")
            })

        ratings = engine.rate_batch(rows)

        store = QualitativeStore(db_path=QUAL_DB_PATH)
        store.save(ratings)

        return ratings