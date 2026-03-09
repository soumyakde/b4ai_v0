"""
core/analytics/qualitative/engine.py
-------------------------------------------------------
Qualitative LLM Rating Engine

Responsibilities:
- Execute LLM ratings using Prompt Contract
- Enforce deterministic inference
- Validate JSON outputs
- Retry on malformed outputs
- Attach hashes + provenance metadata
- Batch processing support

NO UI LOGIC
NO STORAGE LOGIC
-------------------------------------------------------
"""

from typing import Dict, Any, List, Optional
import time
import hashlib
import logging

from openai import OpenAI

from .contracts import (
    LLMPromptContract,
    safe_parse_llm_json,
    LLM_INFERENCE_CONFIG,
)

# =====================================================
# LOGGER
# =====================================================

logger = logging.getLogger("qualitative_engine")
logger.setLevel(logging.INFO)


# =====================================================
# ENGINE CONFIG
# =====================================================

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 60


# =====================================================
# ENGINE
# =====================================================

class QualitativeLLMEngine:
    """
    Deterministic qualitative rating executor.
    """

    def __init__(
        self,
        api_key: Optional[str],
        contract: LLMPromptContract,
        model: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):

        self.client = OpenAI(api_key=api_key)
        self.contract = contract
        self.model = model or contract.model_name
        self.max_retries = max_retries

        self.prompt_hash = contract.compute_prompt_hash()

    # -------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Low-level LLM call.
        """

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            **LLM_INFERENCE_CONFIG,
        )

        return response.output_text

    # -------------------------------------------------

    def _response_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # -------------------------------------------------

    def rate_single(
        self,
        student_id: str,
        module_id: str,
        question_id: str,
        question_text: str,
        response_text: str,
    ) -> Dict[str, Any]:
        """
        Rate a single qualitative response.
        """

        prompt = self.contract.assemble_prompt(
            module_id=module_id,
            question_text=question_text,
            response_text=response_text,
        )

        last_error = None

        for attempt in range(1, self.max_retries + 1):

            try:
                start = time.time()

                raw_output = self._call_llm(prompt)

                latency = time.time() - start

                parsed = safe_parse_llm_json(raw_output)

                return {
                    "student_id": student_id,
                    "module_id": module_id,
                    "question_id": question_id,
                    "prompt_hash": self.prompt_hash,
                    "response_hash": self._response_hash(
                        response_text
                    ),
                    "model": self.model,
                    "latency_seconds": round(latency, 3),
                    "attempt": attempt,
                    "ratings": parsed["ratings"],
                    "raw_llm_output": raw_output,
                }

            except Exception as e:

                logger.warning(
                    f"LLM attempt {attempt} failed: {str(e)}"
                )

                last_error = e
                time.sleep(1.2 * attempt)

        raise RuntimeError(
            f"LLM rating failed after retries: {last_error}"
        )

    # -------------------------------------------------

    def rate_batch(
        self,
        rows: List[Dict[str, Any]],
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Batch rating executor.

        rows format:
        [
            {
                student_id,
                module_id,
                question_id,
                question_text,
                response_text
            }
        ]
        """

        results = []

        total = len(rows)

        for i, row in enumerate(rows, start=1):

            result = self.rate_single(
                student_id=row["student_id"],
                module_id=row["module_id"],
                question_id=row["question_id"],
                question_text=row["question_text"],
                response_text=row["response_text"],
            )

            results.append(result)

            if progress_callback:
                progress_callback(i, total)

        return results


# =====================================================
# OPTIONAL: DATAFRAME ADAPTER
# =====================================================

def flatten_ratings(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts nested ratings into flat rows suitable
    for pandas ingestion.
    """

    flat_rows = []

    for rec in records:

        base = {
            "student_id": rec["student_id"],
            "module_id": rec["module_id"],
            "question_id": rec["question_id"],
            "prompt_hash": rec["prompt_hash"],
            "response_hash": rec["response_hash"],
            "model": rec["model"],
            "latency_seconds": rec["latency_seconds"],
        }

        for r in rec["ratings"]:
            row = base.copy()
            row.update(
                {
                    "construct": r["construct"],
                    "rating": r["rating"],
                    "evidence": r["evidence"],
                    "confidence": r["confidence"],
                }
            )

            flat_rows.append(row)

    return flat_rows