"""
core/analytics/qualitative/contracts.py
-------------------------------------------------------
Qualitative Layer Contracts for BasicsB4AI

Defines:
- Prompt contract
- Output schema
- Hashing for reproducibility
- Prompt assembly
- JSON validation
- Deterministic guarantees

This file contains NO business logic.
It defines ONLY data contracts.
-------------------------------------------------------
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import json
import hashlib
import jsonschema


# =====================================================
# VERSIONING
# =====================================================

PROMPT_CONTRACT_VERSION = "LLM_PROMPT_CONTRACT_V1"
OUTPUT_SCHEMA_VERSION = "LLM_RATING_OUTPUT_V1"


# =====================================================
# OUTPUT JSON SCHEMA
# =====================================================

LLM_RATING_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "construct": {"type": "string"},
                    "rating": {"type": "integer"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": [
                    "construct",
                    "rating",
                    "evidence",
                    "confidence"
                ]
            }
        }
    },
    "required": ["ratings"]
}


# =====================================================
# SYSTEM PROMPT (INVARIANT)
# =====================================================

SYSTEM_PROMPT = """
You are a trained qualitative research rater.

Your task is to assign analytic construct ratings to student reflections.

Rules:
- Evaluate ONLY evidence present in the text.
- Do NOT infer intentions.
- Do NOT provide feedback.
- Do NOT summarize.
- You are NOT grading correctness.
- You are measuring construct articulation.

If evidence is unclear, assign the LOWER rating.

Return ONLY valid JSON.
""".strip()


# =====================================================
# RUBRIC BLOCK (FIXED SCALE)
# =====================================================

RUBRIC_BLOCK = """
RATING SCALE (0–3)

0 — Absent
No evidence of the construct.

1 — Implicit
Construct suggested but unclear.

2 — Explicit
Clear statement demonstrating the construct.

3 — Elaborated
Detailed explanation with reasoning or examples.

When uncertain, choose the LOWER rating.
""".strip()


# =====================================================
# DATA CONTRACT
# =====================================================

@dataclass(frozen=True)
class LLMPromptContract:
    """
    Immutable prompt specification.
    """

    theory_block: str
    constructs: List[str]
    model_name: str

    system_prompt: str = SYSTEM_PROMPT
    rubric_block: str = RUBRIC_BLOCK
    prompt_version: str = PROMPT_CONTRACT_VERSION

    # -------------------------------------------------

    def build_instance_block(
        self,
        module_id: str,
        question_text: str,
        response_text: str,
    ) -> str:

        return f"""
STUDENT RESPONSE

Module: {module_id}
Question: {question_text}

Response:
\"\"\"
{response_text}
\"\"\"
""".strip()

    # -------------------------------------------------

    def assemble_prompt(
        self,
        module_id: str,
        question_text: str,
        response_text: str,
    ) -> str:

        instance = self.build_instance_block(
            module_id,
            question_text,
            response_text,
        )

        prompt = f"""
{self.system_prompt}

CONSTRUCT DEFINITIONS
{self.theory_block}

{self.rubric_block}

{instance}

OUTPUT FORMAT:
Return ONLY valid JSON matching the schema.
""".strip()

        return prompt

    # -------------------------------------------------

    def compute_prompt_hash(self) -> str:
        """
        Hash ensures reproducibility across studies.
        """

        payload = (
            self.system_prompt
            + self.theory_block
            + self.rubric_block
            + self.model_name
            + self.prompt_version
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================
# VALIDATION UTILITIES
# =====================================================

def validate_llm_output(output_json: Dict[str, Any]) -> None:
    """
    Validates JSON structure against schema.
    Raises exception if invalid.
    """

    jsonschema.validate(
        instance=output_json,
        schema=LLM_RATING_OUTPUT_SCHEMA
    )

    for item in output_json["ratings"]:
        rating = item["rating"]

        if rating not in (0, 1, 2, 3):
            raise ValueError(
                f"Invalid rating value: {rating}"
            )

        if not (0.0 <= item["confidence"] <= 1.0):
            raise ValueError(
                "Confidence must be between 0 and 1."
            )


# =====================================================
# PARSING SAFETY
# =====================================================

def safe_parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """
    Extract and validate JSON returned by LLM.
    """

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM output is not valid JSON"
        ) from e

    validate_llm_output(data)

    return data


# =====================================================
# DETERMINISTIC INFERENCE CONFIG
# =====================================================

LLM_INFERENCE_CONFIG = {
    "temperature": 0,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}