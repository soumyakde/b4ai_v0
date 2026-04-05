# core/analytics/cpi/cpi_rubric.py
"""
CPI Reflection Quality Rubric
==============================
Defines the LLM-as-judge rubric for scoring student reflection text
(CPI_qual component of the Competency Progression Index).

Three dimensions, 1-4 Likert scale (matches platform-wide 1-4 convention
used by SCCCES and SIMS; more granular than 1-5 would be for this age group).

Dimension anchors are calibrated for Basics4AI participants aged 10-14,
informed by Zheng et al. (2023) LLM-as-judge methodology.

Reference
---------
Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge
with MT-Bench and Chatbot Arena (Version 4). arXiv.
https://doi.org/10.48550/ARXIV.2306.05685
"""

from __future__ import annotations
from typing import Dict, List, Any

# -----------------------------------------------------------------------
# Rubric dimensions — calibrated for ages 10-14
# -----------------------------------------------------------------------

CPI_QUAL_DIMENSIONS: List[Dict[str, Any]] = [
    {
        "id":          "depth_of_insight",
        "label":       "Depth of insight",
        "description": "Does the student demonstrate genuine understanding "
                       "beyond surface recall? Does the response go beyond "
                       "repeating lesson vocabulary to show the student "
                       "has internalized an idea?",
        "anchors": {
            1: "Vague, absent, or simply restates the question. "
               "No evidence of understanding.",
            2: "Recalls a fact or term from the module but offers no "
               "explanation or personal interpretation.",
            3: "Explains a concept in their own words with a partial "
               "reason or example. Shows basic understanding.",
            4: "Explains a concept clearly in their own words, provides "
               "a concrete example or analogy, and demonstrates why it "
               "matters — appropriate for a 10-14 year old.",
        },
        "max_score": 4,
    },
    {
        "id":          "conceptual_grounding",
        "label":       "Conceptual grounding",
        "description": "Does the response specifically reference AI concepts, "
                       "activities, or vocabulary from the module "
                       "(e.g. sensors, algorithms, supervised learning, "
                       "decision trees, bias)?",
        "anchors": {
            1: "No module-specific content referenced.",
            2: "Mentions a module term but uses it incorrectly or without context.",
            3: "Correctly references one module concept with a basic connection "
               "to what was learned.",
            4: "Correctly references two or more module concepts and shows how "
               "they connect to each other or to the activity.",
        },
        "max_score": 4,
    },
    {
        "id":          "personal_connection",
        "label":       "Personal connection",
        "description": "Does the student connect the AI concept to their own "
                       "experience, prior knowledge, or everyday life? "
                       "Does the response feel personally engaged rather than "
                       "generic?",
        "anchors": {
            1: "No personal connection. Generic or copied-sounding response.",
            2: "Brief personal reference but superficial "
               "(e.g. 'it was fun' or 'I liked it').",
            3: "Makes a specific personal connection to an activity, "
               "a prior belief, or everyday technology.",
            4: "Makes a specific personal connection AND reflects on how "
               "their thinking or understanding has changed.",
        },
        "max_score": 4,
    },
]

# Derived constants — used by engine for normalization
CPI_QUAL_MAX_PER_QUESTION: int = sum(d["max_score"] for d in CPI_QUAL_DIMENSIONS)  # 12
CPI_QUAL_DIMENSION_IDS:    List[str] = [d["id"] for d in CPI_QUAL_DIMENSIONS]

# -----------------------------------------------------------------------
# Per-question rubric context (appended to prompt)
# -----------------------------------------------------------------------

_QUESTION_CONTEXT: Dict[str, str] = {
    "conceptual_change": (
        "This reflection asks the student to identify something new they "
        "now understand about AI that they could not explain before the module. "
        "They may include an example. Focus on whether they demonstrate "
        "a genuine conceptual shift, not just knowledge acquisition."
    ),
    "module_takeaway": (
        "This reflection asks the student to summarize what the module taught "
        "them about AI. They may comment on difficulty, clarity, or whether "
        "activities helped. Focus on accuracy and depth of the summary, "
        "not on their opinion of the module design."
    ),
}

# -----------------------------------------------------------------------
# System prompt
# -----------------------------------------------------------------------

CPI_QUAL_SYSTEM_PROMPT: str = (
    "You are an educational assessment assistant supporting the Basics4AI "
    "programme — a 7-module AI literacy curriculum for young people aged "
    "10–14 years. Your role is to score student reflection responses against "
    "a structured rubric. You must be fair, consistent, and calibrate "
    "expectations appropriately for this age group: a score of 4 represents "
    "excellent work for a 12-year-old, not for a university student. "
    "Score strictly from the evidence in the text. Do not reward students "
    "for length, and do not penalize short responses that are genuinely "
    "insightful. Return ONLY valid JSON — no preamble, no explanation outside "
    "the JSON structure."
)

# -----------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------

def build_cpi_qual_prompt(
    participant_id: str,
    question_id: str,
    text: str,
    module_id: str,
    module_title: str = "",
) -> str:
    """
    Build the scoring prompt for one reflection response.

    Parameters
    ----------
    participant_id : str
    question_id    : str — "conceptual_change" | "module_takeaway"
    text           : str — raw student response text
    module_id      : str — e.g. "module_1"
    module_title   : str — optional, improves context

    Returns
    -------
    str : prompt ready for call_model(model, prompt, system=CPI_QUAL_SYSTEM_PROMPT)
    """
    question_context = _QUESTION_CONTEXT.get(
        question_id,
        "This is a module reflection from a Basics4AI participant.",
    )

    # Build rubric block
    rubric_lines = []
    for dim in CPI_QUAL_DIMENSIONS:
        rubric_lines.append(f"\n### Dimension: {dim['label']}")
        rubric_lines.append(f"Description: {dim['description']}")
        rubric_lines.append("Score anchors (1–4):")
        for score, anchor in dim["anchors"].items():
            rubric_lines.append(f"  {score}: {anchor}")

    rubric_text = "\n".join(rubric_lines)

    # Expected JSON shape
    dim_scores_example = ", ".join(
        f'"{d["id"]}": <integer 1-4>'
        for d in CPI_QUAL_DIMENSIONS
    )

    prompt = f"""\
You are scoring a student reflection from the Basics4AI programme.

PARTICIPANT ID: {participant_id}
MODULE: {module_id}{f" — {module_title}" if module_title else ""}
QUESTION TYPE: {question_id}

QUESTION CONTEXT:
{question_context}

STUDENT RESPONSE:
{text.strip()}

SCORING RUBRIC:
{rubric_text}

INSTRUCTIONS:
Score the student response on each dimension using the 1-4 anchors above.
Remember: calibrate for ages 10-14. A score of 4 is achievable by an engaged young person.
If the response is blank, all scores must be 1.
Quote no more than one short phrase from the text as justification.

Return ONLY valid JSON with this exact structure:
{{
  "participant_id": "{participant_id}",
  "question_id": "{question_id}",
  "scores": {{{dim_scores_example}}},
  "justification": "<1-2 sentences explaining the overall scores>"
}}

Rules:
- participant_id must be exactly "{participant_id}" — do not alter it
- question_id must be exactly "{question_id}" — do not alter it
- Every score must be an integer 1, 2, 3, or 4
- Return ONLY the JSON object, no other text"""

    return prompt


# -----------------------------------------------------------------------
# Response parser
# -----------------------------------------------------------------------

def parse_cpi_qual_response(
    raw_text: str,
    participant_id: str,
    question_id: str,
) -> Dict[str, Any]:
    """
    Parse LLM JSON response into a validated score dict.

    Returns
    -------
    dict with keys:
        participant_id, question_id,
        scores (dict: dim_id -> int),
        justification (str),
        parse_error (str | None)
    """
    import json
    import re

    result: Dict[str, Any] = {
        "participant_id": participant_id,
        "question_id":    question_id,
        "scores":         {dim["id"]: 1 for dim in CPI_QUAL_DIMENSIONS},
        "justification":  "",
        "parse_error":    None,
    }

    if not raw_text:
        result["parse_error"] = "Empty LLM response."
        return result

    # Strip markdown fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [ln for ln in lines[1:] if ln.strip() != "```"]
        text  = "\n".join(inner).strip()

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        result["parse_error"] = f"No JSON object found in response: {text[:200]}"
        return result

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        result["parse_error"] = f"JSON decode error: {e}"
        return result

    # Extract scores
    raw_scores = parsed.get("scores", {})
    validated  = {}
    for dim in CPI_QUAL_DIMENSIONS:
        did = dim["id"]
        val = raw_scores.get(did, 1)
        try:
            val = int(val)
        except (TypeError, ValueError):
            val = 1
        validated[did] = max(1, min(4, val))  # clamp to [1, 4]
    result["scores"]        = validated
    result["justification"] = str(parsed.get("justification", ""))[:500]
    return result
