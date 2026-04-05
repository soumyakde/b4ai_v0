"""
utils/quiz_mode.py
Quiz Mode Authority

Responsibilities:
- Read and expose the active quiz mode from the environment
- Provide per-module research question ID lists
- Single, auditable source of truth for all mode decisions

This module has:
- No DB access
- No Streamlit
- No file I/O beyond os.environ

Quiz Modes:
-----------
  "random"    — Current behaviour. Each session draws a fresh random
                sample of N questions from the full question bank.
                DEFAULT — safe fallback if env var is absent/misspelled.

  "research"  — Fixed set. Every student always sees the SAME questions
                for every module. Which questions are chosen is controlled
                per-module via QUIZ_QUESTION_IDS_MODULE_N (Sub-strategy A),
                or falls back to the first N questions in the JSON file
                (Sub-strategy B — simplest, no extra config required).

.env configuration reference:
------------------------------
  # Global switch
  QUIZ_MODE=research          # "random" | "research"

  # Per-module question ID lists  (only used when QUIZ_MODE=research)
  # IDs must exactly match the "id" field in each module's question bank.
  # If a module's variable is absent or blank, loader falls back to first-N.
  QUIZ_QUESTION_IDS_MODULE_1=Q1,Q5,Q10,Q15,Q20,Q25,Q30,Q35,Q40,Q45
  QUIZ_QUESTION_IDS_MODULE_2=Q1,Q5,Q10,Q15,Q20,Q25,Q30,Q35,Q38,Q40
  QUIZ_QUESTION_IDS_MODULE_3=Q1,Q5,Q10,Q15,Q20,Q22,Q24,Q27,Q28,Q30
  QUIZ_QUESTION_IDS_MODULE_4=Q1,Q3,Q5,Q7,Q9,Q11,Q13,Q15,Q17,Q19
  QUIZ_QUESTION_IDS_MODULE_5=Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10
  QUIZ_QUESTION_IDS_MODULE_6=Q1,Q3,Q5,Q7,Q9,Q10,Q11,Q14,Q16,Q18
  QUIZ_QUESTION_IDS_MODULE_7=Q1,Q3,Q5,Q7,Q9,Q10,Q11,Q12,Q13,Q15

Env key convention (deterministic, no manual mapping needed):
  module_id constant → upper()
  "module_1"  → "QUIZ_QUESTION_IDS_MODULE_1"
  "module_7"  → "QUIZ_QUESTION_IDS_MODULE_7"
"""

import os


# ------------------------------------------------------------------
# Internal constants
# ------------------------------------------------------------------
_ENV_KEY_MODE = "QUIZ_MODE"
_VALID_MODES  = {"random", "research"}
_DEFAULT_MODE = "random"

# Prefix for per-module ID vars: QUIZ_QUESTION_IDS_ + MODULE_ID.upper()
_ID_PREFIX = "QUIZ_QUESTION_IDS_"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_quiz_mode() -> str:
    """
    Return the active quiz mode: "random" or "research".

    Reads QUIZ_MODE from the environment.  Falls back to "random" for
    any absent or unrecognised value — the system is always safe even
    if .env is misconfigured.
    """
    raw = os.environ.get(_ENV_KEY_MODE, _DEFAULT_MODE).strip().lower()
    return raw if raw in _VALID_MODES else _DEFAULT_MODE


def is_research_mode() -> bool:
    """Convenience predicate — True when QUIZ_MODE=research."""
    return get_quiz_mode() == "research"


def get_research_question_ids(module_id: str) -> list[str] | None:
    """
    Return the explicit question ID list for a specific module in
    research mode, or None if not configured (caller falls back to
    first-N ordering).

    Args:
        module_id: The MODULE_ID constant from the renderer, e.g.
                   "module_1", "module_2", ... "module_7".

    Returns:
        list[str]  — ordered IDs as specified in .env, or
        None       — if the env var is absent or blank

    Env var derivation from module_id (automatic, no manual mapping):
        "module_1"  →  QUIZ_QUESTION_IDS_MODULE_1
        "module_7"  →  QUIZ_QUESTION_IDS_MODULE_7
    """
    env_key = f"{_ID_PREFIX}{module_id.upper()}"
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return None
    ids = [qid.strip() for qid in raw.split(",") if qid.strip()]
    return ids if ids else None
