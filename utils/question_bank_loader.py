"""
utils/question_bank_loader.py
Question Bank Loader Utility

Responsibilities:
- Load a question bank JSON file (path as supplied by assessment_def)
- Apply quiz-mode-aware question selection
- Return a pure list of question dicts ready for the renderer

This module has:
- No DB access
- No Streamlit
- No resolver logic
- No scoring logic

Design mirrors yaml_loader.py:
- Fail fast on missing files
- Fail fast on malformed JSON
- Pure function: same inputs always produce the same outputs (research mode)
                 or a random sample (random mode)

Expected JSON structure for every question bank file:
-----------------------------------------------------
  [
    {
      "id":       "Q1",
      "question": "...",
      "options":  ["A text", "B text", "C text", "D text"],
      "answer":   "A"          ← optional; letter of correct option
    },
    ...
  ]
  Root must be a list of dicts, each with at minimum an "id" field.

Question bank sizes (for researcher reference):
  module_1  — 57 questions  (Q1–Q57)
  module_2  — 40 questions  (Q1–Q40)
  module_3  — 30 questions  (Q1–Q30)
  module_4  — 19 questions  (Q1–Q19)
  module_5  — 15 questions  (Q1–Q15)
  module_6  — 18 questions  (Q1–Q18)
  module_7  — 15 questions  (Q1–Q15)

Selection strategy (governed by QUIZ_MODE in .env):
----------------------------------------------------
  QUIZ_MODE=random   (DEFAULT)
    → random.sample(bank, n)
      Preserves the original behaviour exactly.

  QUIZ_MODE=research
    Sub-strategy A — explicit IDs  (QUIZ_QUESTION_IDS_MODULE_N set in .env)
      → Returns questions whose "id" matches the list, in the researcher's
        specified order.  Unknown IDs are logged and skipped.

    Sub-strategy B — first-N  (QUIZ_QUESTION_IDS_MODULE_N absent / blank)
      → Returns bank[:n].
        The researcher controls which questions appear by ordering them
        first in the JSON file. Zero extra config required.

Usage (drop-in replacement for the load + if/else block in module_N.py):
------------------------------------------------------------------------
  # Before (module_1.py):
  with open(question_bank_path, "r", encoding="utf-8") as f:
      question_bank = json.load(f)
  if randomize:
      questions = random.sample(question_bank, min(num_questions, len(question_bank)))
  else:
      questions = question_bank[:num_questions]

  # After:
  from utils.question_bank_loader import load_question_bank
  questions = load_question_bank(question_bank_path, MODULE_ID, num_questions)

  Note: the 'randomize' field in assessment_def is superseded by QUIZ_MODE.
        In research mode the loader always returns a fixed set regardless of
        what 'randomize' says.  In random mode it always samples randomly.
"""

import json
import os
import random
import logging
from typing import Any

from utils.quiz_mode import get_quiz_mode, get_research_question_ids


log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_question_bank(
    question_bank_path: str,
    module_id: str,
    n: int = 10,
) -> list[dict[str, Any]]:
    """
    Load a question bank JSON file and return a mode-selected list.

    The path is used exactly as supplied — matching how module_N.py
    already opens the file — so no path translation is needed.

    Args:
        question_bank_path:
            Path to the JSON file, exactly as it appears in
            assessment_def["question_bank_path"].
            Can be absolute or relative to the working directory.

        module_id:
            The MODULE_ID constant from the calling renderer, e.g.
            "module_1".  Used to look up QUIZ_QUESTION_IDS_MODULE_1
            when QUIZ_MODE=research.

        n:
            Number of questions to return (default 10).
            Ignored when research mode uses explicit IDs — all listed
            IDs are returned regardless of n.

    Returns:
        list[dict]: Selected question dicts, ordered and ready to render.

    Raises:
        FileNotFoundError: Question bank JSON does not exist at path.
        ValueError:        JSON is malformed or root is not a list.
    """

    # ── Load ────────────────────────────────────────────────────────
    if not os.path.exists(question_bank_path):
        raise FileNotFoundError(
            f"Question bank not found: {question_bank_path}"
        )

    try:
        with open(question_bank_path, "r", encoding="utf-8") as f:
            bank = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in question bank '{question_bank_path}': {e}"
        )

    if not isinstance(bank, list):
        raise ValueError(
            f"Question bank root must be a list of dicts: {question_bank_path}"
        )

    if not bank:
        raise ValueError(f"Question bank is empty: {question_bank_path}")

    # ── Select ──────────────────────────────────────────────────────
    mode = get_quiz_mode()

    if mode == "research":
        return _select_research(bank, module_id, n, question_bank_path)
    else:
        return _select_random(bank, n, question_bank_path)


# ------------------------------------------------------------------
# Private selection strategies
# ------------------------------------------------------------------

def _select_random(
    bank: list[dict], n: int, path: str
) -> list[dict]:
    """
    Random mode — preserves the original behaviour exactly.
    Equivalent to: random.sample(bank, min(n, len(bank)))
    """
    sample_size = min(n, len(bank))
    if sample_size < n:
        log.warning(
            "Question bank '%s' has only %d questions; requested %d.",
            path, len(bank), n,
        )
    return random.sample(bank, sample_size)


def _select_research(
    bank: list[dict], module_id: str, n: int, path: str
) -> list[dict]:
    """
    Research mode dispatcher.
    Calls Sub-strategy A (explicit IDs) or B (first-N).
    """
    explicit_ids = get_research_question_ids(module_id)

    if explicit_ids:
        return _select_by_ids(bank, explicit_ids, module_id, n, path)
    else:
        return _select_first_n(bank, n, module_id, path)


def _select_by_ids(
    bank: list[dict],
    ids: list[str],
    module_id: str,
    n: int,
    path: str,
) -> list[dict]:
    """
    Sub-strategy A: return questions matching the researcher's ID list,
    in the exact order the researcher specified in .env.

    Unknown IDs are logged and skipped rather than raising an error,
    so a typo in .env degrades gracefully instead of crashing a session.
    If NO valid IDs are found at all, falls back to first-N and logs
    an error so the researcher sees it immediately.
    """
    index: dict[str, dict] = {q.get("id", ""): q for q in bank}

    selected = []
    for qid in ids:
        if qid in index:
            selected.append(index[qid])
        else:
            log.warning(
                "Research mode [%s]: question ID '%s' not found in '%s'. "
                "Check QUIZ_QUESTION_IDS_%s in .env.",
                module_id, qid, path, module_id.upper(),
            )

    if not selected:
        log.error(
            "Research mode [%s]: no valid IDs matched in '%s'. "
            "Falling back to first-%d questions. "
            "Fix QUIZ_QUESTION_IDS_%s in .env.",
            module_id, path, n, module_id.upper(),
        )
        return _select_first_n(bank, n, module_id, path)

    return selected


def _select_first_n(
    bank: list[dict], n: int, module_id: str, path: str
) -> list[dict]:
    """
    Sub-strategy B: return the first N questions in the JSON file.

    The researcher controls which questions are included by placing
    them first in the JSON file. No .env variable needed — the
    simplest research setup, fully version-controllable.
    """
    if len(bank) < n:
        log.warning(
            "Research mode [%s]: bank '%s' has only %d questions; "
            "requested first %d. Returning all %d.",
            module_id, path, len(bank), n, len(bank),
        )
    return bank[:n]
