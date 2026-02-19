# core/progress_engine.py

"""
progress_engine.py

Purpose:
--------
Deterministic derivation of:

- Instrument completion
- Module completion
- Sequential unlock state

Rules:
------
- No derived state stored
- No module completion flags stored
- Truth is always derived from completions table
- No Streamlit imports
- No UI logic
"""

from typing import Set, Dict
from core.db_utils import get_connection
from modules.resolution.learning_unit_resolver import LearningUnitResolver

resolver = LearningUnitResolver()

from typing import Any

def get_instrument_type(module: Dict[str, Any], instrument_key: str) -> str:
    """
    Determine whether an instrument_key belongs to:

        - "survey"
        - "assessment"

    Operates ONLY on provided module dict.
    Raises KeyError if instrument not found.
    """

    instruments_block = module.get("instruments", {})

    surveys = instruments_block.get("surveys", {})
    assessments = instruments_block.get("assessments", {})

    # Check surveys
    for survey in surveys.values():
        if survey.get("survey_key") == instrument_key:
            return "survey"

    # Check assessments
    for assessment in assessments.values():
        if assessment.get("assessment_code") == instrument_key:
            return "assessment"

    raise KeyError(
        f"Instrument '{instrument_key}' not found in module "
        f"'{module.get('meta', {}).get('module_id', 'unknown')}'."
    )


# ---------------------------------------------------------------------
# Instrument Completion
# ---------------------------------------------------------------------

def get_completed_instruments(user_id: str, module_id: str) -> Set[str]:
    """
    Returns completed instrument_keys for a specific user and module.
    
    IMPORTANT:
    ----------
    Completion is scoped to module to prevent cross-module leakage.
    """

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT instrument_key
            FROM completions
            WHERE user_id = ?
              AND module_id = ?
        """, (user_id, module_id)).fetchall()

    return {row[0] for row in rows}


def mark_instrument_complete(user_id: str, module_id: str, instrument_key: str):
    """
    Marks an instrument as complete for a user.

    Safety:
    -------
    - Verifies module exists
    - Verifies instrument belongs to module
    - Uses INSERT OR IGNORE to prevent duplication
    """

    if not resolver.exists(module_id):
        raise KeyError(f"Unknown module_id: {module_id}")

    module = resolver.get(module_id)
    required = get_required_instruments(module)

    if instrument_key not in required:
        raise ValueError(
            f"Instrument '{instrument_key}' not registered for module '{module_id}'"
        )

    with get_connection() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO completions
            (user_id, module_id, instrument_key, completed_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (user_id, module_id, instrument_key))
        conn.commit()


# ---------------------------------------------------------------------
# Module Derivation
# ---------------------------------------------------------------------

def get_required_instruments(module: Dict) -> Set[str]:
    """
    Extract all required instrument keys from module definition.

    Fail-fast if malformed.
    """

    instruments = set()

    instruments_block = module.get("instruments", {})

    surveys = instruments_block.get("surveys", {})
    assessments = instruments_block.get("assessments", {})

    try:
        for survey in surveys.values():
            instruments.add(survey["survey_key"])

        for assessment in assessments.values():
            instruments.add(assessment["assessment_code"])

    except KeyError as e:
        raise KeyError(f"Malformed module instrument definition: missing {e}")

    return instruments


def is_module_complete(user_id: str, module: Dict) -> bool:
    """
    A module is complete if:
        required instruments ⊆ completed instruments
    """

    required = get_required_instruments(module)

    # No auto-complete for empty modules
    if not required:
        return False

    completed = get_completed_instruments(
        user_id,
        module["meta"]["module_id"]
    )

    return required.issubset(completed)


# ---------------------------------------------------------------------
# Unlock Logic (Sequential)
# ---------------------------------------------------------------------

def is_module_unlocked(user_id: str, module_id: str) -> bool:
    """
    Sequential unlock rule:

    - First module is always unlocked
    - A module is unlocked if previous module is complete
    """

    modules = resolver.list_all()
    ids = [m["meta"]["module_id"] for m in modules]

    if module_id not in ids:
        return False

    index = ids.index(module_id)

    # First module always unlocked
    if index == 0:
        return True

    previous_module = modules[index - 1]

    return is_module_complete(user_id, previous_module)