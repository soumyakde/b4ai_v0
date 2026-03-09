"""
modules/registry/validators.py 
Passive validators for module and instrument definitions.

This module is intentionally NON-BLOCKING:
- It logs warnings only
- It never raises exceptions
- It never mutates input data

It exists to catch contract drift early while preserving
full backward compatibility with existing registry code.
"""

import logging
from typing import Dict, Any


# Module-level logger so logging can be configured centrally
logger = logging.getLogger(__name__)


# ============================================================
# Core instrument-level validator
# ============================================================
def validate_instruments(module_id: str, instruments: Dict[str, Any]) -> None:
    """
    Passive validation of instrument definitions.

    Parameters
    ----------
    module_id : str
        Identifier of the learning unit (e.g. "pre_course", "module1")

    instruments : dict
        Flexible expected shape:

        {
            "surveys": dict | list,
            "assessments": dict | list,
            "reflections": dict | list (optional)
        }

    Notes
    -----
    - Logs warnings only
    - Never raises
    - No side effects
    """

    def _iter_items(section):
        """
        Normalize dicts and lists into iterable form so
        legacy and canonical definitions both work.
        """
        if isinstance(section, dict):
            return section.items()
        if isinstance(section, list):
            return enumerate(section)
        return []

    # --------------------
    # Survey validation
    # --------------------
    for _, survey in _iter_items(instruments.get("surveys", {})):
        instrument_type = survey.get("instrument_type", "survey")
        scope = survey.get("scope")

        if instrument_type != "survey":
            logger.warning(
                "[InstrumentValidator] %s: survey declared as '%s'",
                module_id,
                instrument_type,
            )

        if module_id == "pre_course" and scope != "pre_course":
            logger.warning(
                "[InstrumentValidator] %s: survey has invalid scope '%s'",
                module_id,
                scope,
            )

    # --------------------
    # Assessment validation
    # --------------------
    for _, assessment in _iter_items(instruments.get("assessments", {})):
        instrument_type = assessment.get("instrument_type", "assessment")
        scope = assessment.get("scope")

        if instrument_type != "assessment":
            logger.warning(
                "[InstrumentValidator] %s: assessment declared as '%s'",
                module_id,
                instrument_type,
            )

        if module_id == "pre_course" and scope != "pre_course":
            logger.warning(
                "[InstrumentValidator] %s: assessment has invalid scope '%s'",
                module_id,
                scope,
            )

    # --------------------
    # Reflection validation
    # --------------------
    for _, _ in _iter_items(instruments.get("reflections", {})):
        if module_id == "pre_course":
            logger.warning(
                "[InstrumentValidator] %s: reflections are not allowed in pre-course",
                module_id,
            )


# ============================================================
# Module-level validator (BACKWARD COMPATIBILITY)
# ============================================================
def validate_module_definition(defn: Dict[str, Any]) -> None:
    """
    Backward-compatible module definition validator.

    This function exists because registry/bootstrap code
    already imports and calls it.

    Behavior:
    ---------
    - Extracts module_id and runtime instruments
    - Delegates validation to validate_instruments
    - Logs warnings only
    - Never raises

    Expected minimal shape of defn:
    {
        "meta": {
            "module_id": str
        },
        "runtime": {
            ...
        }
    }
    """

    if not isinstance(defn, dict):
        logger.warning(
            "[ModuleValidator] Invalid module definition type: %s",
            type(defn),
        )
        return

    meta = defn.get("meta", {})
    module_id = meta.get("module_id", "unknown_module")
    instruments = defn.get("runtime", {})

    validate_instruments(module_id=module_id, instruments=instruments)