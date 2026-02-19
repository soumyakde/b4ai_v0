"""
Strict v6 — Phase 1 Contract Validator

This validator operates on normalized module definitions
(after adapt_module_definition).

It is:
- Pure (no side effects)
- Non-mutating
- Deterministic
- Environment-agnostic

It returns structured validation errors.
It does NOT raise or log.
"""


from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================================
# Error Model
# ============================================================

@dataclass(frozen=True)
class ValidationError:
    module_id: Optional[str]
    location: str
    message: str
    severity: str = "ERROR"


# ============================================================
# Public Entry Point
# ============================================================

def validate_module_v6(module: Dict[str, Any]) -> List[ValidationError]:
    """
    Validate a normalized module definition against
    Strict v6 Phase 1 contract.

    Returns:
        List[ValidationError]
    """

    errors: List[ValidationError] = []

    # --------------------------------------------------------
    # Top-Level Structure
    # --------------------------------------------------------

    if not isinstance(module, dict):
        errors.append(
            ValidationError(
                module_id=None,
                location="module",
                message="Module must be a dictionary.",
            )
        )
        return errors  # Fatal — cannot continue

    module_id = _safe_extract_module_id(module)

    # Required top-level keys
    errors += _validate_required_dict(module, "meta", module_id)
    errors += _validate_required_dict(module, "content", module_id)
    errors += _validate_required_dict(module, "instruments", module_id)

    # --------------------------------------------------------
    # META
    # --------------------------------------------------------

    meta = module.get("meta")
    if isinstance(meta, dict):
        errors += _validate_meta(meta, module_id)

    # --------------------------------------------------------
    # CONTENT
    # --------------------------------------------------------

    # Already validated as dict if no top-level error
    # No deep validation in Phase 1

    # --------------------------------------------------------
    # INSTRUMENTS
    # --------------------------------------------------------

    instruments = module.get("instruments")
    if isinstance(instruments, dict):

        # Validate container keys
        errors += _validate_required_dict(instruments, "surveys", module_id, parent="instruments")
        errors += _validate_required_dict(instruments, "assessments", module_id, parent="instruments")
        errors += _validate_required_dict(instruments, "labs", module_id, parent="instruments")

        surveys = instruments.get("surveys")
        if isinstance(surveys, dict):
            errors += _validate_surveys(surveys, module_id)

        assessments = instruments.get("assessments")
        if isinstance(assessments, dict):
            errors += _validate_assessments(assessments, module_id)

        # labs: container only in Phase 1 (no deep validation)

    return errors


# ============================================================
# Internal Helpers
# ============================================================

def _safe_extract_module_id(module: Dict[str, Any]) -> Optional[str]:
    meta = module.get("meta")
    if isinstance(meta, dict):
        module_id = meta.get("module_id")
        if isinstance(module_id, str) and module_id.strip():
            return module_id.strip()
    return None


def _validate_required_dict(
    container: Dict[str, Any],
    key: str,
    module_id: Optional[str],
    parent: Optional[str] = None,
) -> List[ValidationError]:

    errors: List[ValidationError] = []

    location = f"{parent}.{key}" if parent else key

    if key not in container:
        errors.append(
            ValidationError(
                module_id=module_id,
                location=location,
                message="Missing required field.",
            )
        )
        return errors

    if not isinstance(container[key], dict):
        errors.append(
            ValidationError(
                module_id=module_id,
                location=location,
                message="Must be a dictionary.",
            )
        )

    return errors


# ============================================================
# META VALIDATION
# ============================================================

def _validate_meta(meta: Dict[str, Any], module_id: Optional[str]) -> List[ValidationError]:

    errors: List[ValidationError] = []

    # module_id
    if "module_id" not in meta:
        errors.append(
            ValidationError(
                module_id=module_id,
                location="meta.module_id",
                message="Missing required field.",
            )
        )
    else:
        value = meta.get("module_id")
        if not isinstance(value, str) or not value.strip():
            errors.append(
                ValidationError(
                    module_id=module_id,
                    location="meta.module_id",
                    message="Must be a non-empty string.",
                )
            )

    # title
    if "title" not in meta:
        errors.append(
            ValidationError(
                module_id=module_id,
                location="meta.title",
                message="Missing required field.",
            )
        )
    else:
        value = meta.get("title")
        if not isinstance(value, str) or not value.strip():
            errors.append(
                ValidationError(
                    module_id=module_id,
                    location="meta.title",
                    message="Must be a non-empty string.",
                )
            )

    # order
    if "order" not in meta:
        errors.append(
            ValidationError(
                module_id=module_id,
                location="meta.order",
                message="Missing required field.",
            )
        )
    else:
        value = meta.get("order")
        if not isinstance(value, int) or value < 0:
            errors.append(
                ValidationError(
                    module_id=module_id,
                    location="meta.order",
                    message="Must be an integer >= 0.",
                )
            )

    return errors


# ============================================================
# SURVEY VALIDATION
# ============================================================

def _validate_surveys(
    surveys: Dict[str, Any],
    module_id: Optional[str],
) -> List[ValidationError]:

    errors: List[ValidationError] = []

    for survey_name in sorted(surveys.keys()):
        survey = surveys[survey_name]
        location_prefix = f"instruments.surveys.{survey_name}"

        if not isinstance(survey, dict):
            errors.append(
                ValidationError(
                    module_id=module_id,
                    location=location_prefix,
                    message="Survey definition must be a dictionary.",
                )
            )
            continue

        if "survey_key" not in survey:
            errors.append(
                ValidationError(
                    module_id=module_id,
                    location=f"{location_prefix}.survey_key",
                    message="Missing required field.",
                )
            )
        else:
            value = survey.get("survey_key")
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    ValidationError(
                        module_id=module_id,
                        location=f"{location_prefix}.survey_key",
                        message="Must be a non-empty string.",
                    )
                )

    return errors


# ============================================================
# ASSESSMENT VALIDATION
# ============================================================

def _validate_assessments(
    assessments: Dict[str, Any],
    module_id: Optional[str],
) -> List[ValidationError]:

    errors: List[ValidationError] = []

    for assessment_name in sorted(assessments.keys()):
        assessment = assessments[assessment_name]
        location_prefix = f"instruments.assessments.{assessment_name}"

        if not isinstance(assessment, dict):
            errors.append(
                ValidationError(
                    module_id=module_id,
                    location=location_prefix,
                    message="Assessment definition must be a dictionary.",
                )
            )
            continue

        if "assessment_code" not in assessment:
            errors.append(
                ValidationError(
                    module_id=module_id,
                    location=f"{location_prefix}.assessment_code",
                    message="Missing required field.",
                )
            )
        else:
            value = assessment.get("assessment_code")
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    ValidationError(
                        module_id=module_id,
                        location=f"{location_prefix}.assessment_code",
                        message="Must be a non-empty string.",
                    )
                )

    return errors