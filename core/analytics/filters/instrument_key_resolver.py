# core/analytics/filters/instrument_key_resolver.py
"""
Instrument Key Resolver
=======================
Translates DB instrument_name values into canonical YAML instrument_key
values for use by the analytics pipeline ONLY.

Architecture rules:
- Used exclusively by core/analytics/ modules
- Never imported by student_dashboard, admin_dashboard, or submission_engine
- No DB access, no Streamlit, no side effects
- Pure deterministic function

Problem this solves:
--------------------
submission_engine.py builds DB instrument_name as:
    module_prefix + "_" + instrument_key

Examples:
    "module1_b4ai_sccces_survey"          module1_ prefix added
    "module1_b4ai_sims_survey"            module1_ prefix added
    "precourse_demographics_survey"       precourse_ prefix added
    "pre_ai_misconceptions_assessment"    no prefix, already canonical
    "post_ai_misconceptions_assessment"   no prefix, already canonical
    "pre_aici_assessment"                 no prefix, already canonical
    "post_aici_assessment"                no prefix, already canonical
    "module1_content_mcq_assessment"      no prefix, already canonical

Resolution rules (first match wins):
--------------------------------------
Rule 1  Survey suffix match  ->  strip any module prefix
        module{N}_b4ai_sccces_survey  ->  b4ai_sccces_survey
        module{N}_b4ai_sims_survey    ->  b4ai_sims_survey

Rule 2  Demographics suffix match  ->  strip context prefix
        precourse_demographics_survey ->  demographics_survey

Rule 3  Identity pass-through  ->  return unchanged
        pre/post assessments, module MCQs all pass through.

Rules 1 and 2 use suffix matching so new modules (module2_, module3_...)
resolve automatically without changes to this file.
"""

from typing import FrozenSet

# -----------------------------------------------------------------------
# Survey base keys whose module{N}_ prefix must be stripped.
# Extend this set when new survey instruments are created.
# -----------------------------------------------------------------------
_SURVEY_BASE_KEYS: FrozenSet[str] = frozenset([
    "b4ai_sccces_survey",
    "b4ai_sims_survey",
])

# -----------------------------------------------------------------------
# Context prefixes applied by submission_engine to assessments.
# These are stripped to recover the canonical YAML instrument_key.
#   precourse_pre_ai_misconceptions_assessment -> pre_ai_misconceptions_assessment
#   postcourse_post_ai_misconceptions_assessment -> post_ai_misconceptions_assessment
#   precourse_pre_aici_assessment              -> pre_aici_assessment
#   postcourse_post_aici_assessment            -> post_aici_assessment
# -----------------------------------------------------------------------
_ASSESSMENT_CONTEXT_PREFIXES = ("precourse_", "postcourse_")


def resolve_instrument_key(db_instrument_name: str) -> str:
    """
    Translate a DB instrument_name to its canonical YAML instrument_key.

    Parameters
    ----------
    db_instrument_name : str
        Raw instrument_name from the responses table.

    Returns
    -------
    str
        Canonical instrument_key for analytics use.

    Examples
    --------
    >>> resolve_instrument_key("module1_b4ai_sccces_survey")
    'b4ai_sccces_survey'
    >>> resolve_instrument_key("module2_b4ai_sims_survey")
    'b4ai_sims_survey'
    >>> resolve_instrument_key("precourse_demographics_survey")
    'demographics_survey'
    >>> resolve_instrument_key("precourse_pre_ai_misconceptions_assessment")
    'pre_ai_misconceptions_assessment'
    >>> resolve_instrument_key("postcourse_post_ai_misconceptions_assessment")
    'post_ai_misconceptions_assessment'
    >>> resolve_instrument_key("precourse_pre_aici_assessment")
    'pre_aici_assessment'
    >>> resolve_instrument_key("postcourse_post_aici_assessment")
    'post_aici_assessment'
    >>> resolve_instrument_key("module1_content_mcq_assessment")
    'module1_content_mcq_assessment'
    """
    name = db_instrument_name.strip()

    # Rule 1: survey base key suffix match (handles all module prefixes)
    for base_key in _SURVEY_BASE_KEYS:
        if name == base_key or name.endswith("_" + base_key):
            return base_key

    # Rule 2: assessment context prefix strip
    #   precourse_pre_*  ->  pre_*
    #   postcourse_post_* ->  post_*
    for prefix in _ASSESSMENT_CONTEXT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]

    # Rule 3: demographics suffix match
    if name == "demographics_survey" or name.endswith("_demographics_survey"):
        return "demographics_survey"

    # Rule 4: identity pass-through
    return name


def resolve_instrument_keys_series(series) -> "pd.Series":
    """
    Vectorized resolver for a pandas Series column.

    Parameters
    ----------
    series : pd.Series
        Series of DB instrument_name strings.

    Returns
    -------
    pd.Series
        Series of resolved canonical instrument_key strings.
    """
    return series.map(resolve_instrument_key)


def get_survey_base_keys() -> FrozenSet[str]:
    """Return the set of known survey base keys."""
    return _SURVEY_BASE_KEYS
