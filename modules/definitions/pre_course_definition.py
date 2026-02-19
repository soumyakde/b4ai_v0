# ==========================================================
# Pre-Course Diagnostic Module Definition (v6 Compatible)
# Pure declarative metadata — no executable logic
# ==========================================================

MODULE_DEFINITION = {

    # ======================================================
    # Core Metadata
    # ======================================================
    "meta": {
        "module_id": "pre_course",
        "title": "Pre-Course Survey and Assessments",
        "version": "1.0.0",
        "description": "Baseline surveys and assessments before instruction.",
        "author": "Soumya De",
        "status": "active",
        "order": 0,  # ✅ Ensures it appears before Module 1
    },

    # ======================================================
    # Pedagogical Metadata
    # ======================================================
    "pedagogy": {
        "learning_objectives": [
            "Capture baseline student attitudes toward AI",
            "Measure pre-instruction misconceptions",
        ],
        "prerequisite_knowledge": "None",
        "estimated_time_minutes": 30,
    },

    # ======================================================
    # Instruments (v6 Contract Structure)
    # ======================================================
    "instruments": {

        # ✅ Surveys
        "surveys": {
            "demographic": {
                "survey_key": "demographics_survey", #name changed to see if it loads
                "instrument_type": "survey",
                "scope": "pre_course",
                "storage": "yaml",
            },
        },

        # ✅ Assessments
        "assessments": {
            "pre_ai_misconceptions": {
                "assessment_code": "pre_ai_misconceptions_assessment",
                "instrument_type": "assessment",
                "scope": "pre_course",
                "storage": "yaml",
            },
            "pre_aici": {
                "assessment_code": "pre_aici_assessment",
                "instrument_type": "assessment",
                "scope": "pre_course",
                "storage": "yaml",
            },
        },
    },

    # ======================================================
    # UI Configuration
    # ======================================================
    "ui": {
        "show_progress": True,
    },

    # ======================================================
    # Legacy (kept for compatibility)
    # ======================================================
    #"legacy": {},
}