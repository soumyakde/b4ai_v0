# ==========================================================
# Post-Course Diagnostic Module Definition (v6 Compatible)
# Pure declarative metadata — no executable logic
# ==========================================================

MODULE_DEFINITION = {

    # ======================================================
    # Core Metadata
    # ======================================================
    "meta": {
        "module_id": "post_course",
        "title": "Post-Course Survey and Assessments",
        "version": "1.0.0",
        "description": "Post-instruction surveys and assessments.",
        "author": "Soumya De",
        "status": "active",
        "order": 999,  # ✅ Ensures it appears after instructional modules
    },

    # ======================================================
    # Pedagogical Metadata
    # ======================================================
    "pedagogy": {
        "learning_objectives": [
            "Measure post-instruction misconceptions",
            "Assess change in AI conceptual understanding",
        ],
        "prerequisite_knowledge": "Completion of course modules",
        "estimated_time_minutes": 25,
    },

    # ======================================================
    # Instruments (v6 Contract Structure)
    # ======================================================
    "instruments": {

        # ❌ No demographic survey in post-course

        # ✅ Assessments
        "assessments": {
            "post_ai_misconceptions": {
                "assessment_code": "post_ai_misconceptions_assessment",
                "instrument_type": "assessment",
                "scope": "post_course",
                "storage": "yaml",
            },
            "post_aici": {
                "assessment_code": "post_aici_assessment",
                "instrument_type": "assessment",
                "scope": "post_course",
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
    # "legacy": {},
}