# ==========================================================
# Module 7 Definition (v6 Compatible)
# Pure declarative metadata — no executable logic
# ==========================================================

from modules.definitions.shared_reflection_questions import REFLECTION_QUESTIONS

MODULE_DEFINITION = {

    # ======================================================
    # Core Metadata
    # ======================================================
    "meta": {
        "module_id": "module_7",
        "title": "Introduction to Machine Learning (ML)",
        "version": "1.0.0",
        "description": "Teaching computers to learn like you.",
        "author": "Soumya De",
        "status": "disabled",  # Paused for this pilot only — flip back to "active" to re-enable
        "order": 7,
    },

    # ======================================================
    # Pedagogical Metadata
    # ======================================================
    "pedagogy": {
        "learning_objectives": [
            "Teach a computer to read hand gestures with least uncertainty.",
            "Identify and understand the constraints of teaching an AI model.",
            "Implement ‘Supervised Learning' in building a game.",
        ],
        "prerequisite_knowledge": "Module 6 completion",
        "estimated_time_minutes": 60,
    },

    # ======================================================
    # Instruments
    # ======================================================
    "instruments": {

        "assessments": {

            "content_mcq": {
                "assessment_code": "module7_content_mcq_assessment",
                "instrument_type": "assessment",
                "scope": "module_7",
                "storage": "json",
                "question_bank_path": "content_dev/module7_question_bank.json",
                "num_questions": 10,
                "randomize": True,
            },

            "reflection": {
                "assessment_code": "module_reflections",
                "instrument_type": "assessment",
                "type": "open_response",
                "scope": "module_7",
                "questions": REFLECTION_QUESTIONS,
            },
        },

        "surveys": {
            "sccces": {
                "survey_key": "b4ai_sccces_survey",
                "instrument_type": "survey",
                "scope": "module_7",
            },
            "sims": {
                "survey_key": "b4ai_sims_survey",
                "instrument_type": "survey",
                "scope": "module_7",
            },
        },
    },

    # ======================================================
    # Evaluation Behavior
    # ======================================================
    "evaluation": {
        "deterministic_first": True,
        "ai_explanation_enabled": True,
        "explanation_guidelines": "Reference logic in analysis/llm_analysis.py",
    },

    # ======================================================
    # UI Configuration
    # ======================================================
    "ui": {
        "question_order": "fixed",
        "section_labels": None,
        "show_rationale_immediately": False,
    },

    # ======================================================
    # Platform Constraints
    # ======================================================
    "constraints": {
        "allowed_question_types": ["mcq", "open_response"],
        "max_attempts": None,
        "time_limit_minutes": None,
    },
}