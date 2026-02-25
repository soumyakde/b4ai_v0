# ==========================================================
# Module 5 Definition (v6 Compatible)
# Pure declarative metadata — no executable logic
# ==========================================================

from modules.definitions.shared_reflection_questions import REFLECTION_QUESTIONS

MODULE_DEFINITION = {

    # ======================================================
    # Core Metadata
    # ======================================================
    "meta": {
        "module_id": "module_5",
        "title": "Constraints in Problem-solving",
        "version": "1.0.0",
        "description": "Introduction to Problem-solving in the real-world.",
        "author": "Soumya De",
        "status": "active",
        "order": 5,
    },

    # ======================================================
    # Pedagogical Metadata
    # ======================================================
    "pedagogy": {
        "learning_objectives": [
            "Identify and explain constraints in any problem-solving scenario.",
            "Solve problems meeting all constraints.",
            "Identify and Explain Constraints in real-life.",
        ],
        "prerequisite_knowledge": "Module 4 completion",
        "estimated_time_minutes": 60,
    },

    # ======================================================
    # Instruments
    # ======================================================
    "instruments": {

        "assessments": {

            "content_mcq": {
                "assessment_code": "module5_content_mcq_assessment",
                "instrument_type": "assessment",
                "scope": "module_5",
                "storage": "json",
                "question_bank_path": "content_dev/module5_question_bank.json",
                "num_questions": 10,
                "randomize": True,
            },

            "reflection": {
                "assessment_code": "module_reflections",
                "instrument_type": "assessment",
                "type": "open_response",
                "scope": "module_5",
                "questions": REFLECTION_QUESTIONS,
            },
        },

        "surveys": {
            "sccces": {
                "survey_key": "b4ai_sccces_survey",
                "instrument_type": "survey",
                "scope": "module_5",
            },
            "sims": {
                "survey_key": "b4ai_sims_survey",
                "instrument_type": "survey",
                "scope": "module_5",
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