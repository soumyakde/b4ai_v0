# ==========================================================
# Module 4 Definition (v6 Compatible)
# Pure declarative metadata — no executable logic
# ==========================================================

from modules.definitions.shared_reflection_questions import REFLECTION_QUESTIONS

MODULE_DEFINITION = {

    # ======================================================
    # Core Metadata
    # ======================================================
    "meta": {
        "module_id": "module_4",
        "title": "Module 4: Natural Language Concepts",
        "version": "1.0.0",
        "description": "Introduction to sources of Bias and Hallucinations in AI outputs.",
        "author": "Soumya De",
        "status": "active",
        "order": 4,
    },

    # ======================================================
    # Pedagogical Metadata
    # ======================================================
    "pedagogy": {
        "learning_objectives": [
            "Identify AI Hallucinations.",
            "Understand sources of bias.",
            "Utility of concepts from early AI research.",
            "Learn about tools to evaluate AI outputs.",
            "Apply logical reasoning to problems.",
        ],
        "prerequisite_knowledge": "Module 3 completion",
        "estimated_time_minutes": 60,
    },

    # ======================================================
    # Instruments
    # ======================================================
    "instruments": {

        "assessments": {

            "content_mcq": {
                "assessment_code": "module4_content_mcq_assessment",
                "instrument_type": "assessment",
                "scope": "module_4",
                "storage": "json",
                "question_bank_path": "content_dev/module4_question_bank.json",
                "num_questions": 10,
                "randomize": True,
            },

            "reflection": {
                "assessment_code": "module_reflections",
                "instrument_type": "assessment",
                "type": "open_response",
                "scope": "module_4",
                "questions": REFLECTION_QUESTIONS,
            },
        },

        "surveys": {
            "sccces": {
                "survey_key": "b4ai_sccces_survey",
                "instrument_type": "survey",
                "scope": "module_4",
            },
            "sims": {
                "survey_key": "b4ai_sims_survey",
                "instrument_type": "survey",
                "scope": "module_4",
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