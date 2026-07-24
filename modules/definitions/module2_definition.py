# ==========================================================
# Module 2 Definition (v6 Compatible)
# Pure declarative metadata — no executable logic
# ==========================================================

from modules.definitions.shared_reflection_questions import REFLECTION_QUESTIONS

MODULE_DEFINITION = {

    # ======================================================
    # Core Metadata
    # ======================================================
    "meta": {
        "module_id": "module_2",
        "title": "Module 2: Goal-based Problem-solving",
        "version": "1.0.0",
        "description": "Introduction to Problem-solving steps, Learning, and Planning.",
        "author": "Soumya De",
        "status": "active",
        "order": 2,
    },

    # ======================================================
    # Pedagogical Metadata
    # ======================================================
    "pedagogy": {
        "learning_objectives": [
            "Solve a problem using goals and plans.",
            "Learn about problem-solving environments.",
            "Understand what ‘Supervised Learning’ means?",
            "Use ideas like Decomposition and Abstraction.",
            "Know the difference between Strategic and Tactical Planning.",
            "Learn how to Plan – Decide – Act – and Evaluate your solution.",
        ],
        "prerequisite_knowledge": "Module 1 completion",
        "estimated_time_minutes": 60,
    },

    # ======================================================
    # Instruments
    # ======================================================
    "instruments": {

        "assessments": {

            "content_mcq": {
                "assessment_code": "module2_content_mcq_assessment",
                "instrument_type": "assessment",
                "scope": "module_2",
                "storage": "json",
                "question_bank_path": "content_dev/module2_question_bank.json",
                "num_questions": 20,
                "randomize": True,
            },

            "reflection": {
                "assessment_code": "module_reflections",
                "instrument_type": "assessment",
                "type": "open_response",
                "scope": "module_2",
                "questions": REFLECTION_QUESTIONS,
            },
        },

        "surveys": {
            "sccces": {
                "survey_key": "b4ai_sccces_survey",
                "instrument_type": "survey",
                "scope": "module_2",
            },
            "sims": {
                "survey_key": "b4ai_sims_survey",
                "instrument_type": "survey",
                "scope": "module_2",
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