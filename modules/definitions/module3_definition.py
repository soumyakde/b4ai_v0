# ==========================================================
# Module 3 Definition (v6 Compatible)
# Pure declarative metadata — no executable logic
# ==========================================================

from modules.definitions.shared_reflection_questions import REFLECTION_QUESTIONS

MODULE_DEFINITION = {

    # ======================================================
    # Core Metadata
    # ======================================================
    "meta": {
        "module_id": "module_3",
        "title": "How Does AI Work?",
        "version": "1.0.0",
        "description": "Uncover hidden rules, explore decision paths, and master clever strategies.",
        "author": "Soumya De",
        "status": "active",
        "order": 3,
    },

    # ======================================================
    # Pedagogical Metadata
    # ======================================================
    "pedagogy": {
        "learning_objectives": [
            "Identify and distinguish patterns.",
            "Understand and apply rules.",
            "Understand and construct decision trees.",
            "Understand what is MiniMax algorithm."
            "Understand what is Depth First Search (DFS)?"
            "Undertstand and execute algorithms.",
        ],
        "prerequisite_knowledge": "Module 2 completion",
        "estimated_time_minutes": 60,
    },

    # ======================================================
    # Instruments
    # ======================================================
    "instruments": {

        "assessments": {

            "content_mcq": {
                "assessment_code": "module3_content_mcq_assessment",
                "instrument_type": "assessment",
                "scope": "module_3",
                "storage": "json",
                "question_bank_path": "content_dev/module3_question_bank.json",
                "num_questions": 10,
                "randomize": True,
            },

            "reflection": {
                "assessment_code": "module_reflections",
                "instrument_type": "assessment",
                "type": "open_response",
                "scope": "module_3",
                "questions": REFLECTION_QUESTIONS,
            },
        },

        "surveys": {
            "sccces": {
                "survey_key": "b4ai_sccces_survey",
                "instrument_type": "survey",
                "scope": "module_3",
            },
            "sims": {
                "survey_key": "b4ai_sims_survey",
                "instrument_type": "survey",
                "scope": "module_3",
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