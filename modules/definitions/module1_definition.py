# ==========================================================
# Module 1 Definition (v6 Compatible)
# Pure declarative metadata — no executable logic
# ==========================================================

from modules.definitions.shared_reflection_questions import REFLECTION_QUESTIONS

MODULE_DEFINITION = {

    # ======================================================
    # Core Metadata
    # ======================================================
    "meta": {
        "module_id": "module_1",
        "title": "Module 1: Introduction to Natural and Artificial Intelligence",
        "version": "1.0.0",
        "description": "Foundational exploration of individual and collective natural and artificial intelligence.",
        "author": "Soumya De",
        "status": "active",
        "order": 1,  # Appears after Pre-Course (order=0)
    },

    # ======================================================
    # Pedagogical Metadata
    # ======================================================
    "pedagogy": {
        "learning_objectives": [
            "Students will identify key differences between biological and artificial intelligence.",
            "Students will recognize that AI systems follow programmed rules rather than 'thinking' like humans.",
            "Students will begin developing a framework for understanding different types of intelligence.",
        ],
        "prerequisite_knowledge": "Pre-Course completion",
        "estimated_time_minutes": 60,
    },

    # ======================================================
    # Instruments (v6 Contract Structure)
    # ======================================================
    "instruments": {

        # --------------------------------------------------
        # Assessments
        # --------------------------------------------------
        "assessments": {

            # Content Knowledge MCQ Assessment
            "content_mcq": {
                "assessment_code": "module1_content_mcq_assessment",
                "instrument_type": "assessment",
                "scope": "module_1",
                "storage": "json",
                "question_bank_path": "content_dev/module1_question_bank.json", #replaced hardcoded path w/ relative path
                #"question_bank_path": (
                    #r"C:\Users\soumy\basics4aiv1_clean\content_dev"
                    #r"\module1_question_bank.json"
                #),
                "num_questions": 20, # 02/24/26 replaced 20 with 10 to generate questionnaire with 10 randomly selected questions 
                "randomize": True,
            },

            # Open-ended Reflection Activity
            "reflection": {
                "assessment_code": "module_reflections", #changed by skd 02/15/2026
                "instrument_type": "assessment",
                "type": "open_response",
                "scope": "module_1",
                "questions": REFLECTION_QUESTIONS,
            },
        },

        # --------------------------------------------------
        # Surveys
        # --------------------------------------------------
        "surveys": {
            "sccces": {
                "survey_key": "b4ai_sccces_survey",
                "instrument_type": "survey",
                "scope": "module_1",
            },
            "sims": {
                "survey_key": "b4ai_sims_survey",
                "instrument_type": "survey",
                "scope": "module_1",
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
    # UI Configuration (Non-binding Hints)
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