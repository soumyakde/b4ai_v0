# ==========================================================
# modules/module_6.py
# Module 6 Runtime Renderer
# Clean, YAML-driven survey architecture
# ==========================================================

import streamlit as st
import yaml
from pathlib import Path

from modules.registry.module_registry import module_registry
from core.progress_engine import get_completed_instruments
from core.submission_engine import submit_instrument
from utils.question_bank_loader import load_question_bank   # ← NEW


# ==========================================================
# ✅ CONSTANTS
# ==========================================================

MODULE_ID = "module_6"

# Directory where survey YAML files live
SURVEY_YAML_DIR = Path("streamlit_app/surveys")  # ← adjust if needed


# ==========================================================
# ✅ ENTRY POINT
# ==========================================================

def render(username):
    """
    Main render function called by module registry runtime.
    Responsible for:
    - Loading module definition
    - Loading completion state
    - Rendering ALL assessments dynamically
    - Rendering Surveys (YAML-driven)
    """

    module = module_registry.get(MODULE_ID)

    if not module:
        st.error(f"Module '{MODULE_ID}' could not be loaded from registry.")
        return

    meta = module.get("meta", {})
    instruments = module.get("instruments", {})

    # --------------------------------------------------
    # ✅ Load completion state from DB
    # --------------------------------------------------

    completed_from_db = get_completed_instruments(
        user_id=username,
        module_id=MODULE_ID
    )

    st.session_state.completed_instruments = set(completed_from_db)

    # --------------------------------------------------
    # ✅ UI Header
    # --------------------------------------------------

    st.title(meta.get("title", "Untitled Module"))
    st.markdown(meta.get("description", ""))

    # ==================================================
    # ✅ RENDER ASSESSMENTS (DYNAMIC + SCALABLE)
    # ==================================================

    assessments = instruments.get("assessments", {})

    for assessment_registry_key, assessment_def in assessments.items():

        assessment_key = assessment_def.get("assessment_code")

        st.markdown("---")

        if assessment_key in st.session_state.completed_instruments:
            st.success(f"✅ {assessment_registry_key} already completed.")
            continue

        assessment_type = assessment_def.get("type")

        if assessment_registry_key == "content_mcq":
            render_content_mcq(
                assessment_def=assessment_def,
                assessment_key=assessment_key,
                username=username
            )

        elif assessment_type == "open_response":
            render_reflection(
                assessment_def=assessment_def,
                assessment_key=assessment_key,
                username=username
            )

    # ==================================================
    # ✅ RENDER SURVEYS (YAML-DRIVEN)
    # ==================================================

    surveys = instruments.get("surveys", {})

    for survey_registry_key, survey_def in surveys.items():

        survey_key = survey_def.get("survey_key", survey_registry_key)

        st.markdown("---")

        if survey_key in st.session_state.completed_instruments:
            st.success(f"✅ {survey_key} already completed.")
            continue

        render_yaml_survey(
            survey_key=survey_key,
            username=username
        )


# ==========================================================
# ✅ MCQ ASSESSMENT (UPDATED SIGNATURE)
# ==========================================================

def render_content_mcq(assessment_def, assessment_key, username):
    """
    Renders randomized MCQ assessment using JSON question bank.
    """

    st.subheader("A: MCQ Content Assessment")

    question_bank_path = assessment_def.get("question_bank_path")
    num_questions = assessment_def.get("num_questions", 10)
    #randomize = assessment_def.get("randomize", True)

    # --------------------------------------------------
    # ✅ Load and select questions (mode-aware)
    # ONE call replaces: open() + json.load() + if/else random.sample()
    # --------------------------------------------------

    try:
        questions = load_question_bank(
            question_bank_path=question_bank_path,
            module_id=MODULE_ID,
            n=num_questions,
        )
    except FileNotFoundError:
        st.error(f"Question bank file not found: {question_bank_path}")
        return
    except ValueError as e:
        st.error(str(e))
        return

    if not questions:
        st.error("No questions were loaded. Check the question bank file.")
        return

    responses = {}

    # --------------------------------------------------
    # ✅ Render Form
    # --------------------------------------------------

    with st.form(f"form_{assessment_key}"):

        for idx, question in enumerate(questions, start=1):

            qid = question["id"]
            qtext = question["question"]
            options = question["options"]

            widget_key = f"{assessment_key}_{qid}"

            st.markdown(f"**{idx}. ({qid})** {qtext}")

            responses[qid] = st.radio(
                "Select an option",
                options,
                index=None,
                key=widget_key,
                label_visibility="collapsed"
            )

            st.markdown("---")

        submitted = st.form_submit_button("Submit Assessment")

        if submitted:

            if not _all_questions_answered(responses):
                st.error("Please answer all questions.")
                st.stop()

            score = 0
            letter_responses = {}

            for question in questions:

                qid = question["id"]
                options = question["options"]
                correct_letter = question.get("answer")

                selected_text = responses[qid]
                option_index = options.index(selected_text)
                selected_letter = chr(65 + option_index)

                letter_responses[qid] = selected_letter

                if correct_letter and selected_letter == correct_letter:
                    score += 1

            submit_instrument(
                user_id=username,
                module_id=MODULE_ID,
                instrument_key=assessment_key,
                responses=letter_responses,
                score=score
            )

            st.session_state.completed_instruments.add(assessment_key)

            st.success(
                f"✅ Assessment submitted successfully. "
                f"Score: {score}/{len(questions)}"
            )

            st.rerun()


# ==========================================================
# ✅ REFLECTION RENDERER (NEW)
# ==========================================================

def render_reflection(assessment_def, assessment_key, username):

    st.subheader("B: Reflections")

    questions = assessment_def.get("questions", [])

    responses = {}

    with st.form(f"form_{assessment_key}"):

        for idx, question in enumerate(questions, start=1):

            q_id = f"q{idx}"

            responses[q_id] = st.text_area(
                f"{idx}. {question}",
                key=f"{assessment_key}_{q_id}"
            )

            st.markdown("---")

        submitted = st.form_submit_button("Submit Reflection")

        if submitted:

            if not _all_questions_answered(responses):
                st.error("Please answer all reflection questions.")
                st.stop()

            submit_instrument(
                user_id=username,
                module_id=MODULE_ID,
                instrument_key=assessment_key,
                responses=responses,
                score=None
            )

            st.session_state.completed_instruments.add(assessment_key)

            st.success("✅ Reflection submitted successfully.")
            st.rerun()


# ==========================================================
# ✅ YAML SURVEY RENDERER (UNCHANGED)
# ==========================================================

def render_yaml_survey(survey_key, username):

    survey_path = SURVEY_YAML_DIR / f"{survey_key}.yaml"

    if not survey_path.exists():
        st.error(f"Survey YAML not found: {survey_path}")
        return

    with open(survey_path, "r", encoding="utf-8") as f:
        survey_data = yaml.safe_load(f)

    title = survey_data.get("title", survey_key)
    sections = survey_data.get("sections", [])

    st.subheader(title)

    responses = {}

    q_num = 0  # global counter across all sections

    with st.form(f"form_{survey_key}"):

        for section in sections:

            st.markdown(f"### {section.get('name')}")

            for q in section.get("questions", []):

                q_num += 1
                q_id = q.get("id")
                q_text = q.get("text")
                q_type = q.get("type")
                options = q.get("options", [])

                widget_key = f"{survey_key}_{q_id}"
                labeled_text = f"{q_num} ({q_id}) {q_text}"

                if q_type == "text":
                    responses[q_id] = st.text_input(
                        labeled_text,
                        key=widget_key
                    )

                elif q_type == "select":
                    responses[q_id] = st.radio(
                        labeled_text,
                        options,
                        index=None,
                        key=widget_key
                    )

                st.markdown("---")

        submitted = st.form_submit_button("Submit Survey")

        if submitted:

            if not _all_questions_answered(responses):
                st.error("Please answer all required questions.")
                st.stop()

            submit_instrument(
                user_id=username,
                module_id=MODULE_ID,
                instrument_key=survey_key,
                responses=responses,
                score=None
            )

            st.session_state.completed_instruments.add(survey_key)

            st.success("✅ Survey submitted successfully.")
            st.rerun()


# ==========================================================
# ✅ VALIDATION
# ==========================================================

def _all_questions_answered(responses):

    for value in responses.values():
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False

    return True