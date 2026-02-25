"""
Post-Course Learning Unit

Architectural Principles:
--------------------------
1. Database is the single source of truth for completion state.
2. Session state is a UI cache only.
3. Completion is derived — never stored redundantly.
4. Rendering must reflect persisted state after logout/login.
5. This is the terminal module (appears after module_7).
"""

import streamlit as st
from typing import Dict, Any
from pathlib import Path

# ---------------------------------------------------------------------
# Asset Directory (Used for AICI question images)
# ---------------------------------------------------------------------

IMAGE_DIR = Path(__file__).resolve().parents[1] / "streamlit_app" / "assets" / "images"

# ---------------------------------------------------------------------
# Core System Imports
# ---------------------------------------------------------------------

from modules.resolution.learning_unit_resolver import LearningUnitResolver
from core.progress_engine import (
    get_required_instruments,
    get_instrument_type,
    get_completed_instruments,
)
from core.submission_engine import submit_instrument
from core.scoring_engine import compute_score
from utils.yaml_loader import load_yaml


MODULE_ID = "post_course"
resolver = LearningUnitResolver()


# =====================================================================
# ✅ ENTRY POINT
# =====================================================================

def render(username: str):

    module: Dict[str, Any] = resolver.get(MODULE_ID)
    required_instruments = get_required_instruments(module)

    st.subheader("Post-Course Instruments")

    # -----------------------------------------------------------------
    # ✅ HYDRATE COMPLETION STATE FROM DATABASE
    # -----------------------------------------------------------------

    completed_from_db = get_completed_instruments(
        user_id=username,
        module_id=MODULE_ID
    )

    st.session_state.completed_instruments = set(completed_from_db)

    ordered_instruments = _order_instruments(required_instruments)

    # -----------------------------------------------------------------
    # ✅ RENDER EACH INSTRUMENT
    # -----------------------------------------------------------------

    for instrument_key in ordered_instruments:

        instrument_type = get_instrument_type(module, instrument_key)

        if instrument_key in st.session_state.completed_instruments:
            st.success(f"✅ {instrument_key} already completed.")
            st.divider()
            continue

        try:
            instrument_yaml = load_yaml(f"surveys/{instrument_key}.yaml")
        except FileNotFoundError:
            st.error(f"Missing YAML file: surveys/{instrument_key}.yaml")
            continue

        title = instrument_yaml.get("title", instrument_key)
        questions = instrument_yaml.get("questions", [])

        st.markdown(f"### {title}")

        responses: Dict[str, Any] = {}

        with st.form(f"form_{instrument_key}"):

            display_number = 1

            for question in questions:

                qid = question["id"]
                qtext = question["text"]
                qtype = str(question.get("type", "text")).strip().lower()
                options = question.get("options", [])

                widget_key = f"{instrument_key}_{qid}"

                st.markdown(f"**{display_number}. ({qid})**")
                st.markdown(qtext)

                if "aici" in instrument_key.lower():
                    _render_question_image(qid)

                if qtype in ["multiple_choice", "mcq", "radio", "select"]:
                    responses[qid] = st.radio(
                        "Select an option",
                        options,
                        index=None,
                        key=widget_key,
                        label_visibility="collapsed"
                    )

                elif qtype in ["checkbox", "multiselect"]:
                    responses[qid] = st.multiselect(
                        "Select all that apply",
                        options,
                        key=widget_key,
                        label_visibility="collapsed"
                    )

                elif qtype == "likert":
                    responses[qid] = st.slider(
                        "Select value",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=widget_key,
                        label_visibility="collapsed"
                    )

                elif qtype == "number":
                    responses[qid] = st.number_input(
                        "Enter number",
                        key=widget_key,
                        label_visibility="collapsed"
                    )

                else:
                    responses[qid] = st.text_input(
                        "Enter response",
                        key=widget_key,
                        label_visibility="collapsed"
                    )

                st.markdown("---")
                display_number += 1

            submitted = st.form_submit_button(f"Submit {title}")

            if submitted:

                for question in questions:
                    qid = question["id"]
                    widget_key = f"{instrument_key}_{qid}"
                    if widget_key in st.session_state:
                        responses[qid] = st.session_state[widget_key]

                if not _all_questions_answered(responses):
                    st.error("Please answer all required questions.")
                    st.stop()

                score = None

                if instrument_type == "assessment":
                    scoring_path = f"surveys/{instrument_key}_scoring.yaml"
                    try:
                        scoring_yaml = load_yaml(scoring_path)
                        score = compute_score(
                            responses=responses,
                            scoring_yaml=scoring_yaml
                        )
                    except FileNotFoundError:
                        st.warning("Scoring file not found.")

                submit_instrument(
                    user_id=username,
                    module_id=MODULE_ID,
                    instrument_key=instrument_key,
                    responses=responses,
                    score=score
                )

                st.session_state.completed_instruments.add(instrument_key)

                st.success("Submitted successfully.")

                # Terminal module feedback
                if len(st.session_state.completed_instruments) == len(required_instruments):
                    st.info("🎉 You have completed the course. Thank you!")

                st.rerun()

        st.divider()


# =====================================================================
# ✅ ORDERING LOGIC (NO DEMOGRAPHICS)
# =====================================================================

def _order_instruments(required_instruments):

    priority_keywords = ["misconception", "aici"]
    ordered = []

    for keyword in priority_keywords:
        for inst in required_instruments:
            if keyword in inst.lower() and inst not in ordered:
                ordered.append(inst)

    for inst in required_instruments:
        if inst not in ordered:
            ordered.append(inst)

    return ordered


# =====================================================================
# ✅ IMAGE RENDERING
# =====================================================================

def _render_question_image(qid: str):

    normalized = qid.lower().replace("_", ".")
    filename = f"{normalized}-img.jpg"
    image_path = IMAGE_DIR / filename

    if image_path.exists():
        st.image(str(image_path), width="stretch")


# =====================================================================
# ✅ VALIDATION
# =====================================================================

def _all_questions_answered(responses: Dict[str, Any]) -> bool:

    for key, value in responses.items():

        if value is None:
            return False

        if isinstance(value, str) and value.strip() == "":
            return False

        if isinstance(value, list) and len(value) == 0:
            return False

    return True