"""
Pre-Course Learning Unit

Architectural Principles:
--------------------------
1. Database is the single source of truth for completion state.
2. Session state is a UI cache only.
3. Completion is derived — never stored redundantly.
4. Rendering must reflect persisted state after logout/login.
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
    get_completed_instruments,  # ✅ Used to hydrate DB state
)
from core.submission_engine import submit_instrument
from core.scoring_engine import compute_score
from utils.yaml_loader import load_yaml


MODULE_ID = "pre_course"
resolver = LearningUnitResolver()


# =====================================================================
# ✅ ENTRY POINT
# =====================================================================

def render(username: str):
    """
    Render the Pre-Course learning unit.

    Important:
    ----------
    This function hydrates completion state from the database
    every time it runs. This guarantees that:

        - Logout does NOT reset progress
        - Session resets do NOT affect completion
        - Rendering always reflects persistent truth
    """

    module: Dict[str, Any] = resolver.get(MODULE_ID)
    required_instruments = get_required_instruments(module)

    st.subheader("Pre-Course Instruments")

    # -----------------------------------------------------------------
    # ✅ HYDRATE COMPLETION STATE FROM DATABASE
    # -----------------------------------------------------------------
    #
    # Why:
    # ----
    # Streamlit session_state is ephemeral.
    # If a user logs out, session resets.
    #
    # Therefore:
    # We MUST re-derive completion state from DB every render.
    #
    # This ensures deterministic UI behavior.
    # -----------------------------------------------------------------

    completed_from_db = get_completed_instruments(
        user_id=username,
        module_id=MODULE_ID
    )

    # Overwrite session cache with persistent truth
    st.session_state.completed_instruments = set(completed_from_db)

    # Order instruments deterministically
    ordered_instruments = _order_instruments(required_instruments)

    # -----------------------------------------------------------------
    # ✅ RENDER EACH INSTRUMENT
    # -----------------------------------------------------------------

    for instrument_key in ordered_instruments:

        instrument_type = get_instrument_type(module, instrument_key)

        # -------------------------------------------------------------
        # ✅ Skip Already Completed Instruments
        # -------------------------------------------------------------
        # This now reflects DATABASE state — not session guesswork.
        # -------------------------------------------------------------

        if instrument_key in st.session_state.completed_instruments:
            st.success(f"✅ {instrument_key} already completed.")
            st.divider()
            continue

        # -------------------------------------------------------------
        # ✅ Load YAML Definition
        # -------------------------------------------------------------

        try:
            instrument_yaml = load_yaml(f"surveys/{instrument_key}.yaml")
        except FileNotFoundError:
            st.error(f"Missing YAML file: surveys/{instrument_key}.yaml")
            continue

        title = instrument_yaml.get("title", instrument_key)
        questions = instrument_yaml.get("questions", [])

        st.markdown(f"### {title}")

        responses: Dict[str, Any] = {}

        # =============================================================
        # ✅ STREAMLIT FORM
        # =============================================================

        with st.form(f"form_{instrument_key}"):

            display_number = 1

            for question in questions:

                qid = question["id"]
                qtext = question["text"]
                qtype = str(question.get("type", "text")).strip().lower()
                options = question.get("options", [])

                widget_key = f"{instrument_key}_{qid}"

                # ------------------------------------------------------
                # ✅ CONDITIONAL "_other" HANDLING
                # ------------------------------------------------------

                if qid.endswith("_other"):

                    base_key = qid.replace("_other", "")
                    parent_widget_key = f"{instrument_key}_{base_key}"

                    parent_value = st.session_state.get(parent_widget_key)

                    # If English selected, auto-fill
                    if parent_value in ["True", True]:
                        responses[qid] = "English"

                    # If not English, require manual entry
                    if parent_value in ["False", False]:

                        if not st.session_state.get(widget_key):
                            st.session_state[widget_key] = "English"

                        responses[qid] = st.text_input(
                            qtext,
                            key=widget_key
                        )

                    continue

                # ------------------------------------------------------
                # ✅ STANDARD QUESTION RENDERING
                # ------------------------------------------------------

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

            # =========================================================
            # ✅ SUBMISSION LOGIC
            # =========================================================

            if submitted:

                # Pull final widget values from session state
                for question in questions:
                    qid = question["id"]
                    widget_key = f"{instrument_key}_{qid}"
                    if widget_key in st.session_state:
                        responses[qid] = st.session_state[widget_key]

                if not _all_questions_answered(responses):
                    st.error("Please answer all required questions.")
                    st.stop()

                score = None

                # Only compute score if assessment
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

                # -----------------------------------------------------
                # ✅ Persist Submission
                # -----------------------------------------------------

                submit_instrument(
                    user_id=username,
                    module_id=MODULE_ID,
                    instrument_key=instrument_key,
                    responses=responses,
                    score=score
                )

                # -----------------------------------------------------
                # ✅ Update Session Cache Immediately
                # -----------------------------------------------------
                # This ensures instant UI feedback before rerun.
                # -----------------------------------------------------

                st.session_state.completed_instruments.add(instrument_key)

                st.success("Submitted successfully.")

                # -----------------------------------------------------
                # ✅ Force Rerun
                # -----------------------------------------------------
                # This ensures:
                # - Form disappears immediately
                # - Completion message replaces it
                # - UI reflects updated DB state
                # -----------------------------------------------------

                st.rerun()

        st.divider()


# =====================================================================
# ✅ ORDERING LOGIC
# =====================================================================

def _order_instruments(required_instruments):
    """
    Deterministic instrument ordering.

    Ensures:
        - Demographics first
        - Misconceptions second
        - AICI third
        - Everything else after
    """

    priority_keywords = ["demographic", "misconception", "aici"]
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
    """
    Renders associated image for AICI questions if present.
    """

    normalized = qid.lower().replace("_", ".")
    filename = f"{normalized}-img.jpg"
    image_path = IMAGE_DIR / filename

    if image_path.exists():
        st.image(str(image_path), width="stretch")


# =====================================================================
# ✅ VALIDATION
# =====================================================================

def _all_questions_answered(responses: Dict[str, Any]) -> bool:
    """
    Ensures all required questions have valid responses.
    """

    for key, value in responses.items():

        if key.endswith("_other"):
            base_key = key.replace("_other", "")
            parent_value = responses.get(base_key)

            if parent_value in ["False", False]:
                if not value or not str(value).strip():
                    return False
            continue

        if value is None:
            return False

        if isinstance(value, str) and value.strip() == "":
            return False

        if isinstance(value, list) and len(value) == 0:
            return False

    return True