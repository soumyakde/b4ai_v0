# streamlit_app/dashboards/teacher_dashboard.py
"""
Teacher Dashboard
=================
Analytics dashboard for teachers. Derived-state architecture —
all computation is delegated to core/analytics/ modules.
This file contains UI rendering only.

Routing: app.py → show_teacher_dashboard(username) when role == "teacher"

Tab structure:
    Tab 0  Basic Statistics          ← implemented (Phase 1)
    Tab 1  Inferential Statistics    ← placeholder (Phase 2)
    Tab 2  IRT Analysis              ← placeholder (Phase 3)
    Tab 3  LLM Analysis              ← placeholder (Phase 4)
    Tab 4  Competency Progression    ← placeholder (Phase 5)
    Tab 5  Report Generation         ← placeholder (Phase 5)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from core.analytics.datasets.canonical_loader import load_canonical_data
from modules.registry.discover import discover_all_module_numbers
from core.analytics.descriptive.score_aggregator import (
    compute_assessment_scores,
    compute_construct_means,
    aggregate_construct_means,
    summarize_scores,
)
from core.analytics.descriptive.descriptive_stats import participant_summary
from core.analytics.inferential.inferential_tests import (
    run_paired_comparison,
    run_between_groups,
    run_repeated_measures,
    run_bland_altman,           # ← NEW
)
try:
    from core.analytics.irt.irt_runner import (
        build_binary_response_matrix,
        build_likert_response_matrix,
        run_rasch_model,
        run_2pl_model,
        run_grm_model,
        get_icc_data,
        get_wright_map_data,
        MIN_N_2PL,
        MIN_N_WARN,
        _GIRTH_AVAILABLE,
    )
    _IRT_AVAILABLE = _GIRTH_AVAILABLE
except Exception:
    _IRT_AVAILABLE = False

try:
    from core.analytics.llm.llm_clients import (
        call_model as _llm_call_model,
        get_available_models,
        get_display_name as _llm_display_name,
    )
    from core.analytics.llm.transcript_store import (
        load_for_analysis,
        get_persistent_transcripts,
        get_transcript_count,
        delete_transcript,
    )
    from core.analytics.llm.ita_pipeline import (
        run_phase1, run_phase2, run_phase2_dedup,
        run_phase3, run_phase4, run_phase5, run_phase6,
        create_run  as _ita_create_run,
        save_phase_result, load_phase_result,
        get_run     as _ita_get_run,
        list_runs   as _ita_list_runs,
        SYSTEM_PROMPT as _ITA_SYSTEM_PROMPT,
        _PHASE2_PROMPT,
        _PHASE3_PROMPT,
    )
    from core.analytics.llm.theme_comparator import (
        compare_runs as _compare_runs,
        align_themes as _align_themes,
    )
    from core.analytics.llm.dta_pipeline import (
        CODEBOOK             as _DTA_CODEBOOK,
        CONSTRUCT_GROUPS     as _DTA_GROUPS,
        LEARNING_OBJECTIVES  as _DTA_LO,
        run_dta_phase2, run_dta_phase3, run_dta_phase5,
        run_lo_analysis,
        create_dta_run       as _dta_create_run,
        save_dta_results, load_dta_results,
        list_dta_runs        as _dta_list_runs,
        _DTA_PHASE2_PROMPT, _DTA_LO_PROMPT, _DTA_PHASE5_PROMPT,
        DTA_SYSTEM_PROMPT,
        _parse_dta_json, _detect_matched_indicators,
    )
    _LLM_AVAILABLE = True
    _LLM_ERR = ""
except Exception as _llm_e:
    _LLM_AVAILABLE = False
    _LLM_ERR = str(_llm_e)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# Maps DB instrument_name → human-readable label for UI
_ASSESSMENT_LABELS: Dict[str, str] = {
    "precourse_pre_ai_misconceptions_assessment":    "Pre — AI Misconceptions",
    "postcourse_post_ai_misconceptions_assessment":  "Post — AI Misconceptions",
    "precourse_pre_aici_assessment":                 "Pre — AI Conceptual Inventory",
    "postcourse_post_aici_assessment":               "Post — AI Conceptual Inventory",
    **{f"module{n}_content_mcq_assessment": f"Module {n} — Content MCQ"
       for n in discover_all_module_numbers()},
}

_SURVEY_LABELS: Dict[str, str] = {
    "b4ai_sccces_survey": "Cognitive Engagement",
    "b4ai_sims_survey":   "SIMS (Motivation)",
}

# Maps module_id (canonical) → display label
_MODULE_LABELS: Dict[str, str] = {
    **{f"module_{n}": f"Module {n}" for n in discover_all_module_numbers()},
    "global":       "Global (Pre/Post)",
    "demographics": "Demographics",
}

# Maps pre-instrument key → matching post-instrument key.
# Used by Bland-Altman expanders in both the Basic Statistics and
# Inferential Statistics tabs.
_PREPOST_PAIRS: dict[str, str] = {
    "precourse_pre_ai_misconceptions_assessment":
        "postcourse_post_ai_misconceptions_assessment",
    "precourse_pre_aici_assessment":
        "postcourse_post_aici_assessment",
}
# Reverse map: post → pre (for lookups when a post instrument is selected)
_PREPOST_PAIRS_REVERSE: dict[str, str] = {
    v: k for k, v in _PREPOST_PAIRS.items()
}

# -----------------------------------------------------------------------
# Cached data loader — TTL 5 minutes
# -----------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _load_data():
    return load_canonical_data()


# -----------------------------------------------------------------------
# Filter helpers
# -----------------------------------------------------------------------

def _get_available_grades(demographics_df: pd.DataFrame) -> List[str]:
    """Return sorted list of grade strings present in demographics."""
    grade_order = [
        "Fourth (4th) grade", "Fifth (5th) grade", "Sixth (6th) grade",
        "Seventh (7th) grade", "Eighth (8th) grade", "Adult",
    ]
    found = set(demographics_df["grade"].dropna().unique())
    ordered = [g for g in grade_order if g in found]
    others  = sorted(found - set(grade_order))
    return ordered + others


def _get_available_cohorts(cohort_map: dict) -> List[str]:
    """Return sorted list of non-None cohort_ids."""
    cohorts = sorted({v for v in cohort_map.values() if v is not None})
    return cohorts


def _apply_filters(
    canonical_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    selected_module_ids: List[str],
    selected_grades: List[str],
    selected_genders: List[str],
    selected_lang: str,
    selected_cohorts: List[str],
    selected_user: str,
    cohort_map: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply sidebar filter selections to canonical_df and demographics_df.

    Returns (filtered_canonical_df, filtered_demographics_df).
    """
    # Step 1 — filter demographics → determines which user_ids are in scope
    demo = demographics_df.copy()

    if selected_grades:
        demo = demo[demo["grade"].isin(selected_grades)]

    if selected_genders:
        demo = demo[demo["gender"].isin(selected_genders)]

    if selected_lang == "English only":
        demo = demo[demo["first_language_english"] == True]
    elif selected_lang == "Non-English only":
        demo = demo[demo["first_language_english"] == False]

    if selected_cohorts:
        demo = demo.copy()
        demo["cohort_id"] = demo["user_id"].map(cohort_map)
        demo = demo[demo["cohort_id"].isin(selected_cohorts)]

    if selected_user and selected_user != "All students":
        demo = demo[demo["user_id"] == selected_user]

    user_ids = set(demo["user_id"])

    # Step 2 — filter canonical_df
    mask = canonical_df["user_id"].isin(user_ids)

    if selected_module_ids:
        mask &= canonical_df["module_id"].isin(selected_module_ids)

    if selected_cohorts:
        mask &= canonical_df["cohort_id"].isin(selected_cohorts)

    return canonical_df[mask].copy(), demo.copy()


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------

def _render_sidebar(
    canonical_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    cohort_map: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Render sidebar filters and return (filtered_canonical, filtered_demographics)."""

    with st.sidebar:
        st.header("🔍 Filters")
        st.caption("All filters combine with AND logic.")

        st.divider()

        # ---- Module ----
        available_module_ids = sorted(canonical_df["module_id"].dropna().unique())
        module_display = {mid: _MODULE_LABELS.get(mid, mid) for mid in available_module_ids}
        selected_module_labels = st.multiselect(
            "Module",
            options=list(module_display.values()),
            default=[],
            placeholder="All modules",
            help="Leave blank to include all modules.",
        )
        selected_module_ids = [
            mid for mid, label in module_display.items()
            if label in selected_module_labels
        ]

        # ---- Grade ----
        available_grades = _get_available_grades(demographics_df)
        selected_grades = st.multiselect(
            "Grade",
            options=available_grades,
            default=[],
            placeholder="All grades",
        )

        # ---- Gender ----
        selected_genders = st.multiselect(
            "Gender",
            options=["Male", "Female"],
            default=[],
            placeholder="All genders",
        )

        # ---- Language ----
        selected_lang = st.radio(
            "First Language",
            options=["All", "English only", "Non-English only"],
            index=0,
            horizontal=True,
        )

        # ---- Cohort ----
        available_cohorts = _get_available_cohorts(cohort_map)
        selected_cohorts = st.multiselect(
            "Cohort",
            options=available_cohorts,
            default=[],
            placeholder="All cohorts",
        ) if available_cohorts else []

        # ---- Single user ----
        all_users = sorted(demographics_df["user_id"].unique())
        selected_user = st.selectbox(
            "Single Student",
            options=["All students"] + all_users,
            index=0,
            help="Select one student to view their individual data.",
        )

        st.divider()

        # ---- Active filter summary ----
        active = []
        if selected_module_ids:
            active.append(f"Modules: {', '.join(selected_module_labels)}")
        if selected_grades:
            active.append(f"Grades: {len(selected_grades)} selected")
        if selected_genders:
            active.append(f"Gender: {', '.join(selected_genders)}")
        if selected_lang != "All":
            active.append(f"Language: {selected_lang}")
        if selected_cohorts:
            active.append(f"Cohorts: {', '.join(selected_cohorts)}")
        if selected_user != "All students":
            active.append(f"Student: {selected_user}")

        if active:
            st.caption("**Active filters:**")
            for a in active:
                st.caption(f"• {a}")
        else:
            st.caption("No filters active — showing all data.")

    return _apply_filters(
        canonical_df, demographics_df,
        selected_module_ids, selected_grades, selected_genders,
        selected_lang, selected_cohorts, selected_user,
        cohort_map,
    )


# -----------------------------------------------------------------------
# Tab 0 — Section helpers
# -----------------------------------------------------------------------

def _render_participant_summary(
    demographics_df: pd.DataFrame,
    cohort_map: dict,
) -> None:
    """Section 1: Participant counts and breakdowns."""

    st.subheader("👥 Participant Summary")

    summary = participant_summary(demographics_df, cohort_map)
    total   = summary["total"]

    if total == 0:
        st.warning("No participants match the current filters.")
        return

    # Top-level metric
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total)

    # Gender counts
    by_gender = summary["by_gender"]
    male_n    = int(by_gender[by_gender.gender == "Male"]["n"].sum()) if "Male" in by_gender.gender.values else 0
    female_n  = int(by_gender[by_gender.gender == "Female"]["n"].sum()) if "Female" in by_gender.gender.values else 0
    col2.metric("Male", male_n)
    col3.metric("Female", female_n)
    unknown_n = total - male_n - female_n
    if unknown_n:
        col4.metric("Unknown Gender", unknown_n)

    st.divider()

    # Three-column breakdown tables
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**By Grade**")
        gr = summary["by_grade"].rename(columns={
            "grade_level": "Level", "grade": "Grade", "n": "N", "pct": "%"
        })
        st.dataframe(gr[["Level", "N", "%"]], hide_index=True, width='stretch')

    with c2:
        st.markdown("**By Language**")
        lang = summary["by_language"].rename(columns={
            "label": "Language", "n": "N", "pct": "%"
        })
        st.dataframe(lang[["Language", "N", "%"]], hide_index=True, width='stretch')

    with c3:
        if summary["by_cohort"] is not None:
            st.markdown("**By Cohort**")
            coh = summary["by_cohort"].rename(columns={"cohort_id": "Cohort", "n": "N", "pct": "%"})
            st.dataframe(coh[["Cohort", "N", "%"]], hide_index=True, width='stretch')


def _histogram(
    values: pd.Series,
    title: str,
    x_label: str,
    x_range: Optional[list] = None,
    color: str = "#4C9BE8",
) -> None:
    """Render a histogram or fallback bar chart."""
    if values.dropna().empty:
        st.caption("No data to display.")
        return

    if _HAS_PLOTLY:
        # Convert to plain Python list to completely strip pandas index.
        # Passing a Series (even after reset_index) lets Plotly read the
        # index as x-values — list() prevents this entirely.
        clean_list = values.dropna().tolist()
        fig = px.histogram(
            x=clean_list,
            nbins=min(20, max(5, len(clean_list) // 2)),
            title=title,
            range_x=x_range,
            color_discrete_sequence=[color],
        )
        fig.update_layout(
            margin=dict(t=40, b=10, l=10, r=10),
            height=280,
            showlegend=False,
        )
        fig.update_xaxes(title=x_label)
        fig.update_yaxes(title="Count")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    else:
        st.bar_chart(values.dropna())


def _grouped_distribution_chart(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: str,
    y_label: str,
    y_range: Optional[list] = None,
) -> None:
    """Render a box-plot distribution of value_col, one box per group_col value
    (e.g. one box per cohort) — the "By Cohort (Distribution)" chart pattern,
    shared across Assessment Scores and Survey Construct Means."""
    plot_df = df[[value_col, group_col]].dropna()
    if plot_df.empty:
        st.caption("No cohort data to display.")
        return

    if _HAS_PLOTLY:
        fig = px.box(
            plot_df,
            x=group_col,
            y=value_col,
            points="all",
            title=title,
        )
        fig.update_layout(
            margin=dict(t=40, b=10, l=10, r=10),
            height=350,
            showlegend=False,
        )
        fig.update_xaxes(title="Cohort")
        fig.update_yaxes(title=y_label, range=y_range)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    else:
        st.bar_chart(plot_df.groupby(group_col)[value_col].mean())


def _stat_row(row: pd.Series, score_col: str, unit: str = "") -> None:
    """Display mean / median / mode / N metrics in a 4-column row."""
    c1, c2, c3, c4 = st.columns(4)
    suffix = unit
    mean_v   = row.get(f"mean_{score_col}")
    median_v = row.get(f"median_{score_col}")
    mode_v   = row.get(f"mode_{score_col}")
    n        = row.get("n_users")

    c1.metric("Mean",   f"{mean_v:.1f}{suffix}"   if mean_v   is not None else "—")
    c2.metric("Median", f"{median_v:.1f}{suffix}" if median_v is not None else "—")
    c3.metric("Mode",   f"{mode_v:.1f}{suffix}"   if mode_v   is not None else "—")
    c4.metric("N",      int(n) if n is not None else "—")


# -----------------------------------------------------------------------
# Survey question maps — per-instrument to avoid Q-ID key collisions
# (SCCCES and SIMS share question IDs like Q4_1, Q5_1–Q5_3 for different items)
#
# Tuple: (construct, question_text, item_reverse_coded)
#
# item_reverse_coded=True  — the item is negatively worded; the raw score
#   is flipped (5 − x) before the construct mean is computed.
#
# IMPORTANT distinction:
#   • item_reverse_coded=True  → individual item score is flipped       (SCCCES Q9_1/Q9_2, Q10_1/Q10_2)
#   • construct reverse_coded=True in _CONSTRUCT_DEFINITIONS            (SIMS External Regulation, Amotivation)
#     → construct mean is interpreted in reverse (higher = worse)
#     → the individual items are NOT flipped; they are forward-coded
# -----------------------------------------------------------------------

# ── SCCCES ──────────────────────────────────────────────────────────────
_SCCCES_QUESTION_MAP = {
    "Q2_1":  ("Engagement With Task",           "I was engaged with the topic at hand.",                                                                                                         False),
    "Q3_1":  ("Effort And Persistence",          "I put in a lot of effort.",                                                                                                                     False),
    "Q3_2":  ("Effort And Persistence",          "I wish we could still continue with the work for a while.",                                                                                     False),
    "Q4_1":  ("Experience Of Flow",              "I was so involved that I forgot everything around me.",                                                                                         False),
    "Q5_1":  ("Coherency Of Messaging",          "While going through this module, I thought about whether the information was well organized.",                                                   False),
    "Q5_2":  ("Coherency Of Messaging",          "While going through this module, I considered whether the information was easy to understand.",                                                  False),
    "Q5_3":  ("Coherency Of Messaging",          "While going through this module, I thought about whether the information flowed well.",                                                          False),
    "Q6_1":  ("Plausibility Of Messaging",       "While going through this module, I thought about whether the information was believable.",                                                       False),
    "Q6_2":  ("Plausibility Of Messaging",       "While going through this module, I thought about whether the information was reasonable.",                                                       False),
    "Q7_1":  ("Credibility Of Messaging",        "While going through this module, I thought about whether the source of the information was trustworthy.",                                        False),
    "Q7_2":  ("Credibility Of Messaging",        "While going through this module, I thought about whether the source of the information was believable.",                                         False),
    "Q8_1":  ("Comprehensibility Of Messaging",  "While going through this module, I thought about whether the information presented was easy to follow.",                                         False),
    "Q8_2":  ("Comprehensibility Of Messaging",  "While going through this module, I thought about whether the information was clear.",                                                            False),
    # ⚠️ item-reverse-coded — negatively worded; raw score flipped (5−x) before construct mean
    "Q9_1":  ("Attention",                       "I was having trouble paying attention during the module.",                                                                                      True),
    "Q9_2":  ("Attention",                       "I was distracted by other thoughts during the module.",                                                                                         True),
    # ⚠️ item-reverse-coded — framed as cultural conflict; flip so higher = more cultural alignment
    "Q10_1": ("Culture",                         "While going through this module, I thought about whether the topic conflicts with my culture (for example, religion or family values).",        True),
    "Q10_2": ("Culture",                         "While going through this module, I thought about whether I agreed with the topic conflicts based on my culture (for example, religion or family values).", True),
    "Q11_1": ("Personal Relevance",              "While going through this module, I thought about how this topic relates to things I like or care about.",                                       False),
    "Q11_2": ("Personal Relevance",              "While going through this module, I thought about how the information could be useful to me.",                                                    False),
    "Q11_3": ("Personal Relevance",              "While going through this module, I thought about how the activities would be helpful to my personal goals.",                                     False),
}

# ── SIMS ────────────────────────────────────────────────────────────────
# SIMS scoring rationale (De Charms & Muir 1978 / Deci & Ryan 1985 SDT):
# Q4_1–Q4_4 (External Regulation) and Q5_1–Q5_3 (Amotivation) are item-reverse-coded.
# Reason: Agree with "Because I have no choice" is a BAD outcome; after reversal (5−x)
# the scored mean is HIGH when the learner is NOT externally regulated or amotivated.
# This makes ALL four SIMS constructs point in the same positive direction:
#   HIGH score on any SIMS construct = BETTER outcome for the learner.
# The scoring YAML (b4ai_sims_scoring.yaml) lists Q4_1–Q4_4 and Q5_1–Q5_3
# under reverse_questions, which DatasetBuilder applies as 5−raw_score.
_SIMS_QUESTION_MAP = {
    "Q2_1":  ("Intrinsic Motivation",   "Because I think that this activity is interesting.",                                                         False),
    "Q2_2":  ("Intrinsic Motivation",   "Because I like doing this activity.",                                                                         False),
    "Q2_3":  ("Intrinsic Motivation",   "Because I feel good when doing this activity.",                                                               False),
    "Q3_1":  ("Identified Regulation",  "Because I think that this activity is good for me.",                                                          False),
    "Q3_2":  ("Identified Regulation",  "Because I think that this activity is important for me.",                                                     False),
    "Q3_3":  ("Identified Regulation",  "Because this activity will help me later.",                                                                   False),
    # ⚠️ Q4_1–Q4_4: External Regulation — item-reverse-coded
    #   Raw: Agree=3 means externally pressured (bad). Reversed: Agree→2, so
    #   higher score = LESS external regulation = GOOD. Same direction as intrinsic/identified.
    "Q4_1":  ("External Regulation",    "Because I am supposed to do it.",                                                                             True),
    "Q4_2":  ("External Regulation",    "Because I have no choice.",                                                                                   True),
    "Q4_3":  ("External Regulation",    "Because I do not want to get in trouble.",                                                                    True),
    "Q4_4":  ("External Regulation",    "Because I feel I have to do it.",                                                                             True),
    # ⚠️ Q5_1–Q5_3: Amotivation — item-reverse-coded
    #   Raw: Agree=3 means amotivated (bad). Reversed: Agree→2, so
    #   higher score = LESS amotivation = GOOD. Consistent direction across all SIMS constructs.
    "Q5_1":  ("Amotivation",            "There may be good reasons to do this activity, but personally, I do not see any.",                           True),
    "Q5_2":  ("Amotivation",            "I am doing this activity, but I am not sure if it is worth it.",                                             True),
    "Q5_3":  ("Amotivation",            "I am doing this activity, but I am not sure it is a good thing to pursue it.",                               True),
}

# Merged view for backward-compat lookup (instrument+qid → tuple)
# Keys are (instrument, qid) to avoid the Q4_1 / Q5_x collisions
_QUESTION_MAP_BY_INST = {
    ("SCCCES", qid): val for qid, val in _SCCCES_QUESTION_MAP.items()
}
_QUESTION_MAP_BY_INST.update({
    ("SIMS", qid): val for qid, val in _SIMS_QUESTION_MAP.items()
})

# Per-instrument reverse-coded item sets (used for ⚠️ column in tables)
_SCCCES_REVERSE_ITEMS: set = {qid for qid, v in _SCCCES_QUESTION_MAP.items() if v[2]}
_SIMS_REVERSE_ITEMS:   set = {qid for qid, v in _SIMS_QUESTION_MAP.items()   if v[2]}
_REVERSE_CODED_ITEMS = _SCCCES_REVERSE_ITEMS  # backward compat for SCCCES-only code


def _get_reverse_items_for_survey(survey_base: str) -> set:
    """Return the set of item-reverse-coded question IDs for a survey."""
    if "sims" in survey_base.lower():
        return _SIMS_REVERSE_ITEMS
    return _SCCCES_REVERSE_ITEMS


def _build_question_guide(instrument: str) -> list:
    """Build question guide rows for SCCCES or SIMS, with ⚠️ on item-reverse-coded items."""
    qmap = _SCCCES_QUESTION_MAP if instrument == "SCCCES" else _SIMS_QUESTION_MAP
    rows = []
    for qid, (con, txt, rev) in sorted(qmap.items()):
        rows.append({
            "Question ID":    f"⚠️ {qid}" if rev else qid,
            "Construct":      con,
            "Question Text":  txt,
            "Item-reverse-coded": "⚠️ Yes" if rev else "No",
        })
    return rows


# -----------------------------------------------------------------------
# Assessment question maps — AI Misconceptions (AIM-F) and AICI
# Verify question text against your instrument YAMLs if it differs.
# -----------------------------------------------------------------------
# AIM-F question text — sourced from pre_ai_misconceptions_assessment.yaml (v1.0)
# Question IDs match the DB question_id values from that instrument.
_AIM_QUESTION_MAP = {
    "Q3_1": "AI systems learn and understand what they are doing on their own.",
    "Q3_2": "I always identify a machine to be an AI machine if it imitates human characteristics like voice, movements, and appearance.",
    "Q3_3": "AI systems have emotions and intuitions.",
    "Q3_4": "AI methods work similar to the brain.",
    "Q3_5": "Any software that uses a database is AI.",
    "Q3_6": "Recommendation systems in games, social media, or search engines are all examples of AI.",
    "Q3_7": "Can something be called AI if it does not involve technology?",
    "Q3_8": "Would you consider AI as a machine with pre-installed knowledge or intelligence?",
}

# AICI question map intentionally omitted — items include images and multi-part
# scenarios that cannot be meaningfully represented in a plain text table.
# See the published AI-CI instrument (Appendix 1, IJAIED) for the full scale.

# Map instrument key keywords → question map
# Only AIM-F is mapped; AICI shows a PDF reference note instead.
_ASSESSMENT_Q_MAPS = {
    "misconception": _AIM_QUESTION_MAP,
    "aim":           _AIM_QUESTION_MAP,
}

# AICI instruments that should show a PDF reference note instead of question text
_AICI_INSTRUMENT_KEYS = {"aici", "conceptual"}


def _get_assessment_q_map(instrument_key: str) -> dict:
    """Return the AIM-F question text map, or {} if not applicable."""
    key_lower = instrument_key.lower()
    for kw, qmap in _ASSESSMENT_Q_MAPS.items():
        if kw in key_lower:
            return qmap
    return {}


def _is_aici_instrument(instrument_key: str) -> bool:
    """Return True if this is an AI Conceptual Inventory instrument."""
    key_lower = instrument_key.lower()
    return any(kw in key_lower for kw in _AICI_INSTRUMENT_KEYS)


def _render_assessment_scores(canonical_df: pd.DataFrame) -> None:
    """Section 2: Assessment scores — per-question or per-student view."""

    st.subheader("📝 Assessment Scores — % Correct")

    asc = compute_assessment_scores(canonical_df)

    if asc.empty:
        st.info("No assessment data available for the current filters.")
        return

    available_keys = sorted(asc["instrument_key"].unique())
    available_labels = {k: _ASSESSMENT_LABELS.get(k, k) for k in available_keys}

    col_sel, col_view = st.columns([2, 1])
    with col_sel:
        selected_label = st.selectbox(
            "Select Instrument",
            options=list(available_labels.values()),
            key="asc_instrument_select",
        )
    with col_view:
        asc_view = st.radio(
            "View by",
            options=["Question", "Student", "Cohort (Distribution)"],
            horizontal=True,
            key="asc_view_radio",
        )

    selected_key = next(k for k, v in available_labels.items() if v == selected_label)
    asc_filtered  = asc[asc["instrument_key"] == selected_key]

    if asc_filtered.empty:
        st.info("No data for the selected instrument.")
        return

    # Summary metrics row
    summary = summarize_scores(asc_filtered)
    if not summary.empty:
        _stat_row(summary.iloc[0], "pct", unit="%")

    if asc_view == "Question":
        # Per-question: % of students who answered correctly
        mask = (
            (canonical_df["instrument_key"] == selected_key) |
            canonical_df["instrument_key"].str.endswith("_" + selected_key)
        )
        item_df = canonical_df[mask & canonical_df["item_score"].isin([0.0, 1.0])].copy()

        if item_df.empty:
            st.info("No item-level data available.")
            return

        import re as _re
        def _qsort(q):
            m = _re.search(r"(\d+)", str(q))
            return int(m.group(1)) if m else 0

        pct_by_q = (
            item_df.groupby("question_id")
            .agg(
                pct_correct=("item_score", lambda x: x.mean() * 100),
                n_students=("user_id", "nunique"),
            )
            .reset_index()
            .sort_values("question_id", key=lambda s: s.map(_qsort))
        )
        if _HAS_PLOTLY:
            import plotly.express as px
            fig = px.bar(
                pct_by_q,
                x="question_id",
                y="pct_correct",
                text=pct_by_q["pct_correct"].round(1).astype(str) + "%",
                title=f"% Students Correct Per Question — {selected_label}",
                labels={"question_id": "Question", "pct_correct": "% Correct"},
                color="pct_correct",
                color_continuous_scale="Blues",
                range_y=[0, 100],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=380,
                margin=dict(t=50, b=10, l=10, r=10),
                coloraxis_showscale=False,
                xaxis_title="Question Item",
                yaxis_title="% Students Correct",
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.bar_chart(pct_by_q.set_index("question_id")[["pct_correct"]])

        st.caption(
            "Each bar shows the percentage of students who answered that item correctly. "
            "Lower bars indicate harder or more commonly misunderstood items."
        )

        with st.expander("Item difficulty table"):
            pct_by_q["pct_correct"] = pct_by_q["pct_correct"].round(1)
            _diff_display = pct_by_q.copy()
            _diff_display.columns = ["Question", "% Correct", "N Students"]
            st.dataframe(
                _diff_display, hide_index=True, width="stretch",
                column_config={
                    "Question":   st.column_config.TextColumn("Question",  width="small"),
                    "% Correct":  st.column_config.NumberColumn("% Correct", format="%.1f", width="small"),
                    "N Students": st.column_config.NumberColumn("N Students", format="%d",   width="small"),
                }
            )

        # ── Question # → Question Text (AIM-F only) or PDF note (AICI) ───────
        _aq_map = _get_assessment_q_map(selected_key)
        if _aq_map:
            with st.expander("📋 Question # → Question Text (AI Misconceptions)", expanded=False):
                st.caption(
                    "Question IDs match the DB question_id values. "
                    "Text sourced from pre_ai_misconceptions_assessment.yaml."
                )
                import pandas as _aqpd
                _aq_rows = [
                    {"Question #": qnum, "Question Text": qtxt}
                    for qnum, qtxt in sorted(
                        _aq_map.items(),
                        key=lambda x: int("".join(filter(str.isdigit, x[0])) or 0)
                    )
                ]
                st.dataframe(
                    _aqpd.DataFrame(_aq_rows),
                    hide_index=True, width="stretch",
                    column_config={
                        "Question #":    st.column_config.TextColumn("Question #",    width="small"),
                        "Question Text": st.column_config.TextColumn("Question Text", width="large"),
                    }
                )
        elif _is_aici_instrument(selected_key):
            with st.expander("📋 AI Conceptual Inventory (AICI) — Item Reference", expanded=False):
                st.caption(
                    "The AI-CI is a 20-item instrument. Items include images, decision-tree "
                    "diagrams, and multi-part scenarios that cannot be meaningfully represented "
                    "in a plain text table."
                )
                st.info(
                    "📄 **Full item listing:** See Appendix 1 of the published AI-CI instrument "
                    "(International Journal of Artificial Intelligence in Education). "
                    "Item codes: CI2, CI4, CI9, CI12, CI13, CI14, CI15, CI16, CI17, CI18, "
                    "CI19, CI22, CI23, CI24, CI26, CI28, CI29, CI30, CI31, CI32."
                )

        # ── Bland-Altman method agreement (pre/post pairs only) ───────────
        # Determine if selected instrument has a known pre/post counterpart.
        # Works whether a pre OR post instrument is currently selected.
        _ba_pre_key  = None
        _ba_post_key = None
        if selected_key in _PREPOST_PAIRS:
            _ba_pre_key  = selected_key
            _ba_post_key = _PREPOST_PAIRS[selected_key]
        elif selected_key in _PREPOST_PAIRS_REVERSE:
            _ba_pre_key  = _PREPOST_PAIRS_REVERSE[selected_key]
            _ba_post_key = selected_key

        if _ba_pre_key and _ba_post_key:
            with st.spinner("Computing method agreement…"):
                _ba_result = run_bland_altman(
                    canonical_df,
                    pre_instrument=_ba_pre_key,
                    post_instrument=_ba_post_key,
                    use_pct=True,
                )
            _render_bland_altman_expander(_ba_result, score_label="% Correct")
        # ─────────────────────────────────────────────────────────────────

    elif asc_view == "Student":
        # Per-student: distribution of total % correct
        _histogram(
            asc_filtered["pct_correct"],
            title=f"% Correct Distribution Across Students — {selected_label}",
            x_label="% Correct",
            x_range=[0, 100],
        )
        with st.expander("Individual student scores"):
            display = asc_filtered[["user_id","n_items_answered","raw_score","pct_correct"]].copy()
            display.columns = ["Student", "Items Answered", "Raw Score", "% Correct"]
            display["% Correct"] = display["% Correct"].round(1)
            st.dataframe(display.sort_values("% Correct", ascending=False),
                         hide_index=True, width="stretch")

    else:
        # Per-cohort: compare cohorts' % correct distributions for this instrument
        if "cohort_id" not in asc_filtered.columns or asc_filtered["cohort_id"].dropna().empty:
            st.info("No cohort data available for the current filters.")
        else:
            cohort_summary = summarize_scores(asc_filtered, group_by_col="cohort_id")
            if not cohort_summary.empty:
                display_cs = cohort_summary[
                    ["cohort_id", "n_users", "mean_pct", "median_pct", "mode_pct"]
                ].copy()
                display_cs.columns = ["Cohort", "N", "Mean %", "Median %", "Mode %"]
                for col in ["Mean %", "Median %", "Mode %"]:
                    display_cs[col] = display_cs[col].round(1)
                st.dataframe(display_cs, hide_index=True, width="stretch")

            _grouped_distribution_chart(
                asc_filtered,
                value_col="pct_correct",
                group_col="cohort_id",
                title=f"% Correct by Cohort — {selected_label}",
                y_label="% Correct",
                y_range=[0, 100],
            )
            st.caption(
                "Each box shows the spread of % Correct scores for students in that "
                "cohort, for the selected instrument. Individual student scores are "
                "plotted as points alongside the box."
            )



# -----------------------------------------------------------------------
# Construct definitions for teacher-facing explanations (from YAML)
# -----------------------------------------------------------------------
_CONSTRUCT_DEFINITIONS = {
    # Messaging Perception (CCCES)
    "coherency_of_messaging": {
        "label": "Coherency of Messaging",
        "definition": "How logically organized and internally consistent learners found the instructional message.",
        "analytic_focus": ["clarity of sequence", "logical connections", "perceived consistency across activities"],
        "scale_low":  "Learners found the content disjointed, hard to follow, or internally contradictory.",
        "scale_mid":  "Learners found the content somewhat organized but noticed gaps in sequencing.",
        "scale_high": "Learners found the content clearly structured, logical, and easy to follow.",
        "reverse_coded": False,
    },
    "plausibility_of_messaging": {
        "label": "Plausibility of Messaging",
        "definition": "Learners' judgment about whether the message seemed truthful and realistic.",
        "analytic_focus": ["perceived realism", "alignment with prior beliefs", "willingness to reconsider ideas"],
        "scale_low":  "Learners found the message unbelievable or at odds with what they already knew.",
        "scale_mid":  "Learners were uncertain — the message partially matched their expectations.",
        "scale_high": "Learners found the message plausible and consistent with their understanding of the world.",
        "reverse_coded": False,
    },
    "credibility_of_messaging": {
        "label": "Credibility of Messaging",
        "definition": "The extent to which learners trusted the source and perceived it as knowledgeable.",
        "analytic_focus": ["trust in instructor or system", "perceived expertise", "authority of examples"],
        "scale_low":  "Learners did not trust the source or questioned its expertise.",
        "scale_mid":  "Learners partially trusted the source but had some doubts.",
        "scale_high": "Learners found the source highly trustworthy and credible.",
        "reverse_coded": False,
    },
    "comprehensibility_of_messaging": {
        "label": "Comprehensibility of Messaging",
        "definition": "How well learners understood the instructional message.",
        "analytic_focus": ["clarity of language", "ease of understanding", "breakdowns in comprehension"],
        "scale_low":  "Learners found the message difficult to understand, confusing, or unclear.",
        "scale_mid":  "Learners understood some parts but struggled with others.",
        "scale_high": "Learners found the message clear and easy to understand.",
        "reverse_coded": False,
    },
    # Individual Characteristics (CCCES)
    "attention": {
        "label": "Attention",
        "definition": "The amount of focused attention learners reported giving to the task.",
        "analytic_focus": ["sustained focus", "distraction", "task absorption"],
        "scale_low":  "Learners reported being frequently distracted or unable to maintain focus.",
        "scale_mid":  "Learners maintained attention for some but not all of the activity.",
        "scale_high": "Learners reported being fully absorbed and attentive throughout.",
        "reverse_coded": False,
        "item_reverse_note": "⚠️ Items Q9_1 and Q9_2 are negatively worded — raw scores are flipped (5−x) before the construct mean is computed. A resulting mean ≤ 2.5 indicates attention problems warrant follow-up.",
    },
    "personal_relevance": {
        "label": "Personal Relevance",
        "definition": "How connected learners felt the content was to their personal lives, interests, or goals.",
        "analytic_focus": ["real-life connections", "relevance to daily experiences", "personal meaning"],
        "scale_low":  "Learners saw no connection between the content and their own lives.",
        "scale_mid":  "Learners saw some relevance but found it limited or indirect.",
        "scale_high": "Learners found the content highly meaningful and directly relevant to their lives.",
        "reverse_coded": False,
    },
    "culture": {
        "label": "Culture",
        "definition": "How well the content aligned with learners' cultural knowledge and experiences.",
        "analytic_focus": ["cultural congruence", "conflict with prior cultural knowledge", "accessibility of examples"],
        "scale_low":  "Learners felt the content conflicted with or ignored their cultural background.",
        "scale_mid":  "Learners found partial cultural alignment but some examples felt unfamiliar.",
        "scale_high": "Learners found the content culturally accessible and consistent with their background.",
        "reverse_coded": False,
        "item_reverse_note": "⚠️ Items Q10_1 and Q10_2 are negatively framed (cultural conflict) — raw scores are flipped (5−x) before the construct mean is computed, so higher means = greater cultural alignment.",
    },
    # Cognitive Engagement (SCES)
    "engagement_with_task": {
        "label": "Engagement with Task",
        "definition": "Learners' perceived level of involvement and active participation during the activity.",
        "analytic_focus": ["enjoyment", "active participation", "interest during activity"],
        "scale_low":  "Learners reported little involvement or interest in the activity.",
        "scale_mid":  "Learners engaged moderately — interested at times but not consistently.",
        "scale_high": "Learners were highly involved, interested, and actively participating.",
        "reverse_coded": False,
    },
    "effort_and_persistence": {
        "label": "Effort and Persistence",
        "definition": "How hard learners tried and whether they persisted when the task became difficult.",
        "analytic_focus": ["perseverance", "willingness to continue", "response to difficulty"],
        "scale_low":  "Learners gave up easily or put in minimal effort.",
        "scale_mid":  "Learners tried moderately but did not always persist through difficulties.",
        "scale_high": "Learners consistently gave their best effort and persisted through challenges.",
        "reverse_coded": False,
    },
    "experience_of_flow": {
        "label": "Experience of Flow",
        "definition": "The experience of being fully absorbed — losing track of time and focused completely.",
        "analytic_focus": ["loss of time awareness", "deep immersion", "sustained enjoyment"],
        "scale_low":  "Learners remained aware of time passing and did not feel absorbed.",
        "scale_mid":  "Learners experienced brief moments of absorption but not sustained flow.",
        "scale_high": "Learners lost track of time and felt completely immersed in the activity.",
        "reverse_coded": False,
    },
    # Motivation (SIMS)
    "intrinsic_motivation": {
        "label": "Intrinsic Motivation",
        "definition": "Doing the activity for its own inherent enjoyment and satisfaction.",
        "analytic_focus": ["enjoyment", "curiosity", "interest for its own sake"],
        "scale_low":  "Learners did not find the activity enjoyable or interesting in itself.",
        "scale_mid":  "Learners found the activity somewhat enjoyable but not intrinsically compelling.",
        "scale_high": "Learners were genuinely curious and enjoyed the activity for its own sake.",
        "reverse_coded": False,
    },
    "identified_regulation": {
        "label": "Identified Regulation",
        "definition": "Doing the activity because learners personally value it and see it as useful for them.",
        "analytic_focus": ["perceived usefulness", "personal value", "goal alignment"],
        "scale_low":  "Learners saw no personal value in the activity.",
        "scale_mid":  "Learners saw some value but it felt only loosely connected to their goals.",
        "scale_high": "Learners personally endorsed the activity as valuable and important for their development.",
        "reverse_coded": False,
    },
    "external_regulation": {
        "label": "External Regulation",
        "definition": "Doing the activity because of external pressure, rules, or to avoid consequences.",
        "analytic_focus": ["rewards", "pressure", "compliance"],
        "scale_low":  "Learners felt strongly externally pressured — they felt they had no choice or feared consequences.",
        "scale_mid":  "Learners felt some external pressure but also had some personal buy-in.",
        "scale_high": "Learners were NOT externally pressured — they chose to participate freely.",
        "reverse_coded": False,
        "item_reverse_note": "⚠️ Items Q4_1–Q4_4 are item-reverse-coded (5−x) so that HIGH scores indicate LESS external regulation. A low mean (≤ 2.5) warrants attention — it means learners felt coerced rather than freely choosing to participate.",
    },
    "amotivation": {
        "label": "Amotivation",
        "definition": "A lack of motivation — feeling no reason to do the activity and disconnected from outcomes.",
        "analytic_focus": ["disengagement", "helplessness", "lack of purpose"],
        "scale_low":  "Learners were amotivated — they could see no reason to participate and felt disconnected.",
        "scale_mid":  "Learners showed some motivational uncertainty or occasional disengagement.",
        "scale_high": "Learners were NOT amotivated — they had clear reasons to participate and felt engaged.",
        "reverse_coded": False,
        "item_reverse_note": "⚠️ Items Q5_1–Q5_3 are item-reverse-coded (5−x) so that HIGH scores indicate LESS amotivation. A low mean (≤ 2.0) is a concern — it means learners saw little point in the activity.",
    },
}

def _render_survey_construct_means(canonical_df: pd.DataFrame) -> None:
    """Section 3: Survey construct mean score histograms + summary."""

    st.subheader("📋 Survey Construct Means (Likert 1–4)")

    cm = compute_construct_means(canonical_df)

    if cm.empty:
        st.info("No survey data available for the current filters.")
        return

    # Survey selector
    available_survey_keys = sorted(set(
        k.split("_b4ai_")[1].replace("_survey", "") if "_b4ai_" in k else k
        for k in cm["instrument_key"].unique()
    ))
    # Rebuild canonical key from suffix
    suffix_to_key = {}
    for k in cm["instrument_key"].unique():
        if "sccces" in k:
            suffix_to_key["sccces"] = k.split("_b4ai_")[0]  # not needed — use base key approach
    # Simpler: get canonical base keys present
    base_keys_present = set()
    for k in cm["instrument_key"].unique():
        for base in ["b4ai_sccces_survey", "b4ai_sims_survey"]:
            if k == base or k.endswith("_" + base):
                base_keys_present.add(base)

    survey_options = {_SURVEY_LABELS.get(b, b): b for b in sorted(base_keys_present)}

    if not survey_options:
        st.info("No survey data in canonical dataset.")
        return

    selected_survey_label = st.selectbox(
        "Select Survey",
        options=list(survey_options.keys()),
        key="survey_select",
    )
    selected_survey_base = survey_options[selected_survey_label]

    # Filter cm to selected survey (handles DB-style keys with module prefix)
    cm_survey = cm[
        cm["instrument_key"].apply(
            lambda k: k == selected_survey_base or k.endswith("_" + selected_survey_base)
        )
    ]

    # Per-module vs Aggregate toggle
    view_mode = st.radio(
        "View",
        options=["Aggregate (all modules)", "Per module"],
        horizontal=True,
        key="survey_view_mode",
    )

    if view_mode == "Aggregate (all modules)":
        cm_display = aggregate_construct_means(cm_survey)
        module_col = None
    else:
        cm_display = cm_survey.copy()
        # Module selector
        available_mods = sorted(cm_survey["module_id"].dropna().unique()) if "module_id" in cm_survey.columns else []
        if available_mods:
            mod_labels = [_MODULE_LABELS.get(m, m) for m in available_mods]
            selected_mod_label = st.selectbox("Module", mod_labels, key="survey_mod_select")
            selected_mod_id = available_mods[mod_labels.index(selected_mod_label)]
            cm_display = cm_survey[cm_survey["module_id"] == selected_mod_id]
        module_col = "module_id"

        # ── Cross-module trajectory chart (Per module mode only) ──────────────
        # Show how each construct's mean score varies across all 7 modules
        if not cm_survey.empty and "module_id" in cm_survey.columns:
            st.markdown("#### 📈 Construct means across modules")
            st.caption(
                "Mean score per construct per module (averaged across all students). "
                "Each line traces how a construct evolves through the programme."
            )
            # Compute mean score per construct × module
            _traj = (
                cm_survey.groupby(["module_id", "construct"])["mean_score"]
                .mean()
                .reset_index()
            )
            _traj["module_id"] = _traj["module_id"].apply(
                lambda m: _MODULE_LABELS.get(m, m)
            )
            _traj["mean_score"] = _traj["mean_score"].round(3)
            _traj["construct_label"] = _traj["construct"].apply(
                lambda c: _CONSTRUCT_DEFINITIONS.get(c, {}).get("label", c.replace("_"," ").title())
            )

            # Sort by module number
            import re as _re_traj
            def _mod_sort(m):
                _mn = _re_traj.search(r"(\d+)", str(m))
                return int(_mn.group(1)) if _mn else 0

            _traj = _traj.sort_values(
                "module_id", key=lambda s: s.map(_mod_sort)
            )

            if _HAS_PLOTLY:
                import plotly.express as px
                _fig_traj = px.line(
                    _traj,
                    x="module_id",
                    y="mean_score",
                    color="construct_label",
                    markers=True,
                    title=f"Construct Trajectory Across Modules — {selected_survey_label}",
                    labels={
                        "module_id":      "Module",
                        "mean_score":     "Mean Score (1–4)",
                        "construct_label":"Construct",
                    },
                    range_y=[1, 4],
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                _fig_traj.add_hline(
                    y=3.0,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="3.0 (positive threshold)",
                    annotation_position="right",
                )
                _fig_traj.add_hline(
                    y=2.5,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="2.5 (attention threshold)",
                    annotation_position="right",
                )
                _fig_traj.update_layout(
                    height=480,
                    margin=dict(t=50, b=10, l=10, r=120),
                    xaxis_title="Module",
                    yaxis_title="Mean Score (1–4 Likert)",
                    legend_title="Construct",
                )
                st.plotly_chart(_fig_traj, width="stretch")
            else:
                # Fallback: pivot and use st.line_chart
                _pivot = _traj.pivot(
                    index="module_id", columns="construct_label", values="mean_score"
                )
                st.line_chart(_pivot)

            st.caption(
                "Dotted line = 3.0 (positive engagement threshold). "
                "Dashed line = 2.5 (attention warranted). "
                "Scale: 1 = Strongly disagree → 4 = Strongly agree."
            )
            st.divider()

    if cm_display.empty:
        st.info("No data for the selected combination.")
        return

    # Diagnostic: if no constructs found, fall back to canonical_df directly
    available_constructs = sorted(cm_display["construct"].dropna().unique())

    if not available_constructs:
        # cm_display has rows but construct column is empty — fall back to
        # reading construct names directly from canonical_df item-level data
        base_key = selected_survey_base
        fallback_mask = (
            (canonical_df["instrument_key"] == base_key) |
            canonical_df["instrument_key"].str.endswith("_" + base_key)
        )
        available_constructs = sorted(
            canonical_df[fallback_mask]["construct"].dropna().unique()
        )
        if not available_constructs:
            with st.expander("Debug — raw instrument keys in dataset"):
                st.write(sorted(canonical_df["instrument_key"].dropna().unique()))
                st.write("cm columns:", cm_display.columns.tolist())
                st.write("cm constructs:", cm_display["construct"].unique()[:20].tolist() if "construct" in cm_display.columns else "NO CONSTRUCT COLUMN")
    # Key includes the survey name so the widget resets when switching surveys
    selected_constructs = st.multiselect(
        "Constructs",
        options=available_constructs,
        default=available_constructs,
        key=f"construct_select_{selected_survey_base}",
    )

    if not selected_constructs:
        st.info("Select at least one construct.")
        return

    cm_filtered = cm_display[cm_display["construct"].isin(selected_constructs)]

    # Summary across users for each construct — scoped to the current
    # Aggregate/Per-module selection above. Kept for the interpretation
    # guide expander below (per-construct mean lookup), independent of the
    # cross-module table rendered next.
    summary_cm = summarize_scores(cm_filtered)

    if not summary_cm.empty:
        # ── Summary Statistics per Construct — always spans every module ──
        # (Module | Construct | N | Mean), independent of the Aggregate/
        # Per-module toggle above, so every construct's module is always
        # visible without needing to flip the module selector repeatedly.
        cm_for_table = cm_survey[cm_survey["construct"].isin(selected_constructs)]
        table_summary = summarize_scores(cm_for_table, group_by_col="module_id")

        if not table_summary.empty:
            st.markdown("**Summary Statistics per Construct**")

            import re as _re_mod
            def _mod_num(m):
                _mn = _re_mod.search(r"(\d+)", str(m))
                return int(_mn.group(1)) if _mn else 999  # non-numeric (e.g. "global") sorts last

            module_ids = sorted(table_summary["module_id"].dropna().unique(), key=_mod_num)

            # N per module = distinct students who answered ANY selected
            # construct in that module — NOT a sum of per-construct Ns,
            # which would double-count students who answered multiple
            # constructs.
            n_per_module = cm_for_table.groupby("module_id")["user_id"].nunique()

            _rows = []
            _band_flags = []
            _subtotal_flags = []
            _shade = False
            for _mid in module_ids:
                _mod_label = _MODULE_LABELS.get(_mid, _mid)
                _mod_rows = table_summary[table_summary["module_id"] == _mid].sort_values("construct")
                for _, _r in _mod_rows.iterrows():
                    _rows.append({
                        "Module": _mod_label,
                        "Construct": str(_r["construct"]).replace("_", " ").title(),
                        "N": int(_r["n_users"]),
                        "Mean": round(_r["mean_score"], 2) if pd.notna(_r["mean_score"]) else None,
                    })
                    _band_flags.append(_shade)
                    _subtotal_flags.append(False)
                # Subtotal row: mean of that module's construct means (per
                # your confirmed choice — each construct counts equally).
                _mod_mean = _mod_rows["mean_score"].mean()
                _rows.append({
                    "Module": _mod_label,
                    "Construct": f"{_mod_label} — Summary",
                    "N": int(n_per_module.get(_mid, 0)),
                    "Mean": round(_mod_mean, 2) if pd.notna(_mod_mean) else None,
                })
                _band_flags.append(_shade)
                _subtotal_flags.append(True)
                _shade = not _shade

            display_summary = pd.DataFrame(_rows)
            _band = pd.Series(_band_flags)
            _is_subtotal = pd.Series(_subtotal_flags)

            def _row_style(row):
                bg = "background-color: rgba(76, 155, 232, 0.10);" if _band.iloc[row.name] else ""
                bold = "font-weight: 700;" if _is_subtotal.iloc[row.name] else ""
                style = f"{bg} {bold}".strip()
                return [style] * len(row)

            styled_summary = display_summary.style.apply(_row_style, axis=1)

            st.dataframe(
                styled_summary,
                hide_index=True, width="stretch",
                column_config={
                    "Module":    st.column_config.TextColumn("Module", width="small"),
                    "Construct": st.column_config.TextColumn("Construct", width="medium"),
                    "N":         st.column_config.NumberColumn("N", format="%d", width="small"),
                    "Mean":      st.column_config.NumberColumn("Mean", format="%.2f", width="small"),
                }
            )

        # Reverse-coding alerts: construct-level (External Reg, Amotivation)
        rev_coded = [c for c in selected_constructs
                     if _CONSTRUCT_DEFINITIONS.get(c, {}).get("reverse_coded")]
        if rev_coded:
            for rc in rev_coded:
                note = _CONSTRUCT_DEFINITIONS[rc].get("reverse_note", "")
                if note:
                    st.warning(note)
        # Item-level reverse-coding alerts (Attention, Culture)
        item_rev = [c for c in selected_constructs
                    if _CONSTRUCT_DEFINITIONS.get(c, {}).get("item_reverse_note")]
        if item_rev:
            for ir in item_rev:
                st.info(_CONSTRUCT_DEFINITIONS[ir]["item_reverse_note"])

        # Per-construct interpretation guide
        with st.expander("📖 How to interpret these scores", expanded=False):
            st.markdown(
                "**Scale: 1 = Strongly disagree → 4 = Strongly agree**\n\n"
                "Scores above **3.0** are generally positive. "
                "Scores below **2.5** may warrant attention. "
                "For reverse-coded constructs (External Regulation, Amotivation), "
                "lower scores are better."
            )
            st.divider()
            for construct in selected_constructs:
                cdef = _CONSTRUCT_DEFINITIONS.get(construct)
                if not cdef:
                    continue
                # Get mean for this construct from summary
                cmean_row = summary_cm[summary_cm["construct"] == construct]
                cmean_val = cmean_row["mean_score"].iloc[0] if not cmean_row.empty else None

                rev_tag = " ⚠️ Reverse-coded" if cdef.get("reverse_coded") else ""
                st.markdown(f"**{cdef['label']}{rev_tag}**")
                st.caption(cdef["definition"])

                if cmean_val is not None:
                    # Interpret the mean
                    if cdef.get("reverse_coded"):
                        if cmean_val >= 3.0:
                            interp = f"🔴 High ({cmean_val:.2f}) — {cdef['scale_high']}"
                        elif cmean_val >= 2.0:
                            interp = f"🟡 Moderate ({cmean_val:.2f}) — {cdef['scale_mid']}"
                        else:
                            interp = f"🟢 Low ({cmean_val:.2f}) — {cdef['scale_low']}"
                    else:
                        if cmean_val >= 3.0:
                            interp = f"🟢 High ({cmean_val:.2f}) — {cdef['scale_high']}"
                        elif cmean_val >= 2.0:
                            interp = f"🟡 Moderate ({cmean_val:.2f}) — {cdef['scale_mid']}"
                        else:
                            interp = f"🔴 Low ({cmean_val:.2f}) — {cdef['scale_low']}"
                    st.markdown(f"**Group mean: {interp}**")

                st.markdown("*Analytic focus:* " +
                            ", ".join(cdef["analytic_focus"]))
                if cdef.get("reverse_coded") and cdef.get("reverse_note"):
                    st.caption(cdef["reverse_note"])
                if cdef.get("item_reverse_note"):
                    st.caption(cdef["item_reverse_note"])
                st.divider()

    # View toggle: per-question or per-student
    survey_item_view = st.radio(
        "Chart view",
        options=["By Question Item", "By Student (distribution)", "By Cohort (distribution)"],
        horizontal=True,
        key="survey_item_view",
    )

    if survey_item_view == "By Question Item":
        # Per-question: mean score per item across all students
        import re as _re2
        def _qsort2(q):
            m = _re2.search(r"(\d+)", str(q))
            return int(m.group(1)) if m else 0

        # Get item-level data from canonical_df
        base_key = selected_survey_base
        item_mask = (
            (canonical_df["instrument_key"] == base_key) |
            canonical_df["instrument_key"].str.endswith("_" + base_key)
        )
        item_data = canonical_df[
            item_mask &
            canonical_df["construct"].isin(selected_constructs) &
            canonical_df["item_score"].notna()
        ].copy()

        if item_data.empty:
            st.info("No item-level data available for selected constructs.")
        else:

            # Single agg: mean score + distinct student count per question
            item_means = (
                item_data.groupby(["question_id", "construct"])
                .agg(
                    mean_score=("item_score", "mean"),
                    n_students=("user_id", "nunique"),
                )
                .reset_index()
                .sort_values("question_id", key=lambda s: s.map(_qsort2))
            )
            item_means["mean_score"] = item_means["mean_score"].round(3)

            if _HAS_PLOTLY:
                import plotly.express as px
                fig = px.bar(
                    item_means,
                    x="question_id",
                    y="mean_score",
                    color="construct",
                    barmode="group",
                    text=item_means["mean_score"].round(2).astype(str),
                    title=f"Mean Score Per Question Item — {selected_survey_label}",
                    labels={
                        "question_id": "Question Item",
                        "mean_score":  "Mean Score (1–4)",
                        "construct":   "Construct",
                    },
                    range_y=[1, 4],
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    height=420,
                    margin=dict(t=50, b=10, l=10, r=10),
                    xaxis_title="Question Item",
                    yaxis_title="Mean Score (1–4 Likert)",
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.bar_chart(item_means.set_index("question_id")["mean_score"])

            st.caption(
                "Each bar shows the mean Likert score (1–4) for that item across all students. "
                "Scores closer to 4 indicate stronger agreement / higher engagement."
            )

            with st.expander("Item means table"):
                # Build display table with ⚠️ flag for reverse-coded items
                display_items = item_means[
                    ["question_id", "construct", "mean_score", "n_students"]
                ].copy()
                _rev_set = _get_reverse_items_for_survey(selected_survey_base)
                display_items["Item-reverse-coded"] = display_items["question_id"].apply(
                    lambda q: "⚠️ Yes" if q in _rev_set else "No"
                )
                display_items = display_items.rename(columns={
                    "question_id": "Question",
                    "construct":   "Construct",
                    "mean_score":  "Mean Score",
                    "n_students":  "N Students",
                })
                # Put ⚠️ column right after Question
                display_items = display_items[
                    ["Question", "Item-reverse-coded", "Construct", "Mean Score", "N Students"]
                ]
                st.dataframe(
                    display_items.reset_index(drop=True),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Question":      st.column_config.TextColumn("Question",      width="small"),
                        "Item-reverse-coded": st.column_config.TextColumn("Item-reverse-coded", width="small"),
                        "Construct":     st.column_config.TextColumn("Construct",     width="medium"),
                        "Mean Score":    st.column_config.NumberColumn("Mean Score",  format="%.2f", width="small"),
                        "N Students":    st.column_config.NumberColumn("N Students",  format="%d",   width="small"),
                    }
                )
                _n_rev = (display_items["Item-reverse-coded"] == "⚠️ Yes").sum()
                if _n_rev:
                    st.caption(
                        f"⚠️ {_n_rev} item(s) in this view are item-reverse-coded — "
                        "negatively worded items whose raw scores are flipped "
                        "(5 − raw score) before the construct mean is computed. "
                        "This is separate from construct-level reverse interpretation "
                        "(External Regulation, Amotivation) where items are forward-coded "
                        "but a high mean indicates a worse outcome."
                    )

                # ─── Question ID → full question text guide ──────────────────
                _inst_key = "SIMS" if "sims" in selected_survey_base else "SCCCES"
                _guide_rows = _build_question_guide(_inst_key)
                if _guide_rows:
                    import pandas as _gpd
                    with st.expander(f"📋 Question ID → Question Text ({_inst_key})", expanded=False):
                        st.caption(
                            "Questions marked ⚠️ are reverse-coded — "
                            "negatively worded items whose raw scores are flipped "
                            "before computing construct means."
                        )
                        st.dataframe(
                            _gpd.DataFrame(_guide_rows),
                            hide_index=True, width="stretch",
                            column_config={
                                "Question ID":   st.column_config.TextColumn("Question ID",   width="small"),
                                "Item-reverse-coded": st.column_config.TextColumn("Item-reverse-coded", width="small"),
                                "Construct":     st.column_config.TextColumn("Construct",     width="medium"),
                                "Question Text": st.column_config.TextColumn("Question Text", width="large"),
                            }
                        )

    elif survey_item_view == "By Student (distribution)":
        # Original per-student distribution histograms
        n_constructs = len(selected_constructs)
        cols_per_row = min(3, n_constructs)
        rows = (n_constructs + cols_per_row - 1) // cols_per_row

        construct_list = selected_constructs
        for row_i in range(rows):
            cols = st.columns(cols_per_row)
            for col_i, construct in enumerate(
                construct_list[row_i * cols_per_row : (row_i + 1) * cols_per_row]
            ):
                with cols[col_i]:
                    cm_c = cm_filtered[cm_filtered["construct"] == construct]
                    _histogram(
                        cm_c["mean_score"],
                        title=construct.replace("_", " ").title(),
                        x_label="Mean Score",
                        x_range=[1, 4],
                        color="#F4845F",
                    )

    else:
        # By Cohort (distribution): compare cohorts' mean scores per construct.
        # cm_filtered loses cohort_id in "Aggregate (all modules)" view mode
        # (aggregate_construct_means() intentionally collapses across modules
        # and doesn't carry demographic columns through). Fall back to
        # cm_survey — the un-collapsed, per-module-per-user data, which does
        # carry cohort_id — pooling all modules' responses for that construct
        # in that case, rather than erroring.
        if "cohort_id" in cm_filtered.columns:
            cm_cohort_source = cm_filtered
        else:
            cm_cohort_source = cm_survey[cm_survey["construct"].isin(selected_constructs)]
            st.caption(
                "ℹ️ Aggregate view pools all modules' responses together for "
                "this cohort comparison. Switch to \"Per module\" for a "
                "single module's cohort comparison instead."
            )

        n_constructs = len(selected_constructs)
        cols_per_row = min(2, n_constructs)
        rows = (n_constructs + cols_per_row - 1) // cols_per_row

        construct_list = selected_constructs
        for row_i in range(rows):
            cols = st.columns(cols_per_row)
            for col_i, construct in enumerate(
                construct_list[row_i * cols_per_row : (row_i + 1) * cols_per_row]
            ):
                with cols[col_i]:
                    cm_c = cm_cohort_source[cm_cohort_source["construct"] == construct]
                    _grouped_distribution_chart(
                        cm_c,
                        value_col="mean_score",
                        group_col="cohort_id",
                        title=construct.replace("_", " ").title(),
                        y_label="Mean Score",
                        y_range=[1, 4],
                    )


# -----------------------------------------------------------------------
# Tab 1 — Inferential Statistics
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Statistics plain-language glossary (shown to teachers in UI)
# -----------------------------------------------------------------------
_STAT_HELP = {
    "cohens_d": (
        "**Cohen's d — Effect Size**\n\n"
        "Measures *how large* the difference between pre- and post-scores is, "
        "independent of sample size. Think of it as a ruler for practical importance:\n\n"
        "- **d < 0.2** — Negligible (barely noticeable)\n"
        "- **0.2 ≤ d < 0.5** — Small (noticeable but modest)\n"
        "- **0.5 ≤ d < 0.8** — Medium (meaningful in practice)\n"
        "- **d ≥ 0.8** — Large (substantial, clearly visible difference)\n\n"
        "A positive d means post-scores were higher than pre-scores. "
        "A negative d means scores went down."
    ),
    "eta_squared": (
        "**η² (Eta-squared) — Effect Size for Group Comparisons**\n\n"
        "Tells you what proportion of the total score variation is explained "
        "by group membership (e.g. cohort, grade). Ranges from 0 to 1:\n\n"
        "- **η² < 0.01** — Negligible\n"
        "- **0.01 ≤ η² < 0.06** — Small\n"
        "- **0.06 ≤ η² < 0.14** — Medium\n"
        "- **η² ≥ 0.14** — Large\n\n"
        "Example: η² = 0.10 means 10% of score differences can be attributed "
        "to which group students belong to."
    ),
    "kruskal_wallis": (
        "**Kruskal-Wallis Test**\n\n"
        "A non-parametric alternative to ANOVA — it tests whether scores "
        "differ across groups *without assuming* scores follow a normal "
        "distribution. Particularly appropriate for small samples.\n\n"
        "The p-value shown is from this test. If both ANOVA p and "
        "Kruskal-Wallis p are significant, the finding is more robust."
    ),
    "friedman": (
        "**Friedman χ² (Chi-square) — Repeated Measures Test**\n\n"
        "Tests whether scores change significantly across multiple time points "
        "(e.g. Module 1 → Module 7) for the *same* students. "
        "It is the non-parametric equivalent of repeated-measures ANOVA — "
        "appropriate when data are ordinal (like Likert scales) or samples are small.\n\n"
        "A significant result (p < 0.05) means scores did not stay constant "
        "across modules — at least one module was different."
    ),
    "kendalls_w": (
        "**Kendall's W — Effect Size for Repeated Measures**\n\n"
        "Accompanies the Friedman test. Measures the *consistency* of change "
        "across modules. Ranges from 0 to 1:\n\n"
        "- **W < 0.1** — Negligible consistency\n"
        "- **0.1 ≤ W < 0.3** — Weak\n"
        "- **0.3 ≤ W < 0.5** — Moderate\n"
        "- **W ≥ 0.5** — Strong\n\n"
        "A high W means students tended to rank modules similarly — "
        "their scores changed in a consistent pattern."
    ),
    "paired_t": (
        "**Paired t-test p-value**\n\n"
        "Tests whether the *average* difference between pre- and post-scores "
        "is statistically distinguishable from zero. "
        "A p-value below 0.05 means there is less than a 5% chance of seeing "
        "this difference by random chance alone.\n\n"
        "**Important:** A significant p-value does not tell you *how big* "
        "the difference is — always read Cohen's d alongside it."
    ),
    "wilcoxon": (
        "**Wilcoxon Signed-Rank Test**\n\n"
        "A non-parametric companion to the paired t-test. "
        "It makes no assumption about the shape of the score distribution, "
        "making it more appropriate for small samples or ordinal data. "
        "If both the t-test and Wilcoxon are significant, the result is more credible."
    ),
    "p_value": (
        "**p-value (Significance)**\n\n"
        "The probability of observing a result at least this extreme if there "
        "were truly no effect. Convention: p < 0.05 is considered statistically significant.\n\n"
        "- **p < 0.001** — Very strong evidence\n"
        "- **0.001 ≤ p < 0.01** — Strong evidence\n"
        "- **0.01 ≤ p < 0.05** — Moderate evidence\n"
        "- **p ≥ 0.05** — Insufficient evidence to reject no-effect hypothesis\n\n"
        "**Important caveat with small samples (n = 13):** A non-significant "
        "p-value does not mean there is no effect — it may simply mean the "
        "study was underpowered to detect it. Equally, a significant result "
        "from a small sample should be interpreted cautiously as it may reflect "
        "sampling variability rather than a true effect. Both the p-value "
        "and the effect size (Cohen's d / η² / Kendall's W) must be considered together."
    ),
}

_IRT_HELP = {
    "rasch": (
        "**Rasch Model (1-Parameter Logistic, 1PL)**\n\n"
        "The simplest IRT model. It estimates one parameter per item: "
        "**difficulty (b)** — how hard the item is on a logit scale. "
        "A student whose ability (θ) equals an item's difficulty has a 50% "
        "chance of answering correctly.\n\n"
        "Items with very high or very low difficulty estimates may be too hard "
        "or too easy for your students and worth reviewing.\n\n"
        "Runs at any sample size, though estimates are unstable below n = 30."
    ),
    "2pl": (
        "**2-Parameter Logistic Model (2PL)**\n\n"
        "Adds a second parameter per item: **discrimination (a)** — how well "
        "the item separates high- from low-ability students. "
        "A high discrimination value (a > 1.0) means the item reliably "
        "distinguishes students who understand the concept from those who do not.\n\n"
        "Requires n ≥ 50 for stable estimates. With n = 13, Rasch (1PL) is "
        "more appropriate — 2PL estimates will be unreliable."
    ),
    "grm": (
        "**Graded Response Model (GRM)**\n\n"
        "Extends IRT to Likert-scale survey items (1–4 responses). "
        "Instead of a single difficulty, each item has *threshold parameters* "
        "showing where students transition between response categories.\n\n"
        "The person ability parameter (θ) reflects the latent trait being "
        "measured — e.g. intrinsic motivation or engagement with task. "
        "Higher θ means stronger presence of the construct.\n\n"
        "Requires n ≥ 50 for reliable estimates."
    ),
    "theta": (
        "**Person Ability (θ, theta)**\n\n"
        "Each student's estimated position on the latent trait scale "
        "(measured in logits). Higher θ = higher ability or stronger construct. "
        "The scale is centred at 0 by convention — most students will fall "
        "between –3 and +3.\n\n"
        "The Wright Map plots student θ values alongside item difficulties, "
        "showing how well the test is targeted at your students."
    ),
    "aic": (
        "**AIC (Akaike Information Criterion)**\n\n"
        "A model fit index — lower AIC means better fit. "
        "Useful for comparing two models fitted to the same data "
        "(e.g. Rasch vs 2PL). A difference of more than 10 is considered "
        "meaningful. With small samples, AIC comparisons are unreliable."
    ),
    "icc": (
        "**Item Characteristic Curve (ICC)**\n\n"
        "Shows the probability of a correct response (y-axis) as a function "
        "of student ability θ (x-axis). An S-shaped curve is expected — "
        "low-ability students rarely answer correctly, high-ability students "
        "almost always do.\n\n"
        "A very flat ICC means the item barely discriminates between students. "
        "A very steep ICC means the item is highly discriminating."
    ),
    "wright_map": (
        "**Wright Map (Person-Item Map)**\n\n"
        "Places student ability estimates (blue dots) and item difficulties "
        "(orange diamonds) on the same logit scale. "
        "Items near the centre of the student distribution are optimally "
        "targeted. Items far above most students are too hard; "
        "items far below are too easy."
    ),
}

def _render_bland_altman_expander(
    ba: dict,
    score_label: str = "% Correct",
) -> None:
    '''Render a collapsible Bland-Altman method agreement expander.
    Displays:
      - Summary table: N, bias (d-bar), SD(diff), lower LoA, upper LoA
      - Per-participant difference table (inner expander)
      - Proportional bias note when p < 0.05
      - Full citation

    Parameters
    ----------
    ba : dict
        Output of run_bland_altman().
    score_label : str
        Unit label shown in column headers (e.g. "% Correct").
    '''
    if ba.get("error"):
        with st.expander("📐 Method agreement — Bland-Altman limits of agreement",
                         expanded=False):
            st.warning(f"Could not compute: {ba['error']}")
        return

    with st.expander(
        "📐 Method agreement — Bland-Altman limits of agreement",
        expanded=False,
    ):
        n      = ba["n_pairs"]
        d_bar  = ba["mean_diff"]
        s      = ba["sd_diff"]
        lo     = ba["loa_lower"]
        hi     = ba["loa_upper"]

        # ── Summary statistics table ──────────────────────────────────
        summary_df = pd.DataFrame([{
            "N pairs":      n,
            f"Bias d̄ ({score_label})":  f"{d_bar:+.3f}",
            f"SD(diff)":                f"{s:.3f}",
            f"LoA lower (d̄−2s)":       f"{lo:+.3f}",
            f"LoA upper (d̄+2s)":       f"{hi:+.3f}",
        }])
        st.dataframe(summary_df, hide_index=True, width="stretch")

        # ── Interpretation ────────────────────────────────────────────
        bias_dir = (
            "Post scores exceeded pre on average (gain post-intervention)."
            if d_bar > 0 else
            "Pre scores exceeded post on average (decline post-intervention)."
            if d_bar < 0 else
            "No average systematic difference between pre and post."
        )
        st.caption(
            f"{bias_dir}  "
            f"~95% of individual pre-post differences lie between "
            f"{lo:+.2f} and {hi:+.2f} {score_label}. "
            "Whether this range is acceptable for educational decision-making "
            "is a pedagogical judgment, not a statistical one."
        )

        # ── Proportional bias note ────────────────────────────────────
        r_val = ba.get("proportional_bias_r")
        r_p   = ba.get("proportional_bias_p")
        if ba.get("proportional_bias"):
            st.warning(
                f"**Proportional bias detected** (r = {r_val:.3f}, p = {r_p:.4f}). "
                "The size of the difference between pre and post scores grows "
                "with the magnitude of the scores. "
                "Consider applying a log transformation to raw scores before "
                "re-running the analysis."
            )
        elif r_val is not None:
            st.caption(
                f"No proportional bias (r = {r_val:.3f}, p = {r_p:.4f}). "
                "The limits of agreement apply uniformly across the score range."
            )

        if ba.get("low_n_warning"):
            st.caption(
                f"⚠️ n = {n} < {30}. Limits of agreement are wide estimates "
                "at this sample size. Effect sizes are more informative than "
                "the absolute LoA bounds."
            )

        # ── Per-participant difference table (collapsible) ────────────
        per_pair_df = ba.get("per_pair_df")
        if per_pair_df is not None and not per_pair_df.empty:
            with st.expander(
                f"Individual differences ({n} participants)", expanded=False
            ):
                display = per_pair_df.copy()
                display.columns = [
                    "Participant",
                    f"Pre ({score_label})",
                    f"Post ({score_label})",
                    "Diff (Post−Pre)",
                    "Mean (Pre+Post)/2",
                ]
                st.dataframe(display, hide_index=True, width="stretch")

        # ── Citation ──────────────────────────────────────────────────
        st.caption(
            "Method: Bland, J. M., & Altman, D. G. (1990). A note on the use "
            "of the intraclass correlation coefficient in the evaluation of "
            "agreement between two methods of measurement. "
            "*Computers in Biology and Medicine, 20*(5), 337–340. "
            "https://doi.org/10.1016/0010-4825(90)90013-F"
        )

def _render_result_card(result: dict, score_label: str = "% Correct") -> None:
    """Render a single test result as a clean card with source table and means."""
    if result.get("error"):
        st.error(f"Could not compute: {result['error']}")
        return

    sig       = result.get("significant", False)
    alpha     = result.get("alpha", 0.05)
    sig_badge = "✅ Significant" if sig else "— Not significant"

    def _fmt(v, decimals=3):
        return f"{v:.{decimals}f}" if v is not None else "—"

    # ================================================================
    # PAIRED t (Pre vs Post) — TC25
    # ================================================================
    if "pre_mean" in result:
        cols = st.columns(4)
        cols[0].metric("Pre mean",   f"{result['pre_mean']:.2f}")
        cols[1].metric("Post mean",  f"{result['post_mean']:.2f}",
                       delta=f"{result['mean_diff']:+.2f}")
        cols[2].metric("Cohen's d",
                       f"{result['cohens_d']:.3f} ({result['effect_size_label']})")
        cols[3].metric("Paired t p-value",
                       f"{result['t_p_value']:.4f}  {sig_badge}")

        # ── Means by time-point table ─────────────────────────────────
        n_p = result.get("n_pairs", 0)
        with st.expander("📈 Means by time-point", expanded=False):
            _tp_df = pd.DataFrame([
                {"Time-point": "Pre",  "N": str(n_p),
                 f"Mean {score_label}": _fmt(result["pre_mean"],  2)},
                {"Time-point": "Post", "N": str(n_p),
                 f"Mean {score_label}": _fmt(result["post_mean"], 2)},
                {"Time-point": "Change (Post − Pre)", "N": "—",
                 f"Mean {score_label}": _fmt(result["mean_diff"], 2)},
            ])
            st.dataframe(_tp_df, hide_index=True, width="stretch")

        # ── RM ANOVA-style source table for paired t ──────────────────
        # For a paired design: t = d·√n, so F = t² = d²·n
        # SS_time = F·MS_error;  std_diff = mean_diff/d;  MS_error = std_diff²
        import math as _math
        _d  = abs(result.get("cohens_d", 0) or 0)
        _md = abs(result.get("mean_diff", 0) or 0)
        # Accept multiple key names for n — some result dicts use n_subjects or n
        n_p = (result.get("n_pairs") or result.get("n_subjects")
               or result.get("n") or 0)
        # Recompute t² from d and n if t_stat not stored
        _t_raw = result.get("t_stat") or result.get("t_statistic") or result.get("t_value")
        if _t_raw is not None:
            _t2 = float(_t_raw) ** 2
        elif _d > 0 and n_p > 0:
            _t2 = (_d ** 2) * n_p
        else:
            _t2 = None
        # Show table whenever we have d and n — mean_diff not required.
        # F = d²·n is exact for a paired design regardless of mean_diff.
        # If mean_diff=0 exactly (no change), d=0 too, so _d>0 guards correctly.
        if _d > 0 and n_p > 0:
            _df_time  = 1
            _df_err   = max(int(n_p) - 1, 1)
            _F_pt     = _t2 if _t2 else ((_d ** 2) * n_p)
            # std_diff only needed for MS_error; derive from mean_diff/d if available
            if _md > 0:
                _std_diff = _md / _d
                _MS_err   = _std_diff ** 2
                _SS_err   = _df_err * _MS_err
                _SS_time  = _F_pt * _MS_err
                _MS_time  = _SS_time
            else:
                # mean_diff=0 ⟹ d=0 normally, but guard above passed so d>0.
                # Derive MS_error from F and df: MS_error = SS_total/(n-1)
                # Use approximation: MS_error = 1 (unit variance) when unavailable.
                _SS_err  = None
                _MS_err  = None
                _SS_time = None
                _MS_time = None
            with st.expander("📊 RM ANOVA-style source table", expanded=False):
                _src = pd.DataFrame([
                    {"Source": "Time (Pre→Post)", "SS": _fmt(_SS_time),
                     "df": str(_df_time), "MS": _fmt(_MS_time), "F": _fmt(_F_pt)},
                    {"Source": "Subjects (error)", "SS": _fmt(_SS_err),
                     "df": str(_df_err),  "MS": _fmt(_MS_err),  "F": "—"},
                ])
                st.dataframe(_src, hide_index=True, width="stretch",
                    column_config={c: st.column_config.Column(c, width="small")
                                   for c in _src.columns})
                st.caption(
                    "Derived from Cohen's d and n (paired design): F = d²·n = t². "
                    "SS and MS are exact under normality. "
                    "Treat as indicative with small samples — effect size (Cohen's d) "
                    "is more informative than the F ratio at low n."
                )

    # ================================================================
    # ONE-WAY ANOVA (Between Groups) — TC26
    # ================================================================
    elif "f_stat" in result:
        cols = st.columns(4)
        cols[0].metric("F statistic",   f"{result['f_stat']:.4f}")
        cols[1].metric("ANOVA p-value", f"{result['anova_p']:.4f}  {sig_badge}")
        cols[2].metric("η² (eta²)",
                       f"{result['eta_squared']:.4f} ({result['effect_size_label']})")
        cols[3].metric("Kruskal-Wallis p", f"{result['kruskal_p']:.4f}")

        # ── Means by group ────────────────────────────────────────────
        gm = result.get("group_means", {})
        ng = result.get("n_per_group", {})
        gs = result.get("group_stds",  {})
        if gm:
            with st.expander("📈 Means by group", expanded=False):
                _gdf = pd.DataFrame([
                    {"Group": g, "N": str(ng.get(g, "—")),
                     f"Mean {score_label}": _fmt(m, 2),
                     "SD": _fmt(gs.get(g, 0), 2)}
                    for g, m in gm.items()
                ])
                st.dataframe(_gdf, hide_index=True, width="stretch")

        # ── ANOVA source table (computed from group stats) ────────────
        if gm and ng and gs and len(gm) >= 2:
            import numpy as np
            _N_total  = sum(ng.values())
            _k        = len(gm)
            _grand    = sum(ng[g]*gm[g] for g in gm) / _N_total
            _SS_btwn  = sum(ng[g] * (gm[g] - _grand)**2  for g in gm)
            _SS_with  = sum((ng[g] - 1) * (gs[g]**2)     for g in gm)
            _df_btwn  = _k - 1
            _df_with  = _N_total - _k
            _MS_btwn  = _SS_btwn / _df_btwn if _df_btwn > 0 else None
            _MS_with  = _SS_with / _df_with  if _df_with  > 0 else None
            _F_bg     = (_MS_btwn / _MS_with) if (_MS_btwn and _MS_with and _MS_with != 0) else None
            with st.expander("📊 ANOVA source table", expanded=False):
                _src_bg = pd.DataFrame([
                    {"Source": "Between groups", "SS": _fmt(_SS_btwn),
                     "df": str(_df_btwn), "MS": _fmt(_MS_btwn), "F": _fmt(_F_bg)},
                    {"Source": "Within groups (error)", "SS": _fmt(_SS_with),
                     "df": str(_df_with), "MS": _fmt(_MS_with), "F": "—"},
                    {"Source": "Total",
                     "SS": _fmt(_SS_btwn + _SS_with) if _SS_btwn and _SS_with else "—",
                     "df": str(_N_total - 1), "MS": "—", "F": ""},
                ])
                st.dataframe(_src_bg, hide_index=True, width="stretch",
                    column_config={c: st.column_config.Column(c, width="small")
                                   for c in _src_bg.columns})
                st.caption(
                    "SS computed directly from group means, n, and SDs. "
                    "F here matches the ANOVA F above (minor rounding aside). "
                    "η² = SS_between / SS_total."
                )

    # ================================================================
    # FRIEDMAN (Across Modules / RM) — TC27 already complete
    # ================================================================
    elif "friedman_stat" in result:
        cols = st.columns(4)
        cols[0].metric("Friedman χ²", f"{result['friedman_stat']:.4f}")
        cols[1].metric("p-value",     f"{result['p_value']:.4f}  {sig_badge}")
        cols[2].metric("Kendall's W",
                       f"{result['kendalls_w']:.4f} ({result['effect_size_label']})")
        cols[3].metric("N subjects",  str(result.get("n_subjects", "")))

        # ── RM ANOVA-style source table ───────────────────────────────
        _n   = result.get("n_subjects", 0)
        _tp  = result.get("time_points", [])
        _k   = len(_tp) if _tp else 1
        _W   = result.get("kendalls_w",    0)
        _chi = result.get("friedman_stat", 0)
        if _n > 1 and _k > 1:
            _df_cond = _k - 1
            _df_subj = _n - 1
            _df_err  = (_k - 1) * (_n - 1)
            _SS_cond = round(_chi * (_k - 1) / _k, 3) if _k > 0 else None
            _SS_err  = round((_n * _k * (_k + 1) / 12) - _chi / (_k - 1), 3) if _k > 1 else None
            _MS_cond = round(_SS_cond / _df_cond, 3) if _SS_cond and _df_cond else None
            _MS_err  = round(_SS_err  / _df_err,  3) if _SS_err  and _df_err  else None
            _F_val   = round(_MS_cond / _MS_err,  3) if _MS_cond and _MS_err and _MS_err != 0 else None
            with st.expander("📊 RM ANOVA-style source table", expanded=False):
                _rm_tbl = pd.DataFrame([
                    {"Source": "Conditions (modules/time)",
                     "SS": _fmt(_SS_cond), "df": str(_df_cond),
                     "MS": _fmt(_MS_cond), "F":  _fmt(_F_val)},
                    {"Source": "Subjects",
                     "SS": "—", "df": str(_df_subj), "MS": "—", "F": "—"},
                    {"Source": "Error",
                     "SS": _fmt(_SS_err), "df": str(_df_err),
                     "MS": _fmt(_MS_err), "F": ""},
                ])
                st.dataframe(_rm_tbl, hide_index=True, width="stretch",
                    column_config={c: st.column_config.Column(c, width="small")
                                   for c in _rm_tbl.columns})
                st.caption(
                    "SS and MS are approximated from Friedman χ² and Kendall's W. "
                    "The Friedman test is non-parametric — treat these as indicative."
                )

        # ── Means by module / time-point ──────────────────────────────
        mbt = result.get("means_by_time", {})
        sbt = result.get("stds_by_time",  {})
        if mbt:
            with st.expander("📈 Means by module / time-point", expanded=False):
                _mbt_df = pd.DataFrame([
                    {"Module / Time-point": tp,
                     f"Mean {score_label}": _fmt(mbt[tp], 3),
                     "SD":                  _fmt(sbt.get(tp, 0), 3)}
                    for tp in sorted(mbt.keys())
                ])
                st.dataframe(_mbt_df, hide_index=True, width="stretch")

    # ================================================================
    # Plain-language interpretation (all test types)
    # ================================================================
    with st.expander("ℹ️ What do these numbers mean?", expanded=False):
        if "pre_mean" in result:
            st.markdown(_STAT_HELP["cohens_d"])
            st.divider()
            st.markdown(_STAT_HELP["paired_t"])
        elif "f_stat" in result:
            st.markdown(_STAT_HELP["eta_squared"])
            st.divider()
            st.markdown(_STAT_HELP["kruskal_wallis"])
        elif "friedman_stat" in result:
            st.markdown(_STAT_HELP["friedman"])
            st.divider()
            st.markdown(_STAT_HELP["kendalls_w"])
        st.divider()
        st.markdown(_STAT_HELP["p_value"])

    # Wilcoxon supplementary row
    if result.get("wilcoxon_stat") is not None:
        st.caption(
            f"Wilcoxon signed-rank: W={result['wilcoxon_stat']}, "
            f"p={result['wilcoxon_p']:.4f}"
        )
        with st.expander("ℹ️ What is the Wilcoxon test?", expanded=False):
            st.markdown(_STAT_HELP["wilcoxon"])

    # Low-N power warning
    if result.get("low_n_warning"):
        with st.expander("⚠️  Low-N Warning + Sample Size Guidance", expanded=True):
            n_shown = result.get("n_pairs") or result.get("n_subjects") or "?"
            st.warning(
                f"**n = {n_shown} students** — results have limited statistical "
                "power. Do not draw firm conclusions until the full cohort "
                "(n ≈ 90) is available."
            )
            st.markdown(
                "**Why post-hoc N estimates can look too small**\n\n"
                "Power calculations that use the *observed* effect size from a "
                "small sample are misleading — small samples routinely produce "
                "inflated effect estimates by chance, making the required N look "
                "unrealistically low (Lakens, 2022; Gelman & Carlin, 2014)."
            )
            st.markdown("**Prospective sample size requirements (α = 0.05)**")
            is_paired   = "pre_mean"      in result
            is_repeated = "friedman_stat" in result
            if is_paired:
                pwr_df = pd.DataFrame([
                    {"Cohen's d": "Small (0.2)",  "Practical meaning": "Subtle",
                     "N for 80% power": 199, "N for 95% power": 327},
                    {"Cohen's d": "Medium (0.5)", "Practical meaning": "Moderate — recommended minimum",
                     "N for 80% power": 34,  "N for 95% power": 55},
                    {"Cohen's d": "Large (0.8)",  "Practical meaning": "Substantial",
                     "N for 80% power": 15,  "N for 95% power": 24},
                ])
            elif is_repeated:
                pwr_df = pd.DataFrame([
                    {"Kendall's W": "Small (0.1)",  "Practical meaning": "Weak consistency",
                     "N for 80% power": ">200", "N for 95% power": ">300"},
                    {"Kendall's W": "Medium (0.3)", "Practical meaning": "Moderate — recommended minimum",
                     "N for 80% power": "~52",  "N for 95% power": "~85"},
                    {"Kendall's W": "Large (0.5)",  "Practical meaning": "Strong consistency",
                     "N for 80% power": "~21",  "N for 95% power": "~34"},
                ])
            else:
                pwr_df = pd.DataFrame([
                    {"Cohen's f": "Small (0.10)",  "η²": "≈0.01", "Practical meaning": "Subtle",
                     "N per group (80%)": 322, "N per group (95%)": 527},
                    {"Cohen's f": "Medium (0.25)", "η²": "≈0.06", "Practical meaning": "Moderate — recommended minimum",
                     "N per group (80%)": 52,  "N per group (95%)": 85},
                    {"Cohen's f": "Large (0.40)",  "η²": "≈0.14", "Practical meaning": "Substantial",
                     "N per group (80%)": 21,  "N per group (95%)": 34},
                ])
            st.dataframe(pwr_df, hide_index=True, width="stretch")
            power_val = result.get("power_achieved")
            n_shown_p = result.get("n_pairs") or result.get("n_subjects") or "?"
            st.caption(
                f"With your planned n ≈ 90, you will have adequate power to "
                f"detect medium-to-large effects. The observed post-hoc power "
                f"at n = {n_shown_p} is shown below for reference only — "
                f"it should not be used to justify your sample size."
            )
            if power_val is not None:
                st.caption(
                    f"Observed post-hoc power (reference only): "
                    f"**{power_val*100:.1f}%** at n = {n_shown_p}"
                )


def _render_inferential_tab(
    canonical_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
) -> None:
    """Tab 1 — Inferential Statistics."""

    st.subheader("📈 Inferential Statistics")
    st.caption(
        "All tests use α = 0.05. With small n, effect sizes are more "
        "informative than p-values."
    )

    # ---- Section selector via radio (avoids Streamlit sub-tab reset bug) ----
    if "inf_section_radio" not in st.session_state:
        st.session_state["inf_section_radio"] = "Pre vs Post"
    section = st.radio(
        "**Analysis type:**",
        options=["Pre vs Post", "Between Groups", "Across Modules"],
        horizontal=True,
        key="inf_section_radio",
    )
    _INF_COLORS = {
        "Pre vs Post":     ("#0077BB", "#E6F3FB"),
        "Between Groups":  ("#EE7733", "#FFF3E6"),
        "Across Modules":  ("#009E73", "#E6F7F1"),
    }
    _ic, _ibg = _INF_COLORS.get(section, ("#333","#F8F8F8"))
    st.markdown(
        f"<div style='background:{_ibg};border-left:5px solid {_ic};"
        f"border-radius:6px;padding:0.4rem 1rem;margin:0.3rem 0;'>"
        f"<strong style='color:{_ic};'>Showing: {section}</strong></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ================================================================
    # Section A: Pre vs Post (paired comparisons)
    # ================================================================
    if section == "Pre vs Post":
        st.markdown("### Pre vs Post Comparisons")

        instrument_pair = st.selectbox(
            "Select instrument pair",
            options=[
                "AI Misconceptions",
                "AI Conceptual Inventory (AICI)",
            ],
            key="inf_pair_select",
        )

        show_wilcoxon = st.checkbox(
            "Show Wilcoxon signed-rank test",
            value=False,
            key="inf_wilcoxon_toggle",
        )

        pair_map = {
            "AI Misconceptions": (
                "precourse_pre_ai_misconceptions_assessment",
                "postcourse_post_ai_misconceptions_assessment",
            ),
            "AI Conceptual Inventory (AICI)": (
                "precourse_pre_aici_assessment",
                "postcourse_post_aici_assessment",
            ),
        }
        pre_key, post_key = pair_map[instrument_pair]

        with st.spinner("Computing…"):
            r = run_paired_comparison(
                canonical_df, pre_key, post_key,
                alpha=0.05,
                include_wilcoxon=show_wilcoxon,
                use_pct=True,
            )
        _render_result_card(r, score_label="% Correct")

        # ── Bland-Altman method agreement ─────────────────────────────────
        with st.spinner("Computing method agreement…"):
            _ba = run_bland_altman(
                canonical_df, pre_key, post_key, use_pct=True
            )
        _render_bland_altman_expander(_ba, score_label="% Correct")
        # ─────────────────────────────────────────────────────────────────

    # ================================================================
    # Section B: Between Groups
    # ================================================================
    elif section == "Between Groups":
        st.markdown("### Between-Groups Comparison")

        col1, col2 = st.columns(2)
        with col1:
            bg_instrument = st.selectbox(
                "Instrument",
                options=list(_ASSESSMENT_LABELS.values()),
                key="inf_bg_instrument",
            )
            bg_key = next(
                k for k, v in _ASSESSMENT_LABELS.items()
                if v == bg_instrument
            )
        with col2:
            bg_group = st.selectbox(
                "Group by",
                options=["grade", "gender",
                         "first_language_english", "cohort_id"],
                key="inf_bg_group",
            )

        with st.spinner("Computing…"):
            r_bg = run_between_groups(
                canonical_df, bg_key,
                group_col=bg_group,
                demographics_df=demographics_df,
                alpha=0.05, use_pct=True,
            )

        # Graceful single-group message (e.g. only one cohort)
        if r_bg.get("error") and "Need at least 2 groups" in str(r_bg["error"]):
            st.info(
                f"Cannot run between-groups test for **{bg_group}**: "
                f"only one group has data ({r_bg['error'].split('Found: ')[-1]}). "
                f"This test requires at least 2 groups."
            )
        else:
            _render_result_card(r_bg, score_label="% Correct")

    # ================================================================
    # Section C: Across Modules (repeated measures)
    # ================================================================
    else:
        st.markdown("### Across Modules — Repeated Measures (Friedman Test)")

        if "inf_rm_type" not in st.session_state:
            st.session_state["inf_rm_type"] = "Survey construct"
        rm_type = st.radio(
            "**Data type:**",
            options=["MCQ content knowledge", "Survey construct"],
            horizontal=True,
            key="inf_rm_type",
        )
        _RT_COLORS = {
            "MCQ content knowledge": ("#0077BB","#E6F3FB"),
            "Survey construct":      ("#CC79A7","#F9EEF5"),
        }
        _rtc, _rtbg = _RT_COLORS.get(rm_type, ("#333","#F8F8F8"))
        st.markdown(
            f"<div style='background:{_rtbg};border-left:4px solid {_rtc};"
            f"border-radius:5px;padding:0.3rem 0.8rem;margin:0.3rem 0;'>"
            f"<strong style='color:{_rtc};'>Analysis: {rm_type}</strong></div>",
            unsafe_allow_html=True,
        )

        if rm_type == "MCQ content knowledge":
            with st.spinner("Computing…"):
                r_rm = run_repeated_measures(
                    canonical_df,
                    instrument_key="content_mcq_assessment",
                    construct=None, alpha=0.05,
                )
            _render_result_card(r_rm, score_label="% Correct")

        else:
            col1, col2 = st.columns(2)
            with col1:
                rm_survey = st.selectbox(
                    "Survey",
                    options=["SCCCES", "SIMS"],
                    key="inf_rm_survey",
                )
            survey_key = (
                "b4ai_sccces_survey" if rm_survey == "SCCCES"
                else "b4ai_sims_survey"
            )
            sccces_constructs = [
                "engagement_with_task","effort_and_persistence",
                "experience_of_flow","coherency_of_messaging",
                "plausibility_of_messaging","credibility_of_messaging",
                "comprehensibility_of_messaging","attention",
                "culture","personal_relevance",
            ]
            sims_constructs = [
                "intrinsic_motivation","identified_regulation",
                "external_regulation","amotivation",
            ]
            constructs = (
                sccces_constructs if rm_survey == "SCCCES"
                else sims_constructs
            )
            with col2:
                rm_construct = st.selectbox(
                    "Construct",
                    options=constructs,
                    key="inf_rm_construct",
                )

            with st.spinner("Computing…"):
                r_rm = run_repeated_measures(
                    canonical_df,
                    instrument_key=survey_key,
                    construct=rm_construct,
                    alpha=0.05,
                )
            _render_result_card(r_rm, score_label="Mean Score")


# -----------------------------------------------------------------------
# Tab 4 — Competency Progression (CPI)
# -----------------------------------------------------------------------

def _render_cpi_tab(username: str, canonical_df: pd.DataFrame) -> None:
    """
    Tab 4 — Competency Progression Index (CPI).

    CPI_quant: MCQ performance [0, 1]
      - CTT: proportion correct (Crocker & Algina 1986)
      - IRT: sigmoid-normalized EAP theta (Baker 1985)

    CPI_qual: Reflection quality [0, 1]
      - LLM-as-judge, 3 dimensions x 1-4 scale (Zheng et al. 2023)

    CPI+ = w1 * CPI_quant + w2 * CPI_qual
    """
    try:
        from core.analytics.cpi.cpi_engine import (
            compute_cpi_quant_ctt,
            compute_cpi_quant_irt,
            compute_cpi_outcome,
            get_reflection_texts,
            score_reflection_llm,
            compute_cpi_qual_from_scores,
            compute_cpi_combined,
            cpi_summary_stats,
        )
        from core.analytics.cpi.cpi_store import (
            create_cpi_run,
            save_cpi_qual_result,
            save_cpi_summary,
            load_cpi_summary,
            list_cpi_runs,
            update_cpi_run_status,
        )
        from core.analytics.llm.llm_clients import call_model, get_available_models
        _CPI_AVAILABLE = True
    except ImportError as _cpi_imp_err:
        st.error(
            f"CPI module not fully installed: {_cpi_imp_err}. "
            "Ensure core/analytics/cpi/ is present."
        )
        return

    st.subheader("📉 Competency Progression Index (CPI)")
    st.caption(
        "CPI combines quantitative MCQ performance (CPI_quant) and "
        "qualitative reflection quality (CPI_qual) into a composite "
        "index per student per module."
    )

    with st.expander("ℹ️ CPI methodology", expanded=False):
        st.markdown(
            "**CPI_quant — Quantitative Task Performance**\n\n"
            "- *CTT approach*: proportion of MCQ items answered correctly "
            "(Crocker & Algina, 1986).\n"
            "- *IRT approach*: EAP ability estimate (θ) normalized via sigmoid "
            "to [0, 1]. Accounts for item difficulty and discrimination "
            "(Baker, 1985).\n\n"
            "**CPI_qual — Qualitative Reflection Quality**\n\n"
            "An LLM scores each student's module reflection on three dimensions "
            "(1–4 scale, calibrated for ages 10–14):\n"
            "- Depth of insight\n"
            "- Conceptual grounding\n"
            "- Personal connection\n\n"
            "CPI_qual = sum(dimension scores) / max possible score. "
            "Method: Zheng et al. (2023) LLM-as-judge framework.\n\n"
            "**CPI+ = w₁ × CPI_quant + w₂ × CPI_qual**\n\n"
            "Default weights: w₁ = w₂ = 0.5. Adjust below."
        )

    st.divider()

    # ── Step 1: Configuration ──────────────────────────────────────────────
    st.markdown("### Step 1 — Configure")

    _all_module_ids  = [f"module_{n}" for n in discover_all_module_numbers()]
    _module_display  = {mid: mid.replace("module_", "Module ") for mid in _all_module_ids}
    _selected_labels = st.multiselect(
        "Modules — combine one or more (defaults to all)",
        options=list(_module_display.values()),
        default=list(_module_display.values()),
        key="cpi_module_multisel",
    )
    _module_ids = sorted(
        (mid for mid, label in _module_display.items() if label in _selected_labels),
        key=lambda m: int(m.split("_")[-1]) if m.split("_")[-1].isdigit() else 0,
    )
    _module_labels_combined = ", ".join(_module_display[m] for m in _module_ids) or "(none selected)"

    st.markdown("**Instrument types for CPI_quant** (select one or more):")
    _c1, _c2, _c3 = st.columns(3)
    _use_mcq  = _c1.checkbox("Content MCQ", value=True, key="cpi_use_mcq")
    _use_misc = _c2.checkbox("AI Misconceptions gain (Post−Pre)", value=False, key="cpi_use_misc")
    _use_aici = _c3.checkbox("AICI gain (Post−Pre)", value=False, key="cpi_use_aici")
    _instrument_types = [t for use, t in (
        (_use_mcq, "content_mcq"),
        (_use_misc, "misconceptions_gain"),
        (_use_aici, "aici_gain"),
    ) if use]

    # Cohort scope: read-only, driven entirely by the sidebar's existing
    # Cohort filter (already applied to canonical_df before this tab
    # renders) -- no separate CPI-local selector, per the confirmed design
    # (CPI only ever deals with registered users, unlike LLM Analysis).
    _active_cohorts = (
        sorted(canonical_df["cohort_id"].dropna().unique())
        if "cohort_id" in canonical_df.columns else []
    )
    st.caption(
        "**Cohort scope:** "
        + (", ".join(_active_cohorts) if _active_cohorts else "All cohorts")
        + "  _(set via the sidebar's Cohort filter)_"
    )

    if not _module_ids:
        st.warning("Select at least one module.")
        return
    if not _instrument_types:
        st.warning("Select at least one instrument type.")
        return

    # "Legacy" path = the original single-module/MCQ-only behavior,
    # preserved exactly (including IRT support) for backward compatibility.
    # Any multi-module or multi-instrument-type selection routes through
    # the newer, configurable compute_cpi_outcome() instead -- CTT-style
    # % correct / gain-score blending only. Pooling heterogeneous
    # instruments into a single IRT fit is a separate, larger question the
    # user didn't ask for and is out of scope this round.
    _is_legacy = (len(_module_ids) == 1 and _instrument_types == ["content_mcq"])
    _module_id    = _module_ids[0]
    _module_label = _module_display[_module_id]
    _inst_key     = f"module{_module_id.split('_')[-1]}_content_mcq_assessment"

    # Scope strings used consistently for session-state, run storage, and
    # the past-runs filter, whichever path is active.
    _module_scope_str = _module_id if _is_legacy else ",".join(_module_ids)
    _inst_scope_str   = _inst_key if _is_legacy else ",".join(_instrument_types)

    if _is_legacy:
        col_mod, col_inst = st.columns(2)
        with col_mod:
            st.text_input("Module", value=_module_label, disabled=True)
        with col_inst:
            st.text_input(
                "MCQ instrument key", value=_inst_key,
                key="cpi_inst_key", disabled=True
            )

        col_q, col_irt = st.columns(2)
        with col_q:
            _quant_method = st.radio(
                "CPI_quant method",
                options=["CTT (proportion correct)", "IRT (Rasch)", "IRT (2PL, n≥50)",
                         "Both — show side by side"],
                horizontal=False,
                key="cpi_quant_method",
            )
        with col_irt:
            _w1 = st.slider("Weight w₁ (CPI_quant)", 0.0, 1.0, 0.5, 0.05, key="cpi_w1")
            _w2 = round(1.0 - _w1, 2)
            st.caption(f"Weight w₂ (CPI_qual) = {_w2:.2f}  (w₁ + w₂ = 1.0)")
    else:
        st.caption(
            f"**Combining:** {_module_labels_combined} — "
            + ", ".join(t.replace('_', ' ') for t in _instrument_types)
        )
        _quant_method = None
        _w1 = st.slider("Weight w₁ (CPI_quant)", 0.0, 1.0, 0.5, 0.05, key="cpi_w1")
        _w2 = round(1.0 - _w1, 2)
        st.caption(f"Weight w₂ (CPI_qual) = {_w2:.2f}  (w₁ + w₂ = 1.0)")

    st.divider()

    # ── Step 2: CPI_quant (immediate, no LLM needed) ──────────────────────
    st.markdown("### Step 2 — CPI_quant (MCQ / gain-score performance)")

    if st.button("Compute CPI_quant", key="cpi_compute_quant", type="primary"):
        with st.spinner("Computing performance scores…"):
            if _is_legacy:
                _ctt_df = compute_cpi_quant_ctt(canonical_df, _inst_key)

                _irt_rasch = None
                _irt_2pl   = None
                _run_rasch = _quant_method in (
                    "IRT (Rasch)", "Both — show side by side"
                )
                _run_2pl   = _quant_method in (
                    "IRT (2PL, n≥50)", "Both — show side by side"
                )

                if _run_rasch:
                    _irt_rasch = compute_cpi_quant_irt(
                        canonical_df, _inst_key, irt_model="rasch"
                    )
                if _run_2pl:
                    _irt_2pl = compute_cpi_quant_irt(
                        canonical_df, _inst_key, irt_model="2pl"
                    )
            else:
                _outcome_df = compute_cpi_outcome(
                    canonical_df, module_ids=_module_ids,
                    instrument_types=_instrument_types,
                )
                _ctt_df = _outcome_df.rename(columns={"cpi_outcome": "cpi_quant_ctt"}).copy()
                _ctt_df["n_items"] = _outcome_df["n_modules_mcq"]
                _ctt_df["method"]  = "Outcome (" + ", ".join(_instrument_types) + ")"
                _ctt_df = (
                    _ctt_df[["user_id", "cpi_quant_ctt", "n_items", "method"]]
                    .dropna(subset=["cpi_quant_ctt"])
                    .reset_index(drop=True)
                )
                _irt_rasch = None
                _irt_2pl   = None

        st.session_state["_cpi_ctt_df"]    = _ctt_df
        st.session_state["_cpi_irt_rasch"] = _irt_rasch
        st.session_state["_cpi_irt_2pl"]   = _irt_2pl
        st.session_state["_cpi_inst_key"]  = _inst_scope_str
        st.session_state["_cpi_module_id"] = _module_scope_str
        st.session_state["_cpi_is_legacy"] = _is_legacy

    _ctt_df    = st.session_state.get("_cpi_ctt_df")
    _irt_rasch = st.session_state.get("_cpi_irt_rasch")
    _irt_2pl   = st.session_state.get("_cpi_irt_2pl")

    _ctt_is_legacy = st.session_state.get("_cpi_is_legacy", True)
    _ctt_short_label = "CTT" if _ctt_is_legacy else "Outcome"

    if _ctt_df is not None and not _ctt_df.empty:

        # CTT / Outcome display
        with st.expander(
            f"{_ctt_short_label} — "
            + ("proportion correct" if _ctt_is_legacy else "combined score")
            + " per student", expanded=True,
        ):
            _n_ctt = len(_ctt_df)
            col1, col2, col3 = st.columns(3)
            col1.metric(f"Students ({_ctt_short_label})", _n_ctt)
            col2.metric(f"Mean CPI_quant ({_ctt_short_label})",
                        f"{_ctt_df['cpi_quant_ctt'].mean():.3f}")
            col3.metric("SD",
                        f"{_ctt_df['cpi_quant_ctt'].std(ddof=1):.3f}"
                        if _n_ctt > 1 else "—")
            _display_ctt = _ctt_df.copy()
            _display_ctt.columns = [
                "Student", f"CPI_quant ({_ctt_short_label})", "Items/modules", "Method"
            ]
            st.dataframe(
                _display_ctt.sort_values(f"CPI_quant ({_ctt_short_label})", ascending=False),
                hide_index=True, width="stretch",
            )

        # IRT Rasch display
        if _irt_rasch is not None:
            if _irt_rasch.get("error"):
                st.warning(f"IRT (Rasch): {_irt_rasch['error']}")
            else:
                with st.expander(
                    f"IRT (Rasch) — θ→sigmoid per student "
                    f"(n={_irt_rasch['n_persons']})", expanded=False
                ):
                    if _irt_rasch.get("low_n_warning"):
                        st.caption(
                            f"⚠️ n < 100. Rasch estimates are exploratory "
                            f"at this sample size."
                        )
                    _pp = _irt_rasch["person_df"]
                    _pp_disp = _pp[
                        ["user_id", "theta", "theta_se", "cpi_quant_irt"]
                    ].copy()
                    _pp_disp.columns = [
                        "Student", "θ (logit)", "θ SE", "CPI_quant (IRT)"
                    ]
                    st.dataframe(
                        _pp_disp.sort_values("CPI_quant (IRT)", ascending=False),
                        hide_index=True, width="stretch",
                    )

        # IRT 2PL display
        if _irt_2pl is not None:
            if _irt_2pl.get("error"):
                st.warning(f"IRT (2PL): {_irt_2pl['error']}")
            else:
                with st.expander(
                    f"IRT (2PL) — θ→sigmoid per student "
                    f"(n={_irt_2pl['n_persons']})", expanded=False
                ):
                    _pp2 = _irt_2pl["person_df"]
                    _pp2_disp = _pp2[
                        ["user_id", "theta", "theta_se", "cpi_quant_irt"]
                    ].copy()
                    _pp2_disp.columns = [
                        "Student", "θ (logit)", "θ SE", "CPI_quant (IRT)"
                    ]
                    st.dataframe(
                        _pp2_disp.sort_values("CPI_quant (IRT)", ascending=False),
                        hide_index=True, width="stretch",
                    )

    st.divider()

    # ── Step 3: CPI_qual (LLM scoring) ────────────────────────────────────
    st.markdown("### Step 3 — CPI_qual (reflection quality via LLM)")

    _avail_models = get_available_models(check_keys=True)
    _model_opts   = [m for m, avail in _avail_models.items() if avail]

    if not _model_opts:
        st.warning(
            "No LLM API keys configured. Add at least one of "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or "
            "GROQ_API_KEY to .env to enable CPI_qual scoring."
        )
    else:
        col_m, col_t = st.columns(2)
        with col_m:
            _cpi_model = st.selectbox(
                "LLM for scoring",
                options=_model_opts,
                key="cpi_qual_model",
            )
        with col_t:
            _cpi_temp = st.slider(
                "Temperature", 0.0, 0.5, 0.0, 0.1,
                key="cpi_qual_temp",
                help="0.0 = fully deterministic scoring (recommended).",
            )

        _refl_df = get_reflection_texts(canonical_df, module_ids=_module_ids)
        n_refl   = len(_refl_df["user_id"].unique()) if not _refl_df.empty else 0

        if _refl_df.empty:
            st.info(
                f"No reflection responses found for {_module_labels_combined}. "
                "Reflections must be submitted by students before "
                "CPI_qual can be computed."
            )
        else:
            n_resp = len(_refl_df)
            st.caption(
                f"Found {n_resp} reflection response(s) from "
                f"{n_refl} student(s), pooled across: {_module_labels_combined}."
            )

            if st.button(
                f"Score {n_resp} reflection(s) with {_cpi_model.upper()}",
                key="cpi_score_btn",
                type="primary",
            ):
                _run_id = create_cpi_run(
                    created_by=username,
                    model=_cpi_model,
                    module_id=st.session_state.get(
                        "_cpi_module_id", _module_scope_str
                    ),
                    instrument_key=st.session_state.get(
                        "_cpi_inst_key", _inst_scope_str
                    ),
                    temperature=_cpi_temp,
                    w1=_w1,
                    w2=_w2,
                )
                update_cpi_run_status(_run_id, "scoring")

                _scored_list = []
                _errors      = []
                _progress    = st.progress(0)
                _status_txt  = st.empty()

                for i, row in enumerate(_refl_df.itertuples(), 1):
                    _status_txt.caption(
                        f"Scoring {i}/{n_resp}: "
                        f"{row.user_id} / {row.question_id}…"
                    )
                    _s = score_reflection_llm(
                        participant_id=row.user_id,
                        question_id=row.question_id,
                        text=row.text,
                        module_id=row.module_id,
                        model=_cpi_model,
                        call_fn=call_model,
                        temperature=_cpi_temp,
                    )
                    _scored_list.append(_s)
                    save_cpi_qual_result(_run_id, _s)
                    if _s.get("error"):
                        _errors.append(
                            f"{row.user_id}/{row.question_id}: {_s['error']}"
                        )
                    _progress.progress(i / n_resp)

                _status_txt.empty()
                _progress.empty()
                update_cpi_run_status(_run_id, "done")

                # Aggregate CPI_qual
                _qual_df = compute_cpi_qual_from_scores(_scored_list)
                st.session_state["_cpi_run_id"]  = _run_id
                st.session_state["_cpi_qual_df"] = _qual_df

                if _errors:
                    st.warning(
                        f"{len(_errors)} scoring error(s):\\n" +
                        "\\n".join(_errors[:5])
                    )
                else:
                    st.success(
                        f"Scored {n_resp} reflections. "
                        f"Run ID: {_run_id[:8]}…"
                    )
                st.rerun()

    # ── Past runs selector ────────────────────────────────────────────────
    # Matches against the same module/instrument scope string this session
    # would itself save (see _module_scope_str above) -- runs from before
    # Task F (2026-08-09) are single "module_N" values, indistinguishable
    # from a 1-item comma-joined list, so old runs still match correctly.
    _past_runs = [
        r for r in list_cpi_runs()
        if r["module_id"] == _module_scope_str
    ]
    if _past_runs:
        st.divider()
        with st.expander(
            f"Load past CPI_qual run for {_module_labels_combined} "
            f"({len(_past_runs)} available)",
            expanded=False
        ):
            _run_labels = {
                f"{r['created_at'][:16]}  {r['model'].upper()}  "
                f"[{r['status']}]  {r['module_id']}  {r['run_id'][:8]}": r["run_id"]
                for r in _past_runs
            }
            _sel_label = st.selectbox(
                "Select run",
                options=list(_run_labels.keys()),
                key="cpi_past_run_sel",
            )
            if st.button("Load this run", key="cpi_load_run"):
                _run_id  = _run_labels[_sel_label]
                _raw     = load_cpi_qual_results(_run_id)
                if not _raw.empty:
                    # Reconstruct scored_list format for aggregation
                    _grp = (
                        _raw.groupby(["participant_id", "question_id"])
                        .apply(lambda g: {
                            "participant_id": g["participant_id"].iloc[0],
                            "question_id":    g["question_id"].iloc[0],
                            "scores": dict(zip(g["dimension"], g["score"])),
                            "error":  None,
                        })
                        .tolist()
                    )
                    _qual_df = compute_cpi_qual_from_scores(_grp)
                    st.session_state["_cpi_run_id"]  = _run_id
                    st.session_state["_cpi_qual_df"] = _qual_df
                    st.rerun()

    st.divider()

    # ── Step 4: CPI+ combined ─────────────────────────────────────────────
    st.markdown("### Step 4 — CPI+ (combined index)")

    _ctt_df_stored  = st.session_state.get("_cpi_ctt_df")
    _irt_r_stored   = st.session_state.get("_cpi_irt_rasch")
    _qual_df_stored = st.session_state.get("_cpi_qual_df")

    if _ctt_df_stored is None:
        st.info("Complete Step 2 first to compute CPI_quant.")
    elif _qual_df_stored is None:
        st.info("Complete Step 3 first to compute CPI_qual.")
    else:
        # Choose quant source. _quant_method is None on the newer
        # multi-module/multi-instrument path (compute_cpi_outcome has no
        # IRT option), so guard before calling .startswith() on it.
        if (
            _quant_method and _quant_method.startswith("IRT (Rasch)")
            and _irt_r_stored and not _irt_r_stored.get("error")
        ):
            _quant_src = _irt_r_stored["person_df"][["user_id", "cpi_quant_irt"]].rename(
                columns={"cpi_quant_irt": "cpi_quant_ctt"}
            )
            _quant_label = "IRT (Rasch)"
        else:
            _quant_src   = _ctt_df_stored
            _quant_label = "CTT" if st.session_state.get("_cpi_is_legacy", True) else "Outcome"

        _combined = compute_cpi_combined(
            _quant_src, _qual_df_stored,
            quant_col="cpi_quant_ctt",
            w1=_w1, w2=_w2,
        )

        if not _combined.empty:
            _stats = cpi_summary_stats(_combined)
            st.markdown(f"**CPI+ = {_w1} × CPI_quant ({_quant_label}) + {_w2} × CPI_qual**")

            # Summary metrics
            _cols = st.columns(3)
            for i, col_name in enumerate(["cpi_quant", "cpi_qual", "cpi_plus"]):
                if col_name in _stats:
                    _s = _stats[col_name]
                    _cols[i].metric(
                        col_name.replace("_", " ").upper(),
                        f"{_s['mean']:.3f}",
                        delta=f"SD {_s['sd']:.3f}",
                    )

            # Full table
            with st.expander("Per-student CPI+ table", expanded=True):
                _disp = _combined.copy()
                _disp.columns = [
                    "Student",
                    f"CPI_quant ({_quant_label})",
                    "CPI_qual",
                    "CPI+",
                    "w₁",
                    "w₂",
                ]
                st.dataframe(
                    _disp.sort_values("CPI+", ascending=False),
                    hide_index=True, width="stretch",
                    column_config={
                        "CPI+": st.column_config.ProgressColumn(
                            "CPI+", min_value=0, max_value=1, format="%.3f"
                        ),
                    },
                )

                # Save to store
                _run_id_store = st.session_state.get("_cpi_run_id", "")
                if _run_id_store:
                    _ctt_lookup = (
                        _ctt_df_stored.set_index("user_id")["cpi_quant_ctt"]
                        .to_dict() if _ctt_df_stored is not None else {}
                    )
                    _irt_lookup: dict = {}
                    if _irt_r_stored and not _irt_r_stored.get("error"):
                        _irt_lookup = (
                            _irt_r_stored["person_df"]
                            .set_index("user_id")["cpi_quant_irt"]
                            .to_dict()
                        )
                    _theta_lookup:    dict = {}
                    _theta_se_lookup: dict = {}
                    if _irt_r_stored and not _irt_r_stored.get("error"):
                        _theta_lookup = (
                            _irt_r_stored["person_df"]
                            .set_index("user_id")["theta"].to_dict()
                        )
                        if "theta_se" in _irt_r_stored["person_df"].columns:
                            _theta_se_lookup = (
                                _irt_r_stored["person_df"]
                                .set_index("user_id")["theta_se"].to_dict()
                            )

                    for _, row in _combined.iterrows():
                        uid = row["user_id"]
                        save_cpi_summary(
                            run_id=_run_id_store,
                            participant_id=uid,
                            module_id=st.session_state.get("_cpi_module_id", _module_scope_str),
                            instrument_key=st.session_state.get("_cpi_inst_key", _inst_scope_str),
                            cpi_quant_ctt=_ctt_lookup.get(uid),
                            cpi_quant_irt=_irt_lookup.get(uid),
                            cpi_quant=row.get("cpi_quant"),
                            cpi_qual=row.get("cpi_qual"),
                            cpi_plus=row.get("cpi_plus"),
                            n_mcq_items=int(
                                _ctt_df_stored.set_index("user_id")
                                .get("n_items", pd.Series(dtype=int))
                                .get(uid, 0)
                            ) if _ctt_df_stored is not None else 0,
                            theta=_theta_lookup.get(uid),
                            theta_se=_theta_se_lookup.get(uid),
                            quant_method=_quant_label,
                            w1=_w1,
                            w2=_w2,
                        )

    st.divider()
    st.caption(
        "References: "
        "Crocker & Algina (1986). *Introduction to Classical and Modern Test Theory*. "
        "Holt, Rinehart and Winston. — "
        "Baker (1985). *The Basics of Item Response Theory*. Heinemann. — "
        "Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. "
        "arXiv. https://doi.org/10.48550/ARXIV.2306.05685"
    )



# -----------------------------------------------------------------------
# Placeholder helper (for future tabs)
# -----------------------------------------------------------------------

def _render_placeholder(title: str, phase: int, bullets: list) -> None:
    st.markdown(f"### {title}")
    st.info(f"🔧 Coming in **Phase {phase}**")
    st.markdown("**Planned capabilities:**")
    for b in bullets:
        st.markdown(f"- {b}")


# -----------------------------------------------------------------------
# v. Competency Progression PDF Report
# -----------------------------------------------------------------------

def _report_cpi(canonical_df: pd.DataFrame) -> None:
    st.markdown("### v. Competency Progression Report")
    st.caption("Generates a PDF of CPI+ scores per student with component breakdown.")

    # Explicit run picker (added 2026-08-09, Task F) -- previously this
    # always silently used whichever run was globally most recent
    # (list_cpi_runs()[0]), with no way to pick a specific one. Now that a
    # run's module_id/instrument_key can cover any combination of modules
    # and instrument types, "just take the latest" is no longer a safe
    # default -- the teacher needs to see and choose which scope they're
    # reporting on.
    _selected_run_id = None
    try:
        from core.analytics.cpi.cpi_store import list_cpi_runs
        _all_runs = list_cpi_runs()
    except Exception:
        _all_runs = []

    if _all_runs:
        _run_choice_labels = {"Most recent run": None}
        for r in _all_runs:
            _run_choice_labels[
                f"{r['created_at'][:16]}  {r['model'].upper()}  "
                f"[{r['status']}]  scope: {r['module_id']}  {r['run_id'][:8]}"
            ] = r["run_id"]
        _picked_label = st.selectbox(
            "Which stored CPI run to report on:",
            options=list(_run_choice_labels.keys()),
            key="rpt_cpi_run_choice",
            help="Runs now show their module/instrument scope, since a run can "
                 "cover any combination the teacher configured in the CPI tab.",
        )
        _selected_run_id = _run_choice_labels[_picked_label]
    else:
        st.caption("No stored CPI runs yet — will compute the 3-component fallback below.")

    st.session_state.setdefault("rpt_cpi_include_chart", True)
    st.multiselect(
        "Include in report:",
        options=["Per-student CPI+ table", "Component interpretation", "Methodology note"],
        default=["Per-student CPI+ table", "Component interpretation", "Methodology note"],
        key="rpt_cpi_multiselect",
    )

    if st.button("📄 Generate CPI+ Report (PDF)", key="rpt_cpi_gen", type="primary"):
        selected = st.session_state.get(
            "rpt_cpi_multiselect",
            ["Per-student CPI+ table", "Component interpretation", "Methodology note"]
        )
        with st.spinner("Loading CPI data and building PDF…"):
            # Try to load saved CPI summaries from cpi_store
            cpi_df = None
            _picked_run_meta = None
            try:
                from core.analytics.cpi.cpi_store import list_cpi_runs, load_cpi_summary
                from core.analytics.cpi.cpi_engine import cpi_summary_stats
                runs = list_cpi_runs()
                if runs:
                    _run_id_to_load = _selected_run_id or runs[0]["run_id"]
                    _picked_run_meta = next(
                        (r for r in runs if r["run_id"] == _run_id_to_load), runs[0]
                    )
                    cpi_df = load_cpi_summary(_picked_run_meta["run_id"])
            except Exception:
                pass

            # Fall back to cpi_engine compute if no stored run
            if cpi_df is None or (hasattr(cpi_df, 'empty') and cpi_df.empty):
                try:
                    from core.analytics.cpi.cpi_engine import compute_cpi_plus, cpi_band
                    cpi_df = compute_cpi_plus(canonical_df)
                    _band_fn = cpi_band
                except ImportError:
                    st.error("CPI modules not found. Run CPI analysis first.")
                    return
                except Exception as e:
                    st.error(f"CPI computation error: {e}")
                    return
            else:
                try:
                    from core.analytics.cpi.cpi_engine import cpi_band as _band_fn
                except ImportError:
                    _band_fn = lambda x: "—"

            sections = [{"heading": "Competency Progression Index (CPI+) Report", "body": ""}]

            if cpi_df is not None and not cpi_df.empty:
                # Detect column names (cpi_store uses cpi_plus, compute_cpi_plus also uses cpi_plus)
                cpi_col = "cpi_plus" if "cpi_plus" in cpi_df.columns else cpi_df.columns[-1]
                mean_cpi = cpi_df[cpi_col].mean()
                std_cpi  = cpi_df[cpi_col].std()
                _scope_line = (
                    f"  |  Scope: {_picked_run_meta['module_id']} "
                    f"({_picked_run_meta['instrument_key']})"
                    if _picked_run_meta else ""
                )
                sections.append({
                    "heading": "Group Summary",
                    "body":    (
                        f"N = {len(cpi_df)}  |  "
                        f"Mean CPI+ = {mean_cpi:.3f}  |  SD = {std_cpi:.3f}{_scope_line}"
                        if not pd.isna(mean_cpi) else f"N = {len(cpi_df)}{_scope_line}"
                    ),
                })

            if "Per-student CPI+ table" in selected and cpi_df is not None and not cpi_df.empty:
                # Select available columns
                avail = [c for c in ["user_id","participant_id","cpi_quant","cpi_qual",
                                     "cpi_plus","module_id"] if c in cpi_df.columns]
                disp = cpi_df[avail].copy()
                if "cpi_plus" in disp.columns:
                    disp["Band"] = disp["cpi_plus"].apply(_band_fn)
                for c in [col for col in disp.columns
                          if col not in ("user_id","participant_id","module_id","Band")]:
                    disp[c] = disp[c].apply(
                        lambda x: f"{x:.3f}" if x is not None and not pd.isna(x) else "—"
                    )
                sections.append({
                    "heading": "Per-Student CPI+ Scores",
                    "table":   disp,
                    "caption": "CPI+ = w₁ × CPI_quant + w₂ × CPI_qual",
                })

            if "Component interpretation" in selected:
                sections.append({
                    "heading": "Score Band Interpretation",
                    "body":    (
                        "High (≥0.75): Strong performance across all components.\n"
                        "Moderate (0.50–0.74): Solid progress; targeted review of weaker components.\n"
                        "Developing (0.25–0.49): Early stage; structured support recommended.\n"
                        "Emerging (<0.25): Significant gaps; immediate intervention warranted."
                    ),
                })

            if "Methodology note" in selected:
                # Dynamic per selected run, since a run's module/instrument
                # scope can now be anything the teacher configured in the
                # CPI tab -- a hardcoded "single module, MCQ only" string
                # would be actively wrong for a multi-module/gain-score run
                # (fixed 2026-08-09, Task F).
                if _picked_run_meta:
                    _scope_desc = (
                        f"This report covers modules/instruments: "
                        f"**{_picked_run_meta['module_id']}** "
                        f"({_picked_run_meta['instrument_key']}).\n\n"
                    )
                    _quant_desc = (
                        "CPI_quant: "
                        + ("MCQ performance via CTT (proportion correct) or IRT "
                           "(Rasch/2PL sigmoid-normalised θ). References: "
                           "Crocker & Algina (1986); Baker (1985)."
                           if _picked_run_meta["instrument_key"].strip() in (
                               "content_mcq",
                           ) or "_content_mcq_assessment" in _picked_run_meta["instrument_key"]
                           else "a teacher-selected combination of Content MCQ "
                           "performance and/or normalised gain (Hake, 1998) on "
                           "AI Misconceptions and/or AI Conceptual Inventory "
                           "(AICI), blended per the instrument types shown above.")
                    )
                else:
                    _scope_desc = (
                        "No stored CPI run was selected — this report uses the "
                        "3-component fallback formula computed fresh over the "
                        "currently filtered data.\n\n"
                    )
                    _quant_desc = (
                        "CPI_outcome: a blend of Content MCQ performance and "
                        "normalised gain (Hake, 1998) on AI Misconceptions and "
                        "AI Conceptual Inventory (AICI), unconditionally over "
                        "every module."
                    )
                sections.append({
                    "heading": "Methodology",
                    "body":    (
                        _scope_desc
                        + _quant_desc
                        + "\n\nCPI_qual: LLM-as-judge scores on reflection quality "
                        "(depth of insight, conceptual grounding, personal connection, 1–4 scale). "
                        "Reference: Zheng et al. (2023).\n\n"
                        "CPI+ = w₁ × CPI_quant + w₂ × CPI_qual. "
                        "Default equal weights justified for small samples (Dawes, 1979)."
                    ),
                })

            pdf_bytes = _build_pdf(sections, "Competency Progression Index Report")
            st.download_button(
                "⬇️ Download CPI+ Report (PDF)",
                data=pdf_bytes,
                file_name="b4ai_cpi_report.pdf",
                mime="application/pdf",
                key="rpt_cpi_dl",
            )
            st.success("PDF ready.")


# -----------------------------------------------------------------------
# Public entry point — called by app.py
# -----------------------------------------------------------------------

def show_teacher_dashboard(username: str) -> None:
    """
    Render the full teacher analytics dashboard.
    Called by app.py after role == 'teacher' is verified.
    """
    st.title("📊 Teacher Analytics Dashboard")
    st.markdown(f"Welcome **{username}**")

    # ── Data freshness indicator + manual refresh ──────────────────────────
    # _load_data() below is cached for 5 minutes. Right after a student
    # finishes a module, the dashboard can transiently show data that
    # doesn't yet include that submission until the cache expires — with
    # no visible indication, that's indistinguishable from a real bug.
    # Show when the data was last loaded and give a one-click way to force
    # a fresh reload instead of waiting.
    _refresh_col, _freshness_col = st.columns([1, 5])
    with _refresh_col:
        if st.button("🔄 Refresh Data", key="teacher_refresh_data_btn"):
            _load_data.clear()
            st.rerun()
    with _freshness_col:
        _loaded_at = st.session_state.get("_teacher_data_loaded_at")
        if _loaded_at:
            st.caption(
                f"📅 Data as of **{_loaded_at} UTC** — auto-refreshes every 5 min, "
                "or click Refresh Data for the latest submissions right now."
            )

    # ── Minimal safe CSS — no pseudo-selectors that block interactions ────────
    st.markdown("""<style>
    /* Left-align all dataframe cells */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th { text-align: left !important; }
    </style>""", unsafe_allow_html=True)

    # Load data (cached)
    with st.spinner("Loading research dataset…"):
        try:
            canonical_df, demographics_df, cohort_map = _load_data()
            # Explicitly UTC (not the container's local clock) so the
            # caption below is never ambiguous about which timezone it's in.
            st.session_state["_teacher_data_loaded_at"] = datetime.utcnow().strftime("%H:%M:%S")
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    # Sidebar filters — returns filtered views
    filtered_canonical, filtered_demographics = _render_sidebar(
        canonical_df, demographics_df, cohort_map
    )

    # Empty-data guard after filtering
    if filtered_canonical.empty or filtered_demographics.empty:
        st.warning("No data matches the current filter combination. Adjust filters in the sidebar.")
        return

    # ---- Tabs ----
    # ---- Top-level navigation via session-state radio ----
    # st.tabs resets to index 0 on every widget interaction inside any tab.
    # A session-state-backed st.radio persists the selection across reruns.
    _TAB_OPTIONS = [
        "📊 Basic Statistics",
        "📈 Inferential Statistics",
        "🤖 LLM Analysis",
        "📉 Competency Progression",
        "🔬 IRT Analysis",
        "📄 Report Generation",
    ]
    if "teacher_dash_tab" not in st.session_state:
        st.session_state["teacher_dash_tab"] = _TAB_OPTIONS[0]
    active_tab = st.radio(
        "**Select section:**",
        options=_TAB_OPTIONS,
        horizontal=True,
        key="teacher_dash_tab",
    )
    # Active section banner — always visible regardless of CSS support
    _TAB_COLORS = {
        "📊 Basic Statistics":       ("#0077BB", "#E6F3FB"),
        "📈 Inferential Statistics": ("#EE7733", "#FFF3E6"),
        "🤖 LLM Analysis":           ("#CC79A7", "#F9EEF5"),
        "📉 Competency Progression": ("#534AB7", "#EEEDFE"),
        "🔬 IRT Analysis":           ("#009E73", "#E6F7F1"),
        "📄 Report Generation":      ("#888888", "#F0F0F0"),
    }
    _tc, _tbg = _TAB_COLORS.get(active_tab, ("#333", "#F8F8F8"))
    st.markdown(
        f"<div style='background:{_tbg};border-left:5px solid {_tc};"
        f"border-radius:6px;padding:0.4rem 1rem;margin-bottom:0.5rem;'>"
        f"<strong style='color:{_tc};'>▶ {active_tab}</strong></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    if active_tab == "📊 Basic Statistics":
        _render_participant_summary(filtered_demographics, cohort_map)
        st.divider()
        _render_assessment_scores(filtered_canonical)
        st.divider()
        _render_survey_construct_means(filtered_canonical)

    elif active_tab == "📈 Inferential Statistics":
        _render_inferential_tab(filtered_canonical, filtered_demographics)

    elif active_tab == "🔬 IRT Analysis":
        _render_irt_tab(filtered_canonical)

    elif active_tab == "🤖 LLM Analysis":
        _render_llm_tab(username, filtered_canonical)

    elif active_tab == "📉 Competency Progression":
        _render_cpi_tab(username, filtered_canonical)

    elif active_tab == "📄 Report Generation":
        _render_report_tab(username, filtered_canonical, filtered_demographics, cohort_map)

# -----------------------------------------------------------------------
# Tab 2 — IRT Analysis
# -----------------------------------------------------------------------

def _render_irt_tab(canonical_df: pd.DataFrame) -> None:
    """Tab 2 — IRT Analysis using mirt via rpy2."""

    st.subheader("🔬 IRT Analysis")
    st.info("🚧 This section is functional but earmarked for further development.")

    if not _IRT_AVAILABLE:
        st.error(
            "The girth IRT library is not installed. "
            "Run: pip install girth"
        )
        return

    st.caption(
        "Item Response Theory analysis powered by R's **mirt** package. "
        "Rasch (1PL) runs at any n with a low-n warning. "
        f"2PL and GRM require n ≥ {MIN_N_2PL}."
    )

    with st.expander("ℹ️ IRT — Plain-language guide for teachers", expanded=False):
        st.markdown(_IRT_HELP["rasch"])
        st.divider()
        st.markdown(_IRT_HELP["2pl"])
        st.divider()
        st.markdown(_IRT_HELP["grm"])
        st.divider()
        st.markdown(_IRT_HELP["theta"])

    # ---- Instrument type selector ----
    if "irt_type_radio" not in st.session_state:
        st.session_state["irt_type_radio"] = "Binary Assessment"
    irt_type = st.radio(
        "**Instrument type:**",
        options=["Binary Assessment", "Likert Survey"],
        horizontal=True,
        key="irt_type_radio",
    )
    _IRT_COLORS = {
        "Binary Assessment": ("#0077BB", "#E6F3FB"),
        "Likert Survey":     ("#009E73", "#E6F7F1"),
    }
    _ic, _ibg = _IRT_COLORS[irt_type]
    st.markdown(
        f"<div style='background:{_ibg};border-left:5px solid {_ic};"
        f"border-radius:6px;padding:0.4rem 1rem;margin:0.3rem 0;'>"
        f"<strong style='color:{_ic};'>▶ {irt_type}</strong> — "
        f"{'Rasch (1PL) or 2PL calibration' if irt_type == 'Binary Assessment' else 'Graded Response Model (GRM)'}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ==============================================================
    # Binary Assessment IRT
    # ==============================================================
    if irt_type == "Binary Assessment":
        st.markdown("### Binary Assessment — Item Response Theory")

        col1, col2 = st.columns(2)
        with col1:
            instrument_mode = st.radio(
                "Instrument selection",
                options=["Per instrument", "Pre + Post combined"],
                horizontal=True,
                key="irt_bin_mode",
            )
        with col2:
            model_choice = st.selectbox(
                "IRT Model",
                options=["Rasch (1PL)", "2PL (requires n≥50)"],
                key="irt_bin_model",
            )

        # Instrument picker
        binary_instruments = {
            k: v for k, v in _ASSESSMENT_LABELS.items()
            if "module" not in k
        }

        if instrument_mode == "Per instrument":
            selected_label = st.selectbox(
                "Instrument",
                options=list(binary_instruments.values()),
                key="irt_bin_instrument",
            )
            selected_key = next(
                k for k, v in binary_instruments.items()
                if v == selected_label
            )
            try:
                matrix, item_ids = build_binary_response_matrix(
                    canonical_df, selected_key
                )
            except ValueError as e:
                st.warning(str(e)); return

        else:
            # Pre + Post combined
            pair_choice = st.selectbox(
                "Assessment pair",
                options=[
                    "AI Misconceptions (Pre + Post)",
                    "AICI (Pre + Post)",
                ],
                key="irt_bin_pair",
            )
            pair_keys = {
                "AI Misconceptions (Pre + Post)": [
                    "precourse_pre_ai_misconceptions_assessment",
                    "postcourse_post_ai_misconceptions_assessment",
                ],
                "AICI (Pre + Post)": [
                    "precourse_pre_aici_assessment",
                    "postcourse_post_aici_assessment",
                ],
            }
            keys = pair_keys[pair_choice]
            try:
                parts = []
                for k in keys:
                    m, ids = build_binary_response_matrix(canonical_df, k)
                    parts.append(m)
                # Concatenate rows, keeping item columns consistent
                matrix = pd.concat(parts)
                item_ids = list(parts[0].columns)
            except ValueError as e:
                st.warning(str(e)); return

        n = len(matrix)
        st.caption(f"Matrix: {n} persons × {len(item_ids)} items")

        if n < 3:
            st.warning("Insufficient data to fit IRT model (n < 3)."); return

        if st.button("▶ Run IRT Analysis", key="irt_run_binary"):
            with st.spinner("Fitting model in R via mirt…"):
                if "2PL" in model_choice:
                    result = run_2pl_model(matrix, item_ids)
                else:
                    result = run_rasch_model(matrix, item_ids)

            _render_irt_result(result, item_ids, matrix)

    # ==============================================================
    # Likert Survey IRT (GRM)
    # ==============================================================
    else:
        st.markdown("### Likert Survey — Graded Response Model (GRM)")

        col1, col2 = st.columns(2)
        with col1:
            survey_choice = st.selectbox(
                "Survey",
                options=["SCCCES", "SIMS"],
                key="irt_survey_choice",
            )
        survey_key = (
            "b4ai_sccces_survey" if survey_choice == "SCCCES"
            else "b4ai_sims_survey"
        )
        sccces_constructs = [
            "engagement_with_task","effort_and_persistence",
            "experience_of_flow","coherency_of_messaging",
            "plausibility_of_messaging","credibility_of_messaging",
            "comprehensibility_of_messaging","attention",
            "culture","personal_relevance",
        ]
        sims_constructs = [
            "intrinsic_motivation","identified_regulation",
            "external_regulation","amotivation",
        ]
        constructs = (
            sccces_constructs if survey_choice == "SCCCES"
            else sims_constructs
        )
        with col2:
            construct_choice = st.selectbox(
                "Construct",
                options=constructs,
                key="irt_construct_choice",
            )

        try:
            lmatrix, litem_ids = build_likert_response_matrix(
                canonical_df, survey_key, construct_choice
            )
        except ValueError as e:
            st.warning(str(e)); return

        n = len(lmatrix)
        st.caption(f"Matrix: {n} persons × {len(litem_ids)} items")

        if st.button("▶ Run GRM Analysis", key="irt_run_grm"):
            with st.spinner("Fitting GRM in R via mirt…"):
                result = run_grm_model(lmatrix, litem_ids)
            _render_irt_result(result, litem_ids, lmatrix)

        # ── Multi-construct CR summary ─────────────────────────────────────
        st.divider()
        with st.expander(
            f"📋 All-construct CR summary — {survey_choice}",
            expanded=False,
        ):
            st.caption(
                "Compute Construct Reliability (CR), AVE, and Cronbach α for "
                f"every {survey_choice} construct in one table. "
                "Requires Rasch model (fast). "
                "Reference: Rosli et al. (2021)."
            )
            if st.button(f"▶ Compute CR for all {survey_choice} constructs",
                         key="irt_cr_all"):
                try:
                    from core.analytics.irt.reliability_analysis import (
                        compute_reliability_report, build_reliability_summary_df
                    )
                    _all_constructs = (
                        sccces_constructs if survey_choice == "SCCCES"
                        else sims_constructs
                    )
                    _all_reports = []
                    _cr_progress = st.progress(0)
                    for _ci, _con in enumerate(_all_constructs):
                        _cr_progress.progress((_ci + 1) / len(_all_constructs))
                        try:
                            _lmat, _lids = build_likert_response_matrix(
                                canonical_df, survey_key, _con
                            )
                            if len(_lmat) < 3:
                                continue
                            _rr = run_grm_model(_lmat, _lids)
                            _rep = compute_reliability_report(
                                irt_result=_rr,
                                response_matrix=_lmat,
                                construct_name=_con.replace("_", " ").title(),
                                model_type="GRM",
                            )
                        except Exception as _e:
                            _rep = {
                                "construct": _con.replace("_"," ").title(),
                                "n_items": 0, "n_persons": 0, "model_type": "GRM",
                                "cr": float("nan"), "cr_badge": "—",
                                "ave": float("nan"), "ave_badge": "—",
                                "omega": float("nan"),
                                "cronbach_alpha": float("nan"), "alpha_badge": "—",
                                "loadings": [], "error": str(_e),
                            }
                        _all_reports.append(_rep)
                    _cr_progress.empty()
                    if _all_reports:
                        _summ_df = build_reliability_summary_df(_all_reports)
                        st.dataframe(
                            _summ_df, hide_index=True, width="stretch",
                        )
                        st.caption(
                            "CR ≥ 0.70 acceptable, ≥ 0.80 good. "
                            "AVE ≥ 0.50 = adequate convergent validity. "
                            "α shown as legacy baseline only. "
                            "Rosli et al. (2021); Libasin et al. (2025)."
                        )
                except ImportError:
                    st.error("Deploy core/analytics/irt/reliability_analysis.py first.")


def _render_irt_result(
    result: dict,
    item_ids: List[str],
    matrix: pd.DataFrame,
) -> None:
    """Render IRT result cards, tables and plots."""

    if result.get("error"):
        if f"n ≥ {MIN_N_2PL}" in str(result["error"]) or            "n ≥ 50" in str(result["error"]):
            st.info(result["error"])
        else:
            st.error(f"IRT model error: {result['error']}")
        return

    n = result["n_persons"]
    model = result["model_type"]

    # Low-n warning
    if result.get("low_n_warning"):
        st.warning(
            f"⚠️ **n = {n}** — IRT parameter estimates at this sample size "
            f"are unstable. Results are shown for pipeline verification. "
            f"Reliable IRT analysis requires n ≥ {MIN_N_WARN}. "
            f"Re-run after the July/August cohort (expected n ≈ 90)."
        )

    # Model fit row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",     model,
              help="1PL=Rasch (difficulty only), 2PL (difficulty+discrimination), GRM (Likert items)")
    c2.metric("N persons", n,
              help="Number of students included in the analysis")
    c3.metric("N items",   result["n_items"],
              help="Number of assessment items calibrated by the model")
    aic = result.get("aic")
    c4.metric("AIC", f"{aic:.1f}" if aic else "—",
              help="Lower AIC = better model fit. Useful for comparing Rasch vs 2PL on the same data.")

    dropped = result.get("dropped_items", [])
    if dropped:
        st.caption(
            f"⚠️ Items with zero variance dropped from calibration: "
            f"{', '.join(dropped)}"
        )

    st.divider()

    # Item parameters table
    params = result.get("item_params", pd.DataFrame())
    if not params.empty:
        st.markdown("**Item Parameters**")
        display_params = params.copy()
        numeric_cols = display_params.select_dtypes("number").columns
        for col in numeric_cols:
            display_params[col] = display_params[col].round(3)
        st.dataframe(display_params, hide_index=True, width="stretch")

    # Item fit table (Rasch only)
    item_fit = result.get("item_fit", pd.DataFrame())
    if not item_fit.empty and len(item_fit.columns) > 1:
        st.markdown("**Item Fit Statistics**")
        fit_display = item_fit.copy()
        numeric_cols = fit_display.select_dtypes("number").columns
        for col in numeric_cols:
            fit_display[col] = fit_display[col].round(3)
        st.dataframe(fit_display, hide_index=True, width="stretch")

    # Person abilities
    persons = result.get("person_params", pd.DataFrame())
    if not persons.empty and "theta" in persons.columns:
        st.divider()
        st.markdown("**Person Ability Estimates (θ)**")
        p_display = persons.copy()
        for col in p_display.select_dtypes("number").columns:
            p_display[col] = p_display[col].round(3)
        st.dataframe(p_display, hide_index=True, width="stretch")

        if _HAS_PLOTLY:
            import plotly.express as px
            fig = px.histogram(
                x=persons["theta"].tolist(),
                nbins=max(5, n // 2),
                title="Person Ability Distribution (θ)",
                labels={"x": "Ability (θ)"},
                color_discrete_sequence=["#6C63FF"],
            )
            fig.update_layout(height=280, margin=dict(t=40,b=10,l=10,r=10))
            fig.update_xaxes(title="Ability (θ)")
            fig.update_yaxes(title="Count")
            st.plotly_chart(fig, width='content')

    # ── Construct Reliability Panel ──────────────────────────────────────────
    st.divider()
    with st.expander(
        "📐 Reliability Analysis — CR, AVE, Cronbach α (Rosli et al., 2021)",
        expanded=False,
    ):
        try:
            from core.analytics.irt.reliability_analysis import (
                compute_reliability_report,
                build_reliability_summary_df,
            )
            _rel = compute_reliability_report(
                irt_result=result,
                response_matrix=matrix,
                construct_name=result.get("instrument_key", ""),
                model_type=result.get("model_type", "Rasch"),
            )
            if _rel.get("error"):
                st.warning(f"Reliability computation: {_rel['error']}")
            else:
                # Summary tiles
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric(
                    "Construct Reliability (CR)",
                    f"{_rel['cr']:.3f}" if not (isinstance(_rel['cr'], float) and __import__('math').isnan(_rel['cr'])) else "—",
                    help="CR ≥ 0.70 acceptable; ≥ 0.80 good (Rosli et al., 2021)",
                )
                rc2.metric(
                    "AVE",
                    f"{_rel['ave']:.3f}" if not (isinstance(_rel['ave'], float) and __import__('math').isnan(_rel['ave'])) else "—",
                    help="Average Variance Extracted ≥ 0.50 = adequate convergent validity (Fornell & Larcker, 1981)",
                )
                rc3.metric(
                    "Cronbach α",
                    f"{_rel['cronbach_alpha']:.3f}" if not (isinstance(_rel['cronbach_alpha'], float) and __import__('math').isnan(_rel['cronbach_alpha'])) else "—",
                    help="Legacy baseline. α ≥ 0.70 acceptable (Hair et al., 2014)",
                )
                rc4.metric(
                    "McDonald ω",
                    f"{_rel['omega']:.3f}" if not (isinstance(_rel['omega'], float) and __import__('math').isnan(_rel['omega'])) else "—",
                    help="ω ≥ 0.70 acceptable; preferred over α for non-tau-equivalent items (Dunn et al., 2014)",
                )

                # Status banners
                st.markdown(
                    f"**CR status:** {_rel['cr_badge']}  |  "
                    f"**AVE status:** {_rel['ave_badge']}  |  "
                    f"**α status:** {_rel['alpha_badge']}"
                )

                # Factor loadings table
                if _rel["loadings"]:
                    _params = result.get("item_params", pd.DataFrame())
                    _load_df = pd.DataFrame({
                        "Item":    _params["item_id"].tolist() if "item_id" in _params.columns else [f"Item {i+1}" for i in range(len(_rel["loadings"]))],
                        "λ (loading)": [f"{l:.4f}" for l in _rel["loadings"]],
                        "Error var (1−λ²)": [f"{1-l**2:.4f}" for l in _rel["loadings"]],
                    })
                    if "a" in _params.columns:
                        _load_df.insert(1, "a (discrimination)", _params["a"].round(3).tolist())
                    st.markdown("**Item factor loadings (λ = a/√(1+a²))**")
                    st.dataframe(_load_df, hide_index=True, width="stretch")

                # Interpretation guide
                st.divider()
                st.markdown(
                    "**Interpretation (Rosli et al., 2021; Libasin et al., 2025)**\n\n"
                    "| Metric | Formula | Threshold | What it measures |\n"
                    "|---|---|---|---|\n"
                    "| **CR** | (Σλᵢ)² / [(Σλᵢ)² + Σ(1−λᵢ²)] | ≥ 0.70 acceptable, ≥ 0.80 good | Internal consistency accounting for actual item loadings |\n"
                    "| **AVE** | Σλᵢ² / n | ≥ 0.50 | Convergent validity — proportion of variance captured by construct |\n"
                    "| **ω** | Same as CR for unidimensional | ≥ 0.70 | More accurate than α when items have unequal loadings |\n"
                    "| **α** | n/(n−1) × (1 − Σs²ᵢ/s²_T) | ≥ 0.70 | Legacy baseline; assumes tau-equivalence |\n\n"
                    "Factor loadings derived from IRT discrimination parameters via λ = a/√(1+a²) "
                    "(UIRT–FA equivalence, McDonald 1999). "
                    "For Rasch (1PL) all a = 1.0 → λ = 0.707 by model constraint."
                )
                st.caption(
                    "**Primary citation:** Rosli, M. S., Saleh, N. S., Alshammari, S. H., Ibrahim, M. M., "
                    "Atan, A. S., & Atan, N. A. (2021). Improving Questionnaire Reliability using Construct "
                    "Reliability for Researches in Educational Technology. *iJIM, 15*(04), 109. "
                    "https://doi.org/10.3991/ijim.v15i04.20199 | "
                    "**Also:** Libasin, Z., Ahmad, N., & Umar, N. (2025). Beyond Cronbach's Alpha. "
                    "SIG e-Learning@CS. e-ISBN 978-629-98755-7-4."
                )
        except ImportError:
            st.info(
                "Deploy `core/analytics/irt/reliability_analysis.py` "
                "to enable this panel."
            )

    
        # Wright Map
    st.divider()
    with st.expander("ℹ️ How to read the Wright Map", expanded=False):
        st.markdown(_IRT_HELP["wright_map"])
    st.markdown("**Wright Map (Person-Item)**")
    wm = get_wright_map_data(result)
    if not wm["persons"].empty and not wm["items"].empty and _HAS_PLOTLY:
        import plotly.graph_objects as go
        theta_vals = wm["persons"]["theta"].tolist()
        b_vals     = wm["items"]["b"].tolist()
        item_names = wm["items"]["item_id"].tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theta_vals,
            y=["Persons"] * len(theta_vals),
            mode="markers",
            marker=dict(symbol="circle", size=10, color="#4C9BE8"),
            name="Person θ",
        ))
        fig.add_trace(go.Scatter(
            x=b_vals,
            y=["Items"] * len(b_vals),
            mode="markers+text",
            text=item_names,
            textposition="top center",
            marker=dict(symbol="diamond", size=10, color="#F4845F"),
            name="Item difficulty (b)",
        ))
        fig.update_layout(
            title="Wright Map",
            xaxis_title="Logit scale",
            height=320,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, width='content')
    elif wm["persons"].empty:
        st.caption("Wright Map unavailable — item parameters could not be extracted.")

    # ICC section
    st.divider()
    with st.expander("ℹ️ How to read an ICC", expanded=False):
        st.markdown(_IRT_HELP["icc"])
    st.markdown("**Item Characteristic Curves (ICC)**")
    icc_item = st.selectbox(
        "Select item for ICC",
        options=item_ids,
        key="irt_icc_item",
    )
    icc_df = get_icc_data(result, item_id=icc_item)
    if not icc_df.empty and _HAS_PLOTLY:
        import plotly.express as px
        fig = px.line(
            icc_df,
            x="theta", y="probability",
            color="category",
            title=f"ICC — {icc_item}",
            labels={"theta": "Ability (θ)", "probability": "P(response)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, width='content')
    elif icc_df.empty:
        st.caption(
            "ICC data unavailable. This may occur with mirt versions "
            "that require additional extraction steps."
        )



# =======================================================================
# LLM ANALYSIS TAB — ITA + DTA
# =======================================================================
# this helper at the top of the LLM section fixes path lookup in docker by replacing both next(...) path searches in this function with a direct env var lookup. 
def _get_responses_db_path(): #cgpt replaced for syntax errors
    """Return responses.db path using env var first, then filesystem search."""
    import os
    from pathlib import Path as _P

    env_path = os.getenv("SQLITE_PATH")
    if env_path:
        p = _P(env_path)
        if p.exists():
            return p

    # fallback: filesystem search (works locally)
    return next(
        (
            p / "responses.db"
            for p in _P(__file__).resolve().parents
            if (p / "responses.db").exists()
        ),
        None,
    )
# -----------------------------------------------------------------------
# Multi-source loader (shared by ITA and DTA)
# -----------------------------------------------------------------------

def _count_available_sources(cohort_ids: list = None) -> dict:
    """
    Return available counts without loading full text content.
    Returns {"reflections": int, "interviews": int, "observer": int}.

    cohort_ids: optional list — when non-empty, counts are scoped to only
    those cohorts (reflections via the registered-user cohort_map,
    interviews/observer via their own cohort_id tag), matching how
    _load_combined_transcripts() filters the actual run data. Without
    this, the displayed "available" counts would always show the
    unfiltered total even when a cohort filter is active in Step 1.
    """
    import sqlite3 as _sq
    counts = {"reflections": 0, "interviews": 0, "observer": 0}
    db = _get_responses_db_path()
    if db:
        try:
            with _sq.connect(db) as _c:
                rows = _c.execute(
                    "SELECT DISTINCT user_id FROM responses "
                    "WHERE instrument_name LIKE '%module_reflections%' "
                    "AND response_value IS NOT NULL"
                ).fetchall()
            if cohort_ids:
                from auth.user_manager import get_user_cohort_map
                _cmap = get_user_cohort_map()
                counts["reflections"] = sum(
                    1 for (uid,) in rows if _cmap.get(uid) in cohort_ids
                )
            else:
                counts["reflections"] = len(rows)
        except Exception:
            pass
    # Interviews — count persistent transcripts (cohort-scoped when a filter is active)
    try:
        if cohort_ids:
            counts["interviews"] = len(load_for_analysis(
                source="persistent", source_type="interview", cohort_ids=cohort_ids
            ) or [])
        else:
            counts["interviews"] = get_transcript_count("interview") or 0
    except Exception:
        pass
    # Observer/instructor transcripts — count persistent transcripts (cohort-scoped when a filter is active)
    try:
        if cohort_ids:
            counts["observer"] = len(load_for_analysis(
                source="persistent", source_type="observer", cohort_ids=cohort_ids
            ) or [])
        else:
            counts["observer"] = get_transcript_count("observer") or 0
    except Exception:
        pass
    return counts


def _load_combined_transcripts(
    sources: list,
    per_run_files=None,
    cohort_ids: list = None,
) -> list:
    """
    Load and merge transcripts from one or more sources.
    sources: list containing any of "responses", "persistent", "observer", "per_run"
    When a participant appears in multiple sources their texts are concatenated.

    cohort_ids: optional list — when non-empty, scopes this run to only
    those cohorts. Reflections are filtered via the registered-user
    cohort_map (auth.user_manager.get_user_cohort_map()); interview/
    observer transcripts are filtered via their own cohort_id tag, which
    works even for pre-pilot participants with no user record at all.
    """
    import sqlite3 as _sq, re as _re
    from pathlib import Path as _Pth
    from collections import defaultdict

    combined = defaultdict(lambda: {"content": [], "source_types": []})

    for source_key in sources:
        if source_key == "responses":
            #db = next(
            #    (p / "responses.db" for p in _Pth(__file__).resolve().parents #replace both next(...) path searches in this function with a direct env var lookup.
            #if (p / "responses.db").exists()), None
            #)
            db = _get_responses_db_path()
            if not db:
                continue
            conn = _sq.connect(db)
            rows = conn.execute(
                "SELECT user_id, response_value FROM responses "
                "WHERE instrument_name LIKE '%module_reflections%' "
                "AND response_value IS NOT NULL"
            ).fetchall()
            conn.close()
            _reflection_cohort_map = {}
            if cohort_ids:
                try:
                    from auth.user_manager import get_user_cohort_map
                    _reflection_cohort_map = get_user_cohort_map()
                except Exception:
                    _reflection_cohort_map = {}
            for uid, rval in rows:
                if cohort_ids and _reflection_cohort_map.get(uid) not in cohort_ids:
                    continue
                if rval and str(rval).strip():
                    combined[uid]["content"].append(str(rval))
                    if "reflections" not in combined[uid]["source_types"]:
                        combined[uid]["source_types"].append("reflections")

        elif source_key == "persistent":
            try:
                trans = load_for_analysis(source="persistent", source_type="interview",
                                           cohort_ids=cohort_ids or None)
                for t in (trans or []):
                    pid  = t.get("participant_id", "unknown")
                    text = str(t.get("content", "")).strip()
                    if text:
                        combined[pid]["content"].append(text)
                        if "interview" not in combined[pid]["source_types"]:
                            combined[pid]["source_types"].append("interview")
            except Exception:
                pass

        elif source_key == "observer":
            try:
                trans = load_for_analysis(source="persistent", source_type="observer",
                                           cohort_ids=cohort_ids or None)
                for t in (trans or []):
                    pid  = t.get("participant_id", "unknown")
                    text = str(t.get("content", "")).strip()
                    if text:
                        combined[pid]["content"].append(text)
                        if "observer" not in combined[pid]["source_types"]:
                            combined[pid]["source_types"].append("observer")
            except Exception:
                pass

        elif source_key == "per_run" and per_run_files:
            for f_obj in per_run_files:
                raw = f_obj.read()
                try:    text = raw.decode("utf-8", errors="replace")
                except: text = raw.decode("latin-1", errors="replace")
                pid = _re.sub(r"\.(vtt|txt|pdf)$", "", f_obj.name, flags=_re.I).strip()
                if pid and text.strip():
                    combined[pid]["content"].append(text.strip())
                    if "per_run" not in combined[pid]["source_types"]:
                        combined[pid]["source_types"].append("per_run")

    return [
        {
            "participant_id": pid,
            "content":        " ".join(data["content"]),
            "source_types":   data["source_types"],
            "source_type":    "+".join(data["source_types"]),
        }
        for pid, data in combined.items()
        if "".join(data["content"]).strip()
    ]


def _cohort_filter_multiselect(prefix: str) -> list:
    """
    Cohort filter for LLM Analysis data sources. Empty selection = no
    filter (all cohorts included). Uses the full cohort registry
    (core.admin.user_service.get_all_cohorts()) rather than only cohorts
    with registered users, since pre-pilot interview/observer transcripts
    can be tagged with a cohort that has zero registered participants.
    """
    from core.admin import user_service as _us
    try:
        all_cohorts = _us.get_all_cohorts()
    except Exception:
        all_cohorts = []
    if not all_cohorts:
        return []
    _selected = st.multiselect(
        "Filter by cohort (optional — leave empty to include all cohorts)",
        options=all_cohorts,
        default=[],
        key=f"{prefix}_cohort_filter_widget",
        help=(
            "Scopes this analysis run to only the selected cohort(s). "
            "Applies to reflections (via registered students' cohort) and "
            "to interview/observer transcripts (via their own cohort tag)."
        ),
    )
    st.caption(
        "ℹ️ This filter is independent of the **Cohort** filter in the sidebar — "
        "the sidebar filter doesn't affect LLM Analysis, since it can't reach "
        "interview/observer transcripts that have no registered-user record. "
        "Use this control to scope this analysis by cohort."
    )
    return _selected


def _source_checkboxes(prefix: str, default_reflections: bool = True):
    """Render 3-way source checkboxes. Returns (sources_list, None)."""
    st.markdown("**Data sources** (select one or more):")
    c1, c2, c3 = st.columns(3)
    use_ref = c1.checkbox("Module reflections (DB)", value=default_reflections,
                          key=f"{prefix}_src_ref",
                          help="End-of-module reflection notes from responses.db")
    use_int = c2.checkbox("Interview transcripts (store)", value=False,
                          key=f"{prefix}_src_int",
                          help="Semi-structured transcripts uploaded via Admin dashboard")
    use_obs = c3.checkbox("Observer/instructor transcript(s) (store)", value=False,
                          key=f"{prefix}_src_obs",
                          help="Observer/instructor session notes uploaded via Admin dashboard")
    sources = []
    if use_ref: sources.append("responses")
    if use_int: sources.append("persistent")
    if use_obs: sources.append("observer")
    return sources, None


# -----------------------------------------------------------------------
# Top-level LLM tab
# -----------------------------------------------------------------------

def _render_llm_tab(username: str, canonical_df: pd.DataFrame) -> None:
    """Tab 3 — LLM Analysis: ITA + DTA."""
    st.subheader("🤖 LLM Analysis")

    if not _LLM_AVAILABLE:
        st.error(
            f"LLM modules unavailable: {_LLM_ERR}. "
            "Ensure anthropic, google-genai, openai, sentence-transformers "
            "and dta_pipeline are installed/deployed."
        )
        return

    # Seed LLM section + mode defaults so they are visible on first render
    if "llm_section_radio" not in st.session_state:
        st.session_state["llm_section_radio"] = "📖 Inductive Thematic Analysis (ITA)"
    if "llm_mode_radio" not in st.session_state:
        st.session_state["llm_mode_radio"] = "🧭 Guided"

    section = st.radio(
        "**Analysis type:**",
        options=["📖 Inductive Thematic Analysis (ITA)",
                 "🔍 Deductive Thematic Analysis (DTA)"],
        horizontal=True,
        key="llm_section_radio",
    )
    # Explicit active-section banner
    if section == "📖 Inductive Thematic Analysis (ITA)":
        st.markdown(
            "<div style='background:#E6F3FB;border-left:5px solid #0077BB;"
            "border-radius:6px;padding:0.5rem 1rem;margin:0.3rem 0;'>"
            "📖 <strong style='color:#0077BB;'>Inductive Thematic Analysis (ITA)</strong>"
            " — Braun &amp; Clarke (2006) via De Paoli (2024)</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        _render_ita_guided(username, canonical_df)
    else:
        st.markdown(
            "<div style='background:#FFF3E6;border-left:5px solid #EE7733;"
            "border-radius:6px;padding:0.5rem 1rem;margin:0.3rem 0;'>"
            "🔍 <strong style='color:#EE7733;'>Deductive Thematic Analysis (DTA)</strong>"
            " — Basics4AI codebook (CCCES, SCES, SIMS, AI-CI, AIM-F)</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        _render_dta_section(username, canonical_df)


# -----------------------------------------------------------------------
# ITA / DTA cost estimation helpers
# -----------------------------------------------------------------------

# Per-model pricing (USD / 1K tokens): input, output
_LLM_COST_PER_1K: dict = {
    "groq":   (0.0,      0.0),       # free tier
    "claude": (0.0008,   0.004),     # Haiku 3.5
    "gemini": (0.000075, 0.0003),    # Flash
    "gpt":    (0.00015,  0.0006),    # GPT-4o-mini
}


def _ita_cost_estimate(n_texts: int, models: list) -> list:
    """Rough ITA cost: n_texts*2 chunks (Phase 2) + 4 calls (Phases 3-6)."""
    lines = []
    total_calls = max(n_texts * 2, 1) + 4
    avg_in, avg_out = 700, 350
    for m in models:
        in_r, out_r = _LLM_COST_PER_1K.get(m, (0.0, 0.0))
        if in_r == 0.0 and out_r == 0.0:
            lines.append(f"**{m.title()}**: Free ✅")
        else:
            cost = total_calls * ((avg_in / 1000) * in_r + (avg_out / 1000) * out_r)
            lines.append(f"**{m.title()}**: ~${cost:.4f}")
    return lines


def _dta_cost_estimate(n_texts: int, n_constructs: int, models: list) -> list:
    """Rough DTA cost: n_texts * n_constructs Phase-2 calls + 2 summary calls."""
    lines = []
    total_calls = max(n_texts * n_constructs, 1) + 2
    avg_in, avg_out = 800, 300
    for m in models:
        in_r, out_r = _LLM_COST_PER_1K.get(m, (0.0, 0.0))
        if in_r == 0.0 and out_r == 0.0:
            lines.append(f"**{m.title()}**: Free ✅")
        else:
            cost = total_calls * ((avg_in / 1000) * in_r + (avg_out / 1000) * out_r)
            lines.append(f"**{m.title()}**: ~${cost:.4f}")
    return lines


# -----------------------------------------------------------------------
# ITA — Unified mode  (Guided + Expert merged)
# -----------------------------------------------------------------------

def _render_ita_guided(username: str, canonical_df: pd.DataFrame) -> None:
    # ── Session-state defaults (set only on first render) ──────────────
    _avail = get_available_models(check_keys=True)

    # Build the default model selection: Groq if available, nothing otherwise.
    # A single list key is used instead of 4 individual checkbox keys because
    # Streamlit clears widget keys from session state when the widget is not
    # rendered (e.g. when navigating between steps), causing selections to reset.
    _default_models = ["groq"] if _avail.get("groq", False) else []

    _defaults = {
        # Models — single persistent list, not 4 individual checkbox keys
        "ita_g_models":      _default_models,
        # Analysis settings
        "ita_g_temp":        0.0,
        "ita_g_n_themes":    5,
        "ita_g_n_codes":     3,
        "ita_g_dedup":       0.85,
        "ita_g_max_texts":   4,
        # Editable prompts — live widget values (populated from defaults)
        "ita_g_sys_prompt":  _ITA_SYSTEM_PROMPT,
        "ita_g_p2_prompt":   _PHASE2_PROMPT,
        "ita_g_p3_prompt":   _PHASE3_PROMPT,
        # Saved copies — committed by the Save button
        "ita_g_sys_saved":   _ITA_SYSTEM_PROMPT,
        "ita_g_p2_saved":    _PHASE2_PROMPT,
        "ita_g_p3_saved":    _PHASE3_PROMPT,
    }
    for _k, _d in _defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _d

    # ── Reset-flag processing — MUST run before any widget is created ──
    if st.session_state.pop("_ita_reset_sys", False):
        st.session_state["ita_g_sys_prompt"] = _ITA_SYSTEM_PROMPT
        st.session_state["ita_g_sys_saved"]  = _ITA_SYSTEM_PROMPT
    if st.session_state.pop("_ita_reset_p2", False):
        st.session_state["ita_g_p2_prompt"]  = _PHASE2_PROMPT
        st.session_state["ita_g_p2_saved"]   = _PHASE2_PROMPT
    if st.session_state.pop("_ita_reset_p3", False):
        st.session_state["ita_g_p3_prompt"]  = _PHASE3_PROMPT
        st.session_state["ita_g_p3_saved"]   = _PHASE3_PROMPT
    # Navigation flag — set by pipeline on completion, applied here before
    # the radio widget is instantiated (same pattern as prompt reset above)
    if st.session_state.pop("_ita_goto_step6", False):
        st.session_state["ita_guided_step"] = "6 — Results"

    STEPS = [
        "1 — Data Source",
        "2 — Models & Temperature",
        "3 — Theme Settings",
        "4 — Review & Edit Prompt",
        "5 — Run Analysis",
        "6 — Results",
    ]
    if "ita_guided_step" not in st.session_state:
        st.session_state["ita_guided_step"] = STEPS[0]
    step = st.radio("**Step:**", STEPS, horizontal=True, key="ita_guided_step")
    st.markdown(
        f"<div style='background:#F9EEF5;border-left:5px solid #CC79A7;"
        f"border-radius:6px;padding:0.3rem 1rem;margin:0.2rem 0;'>"
        f"<strong style='color:#CC79A7;'>Active: {step}</strong></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1 — DATA SOURCE
    # ═══════════════════════════════════════════════════════════════════
    if step == STEPS[0]:
        st.markdown("### Step 1 — Select Data Source")
        st.caption(
            "Choose which data to include in the analysis. "
            "**Reflections** are short written responses students submit after each module. "
            "**Interviews** are longer semi-structured conversations uploaded by the teacher. "
            "**Observer/instructor transcripts** are session notes or recordings uploaded by the teacher."
        )
        sources, per_run_files = _source_checkboxes("ita_g")
        _ita_cohort_sel = _cohort_filter_multiselect("ita_g")

        # ── Persist source selections to stable (non-widget) keys ─────
        # Checkbox widget keys are wiped by Streamlit when the widget is
        # not rendered (i.e. when you navigate away from Step 1).
        # Writing the values to separate plain keys here keeps them alive
        # across step navigation so Step 5 can read them reliably.
        st.session_state["ita_src_responses"]  = "responses"  in sources
        st.session_state["ita_src_persistent"] = "persistent" in sources
        st.session_state["ita_src_observer"]   = "observer"   in sources
        st.session_state["ita_src_per_run"]    = "per_run"    in sources
        st.session_state["ita_cohort_filter"]  = _ita_cohort_sel
        if per_run_files:
            #st.session_state["ita_g_upload"] = per_run_files #the code tries to write to a session state key that's already bound to a widget. 
            # Replaced with, which use a different key for the stored value
            st.session_state["ita_g_upload_store"] = per_run_files
            
        if not sources:
            st.warning("Select at least one source above.")
        else:
            st.divider()
            # Cohort-scoped counts (bug fixed 2026-08-08: this block used to run
            # its own cohort-unaware raw SQL/get_transcript_count() calls, so it
            # kept showing the unfiltered total even with a cohort filter active
            # in the same step. Now reuses the same cohort-aware helper Step 5
            # already uses, so both places agree.)
            _step1_counts = _count_available_sources(cohort_ids=_ita_cohort_sel or None)
            _step1_suffix = f" ({'/'.join(_ita_cohort_sel)})" if _ita_cohort_sel else ""

            if "responses" in sources:
                st.metric(f"Students with reflection notes{_step1_suffix}",
                          _step1_counts["reflections"])

            if "persistent" in sources:
                st.metric(f"Interviews in store{_step1_suffix}",
                          _step1_counts["interviews"])

            if "observer" in sources:
                st.metric(f"Observer/instructor transcripts in store{_step1_suffix}",
                          _step1_counts["observer"])

            st.info(
                "✅ Sources confirmed. Proceed to Step 2 to select a model, "
                "then set how many texts to include in Step 5."
            )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2 — MODELS & TEMPERATURE
    # ═══════════════════════════════════════════════════════════════════
    elif step == STEPS[1]:
        st.markdown("### Step 2 — Models & Temperature")
        _avail2 = get_available_models(check_keys=True)

        st.markdown("**Select which AI model(s) to run:**")
        st.caption(
            "You can run multiple models on the same data and compare results — "
            "useful for checking consistency across providers. "
            "**Groq (Llama 3.3 70B) is always free** — no billing required. "
            "Paid models appear in the list only when their API key is present in .env."
        )

        # Build options list — only show models whose keys are available
        _model_options = {
            "groq":   "Llama 3.3 70B (Groq — free ✅)",
            "claude": "Claude (Anthropic)",
            "gemini": "Gemini (Google)",
            "gpt":    "GPT (OpenAI)",
        }
        _available_options = [k for k in _model_options if _avail2.get(k, False)]
        _option_labels     = [_model_options[k] for k in _available_options]

        # Filter saved selection to only available models
        _current = [m for m in st.session_state.get("ita_g_models", [])
                    if m in _available_options]

        _selected_labels = st.multiselect(
            "Models",
            options=_option_labels,
            default=[_model_options[m] for m in _current
                     if m in _available_options],
            key="ita_g_models_widget",
            label_visibility="collapsed",
            help="Select one or more. Groq is free; others require API keys in .env.",
        )
        # Map labels back to keys and persist in the stable list key
        _label_to_key = {v: k for k, v in _model_options.items()}
        st.session_state["ita_g_models"] = [
            _label_to_key[lbl] for lbl in _selected_labels
        ]

        if not _selected_labels:
            st.warning("Select at least one model to continue.")
        else:
            st.success(
                "Selected: " + ", ".join(_selected_labels)
            )

        st.divider()
        st.markdown("**Temperature**")
        st.caption(
            "Temperature controls how creative or predictable the AI's responses are. \n\n"
            "- **0.0** — fully deterministic; the AI gives the same answer every time. "
            "Best for reproducible research results.\n"
            "- **0.3** — a small amount of variation; slightly different wording each run "
            "but still focused. A good middle ground.\n"
            "- **1.0** — noticeably varied responses; useful for exploring ideas or "
            "checking whether themes are stable across runs.\n"
            "- **2.0** — highly unpredictable; rarely useful for structured analysis.\n\n"
            "**Recommended for thematic analysis: 0.0–0.3.**"
        )
        st.slider(
            "Temperature (T)", 0.0, 2.0,
            value=float(st.session_state.get("ita_g_temp", 0.0)),
            step=0.1, format="%.1f", key="ita_g_temp",
        )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3 — THEME SETTINGS
    # ═══════════════════════════════════════════════════════════════════
    elif step == STEPS[2]:
        st.markdown("### Step 3 — Theme Settings")

        st.markdown("**Number of themes (Phase 3 & 4)**")
        st.caption(
            "This tells the AI how many broad themes to look for across all the text. "
            "Think of a theme as a recurring idea that keeps coming up — for example, "
            "*'curiosity about how AI thinks'* or *'worry about AI replacing humans'*. \n\n"
            "- **3–5 themes** — good starting point for a small cohort (under 20 students). "
            "Easier to interpret and discuss.\n"
            "- **6–10 themes** — richer picture for larger datasets; may overlap more.\n"
            "- **10+ themes** — very granular; usually only useful when comparing "
            "across many modules or cohorts.\n\n"
            "**Phase 4 note:** Phase 4 re-runs the same grouping task at Temperature = 1.0 "
            "using the same target number of themes you set here. It is a *stability check* — "
            "if Phase 4 produces very similar themes to Phase 3, the themes are robust. "
            "De Paoli (2024) suggests 11 groups as a reference; this slider defaults to a "
            "number scaled from your code list size, but you can set it manually to 11 if "
            "you wish to follow De Paoli's specification exactly."
        )
        st.slider(
            "Number of themes (Phase 3 & 4)", 3, 15,
            value=int(st.session_state.get("ita_g_n_themes", 5)),
            step=1, key="ita_g_n_themes",
        )

        st.divider()
        st.markdown("**Deduplication threshold**")
        st.caption(
            "During Phase 2 the AI extracts many short labels called *codes* "
            "(e.g. *'confusion about training data'*, *'uncertainty about AI decisions'*). "
            "Different chunks often produce very similar codes. "
            "Deduplication automatically merges near-duplicates so you end up with a "
            "cleaner, non-redundant code list before themes are built. \n\n"
            "- **0.70** — aggressive merging; fewer codes, risk of losing nuance.\n"
            "- **0.85** (recommended) — balanced; removes clear duplicates, keeps "
            "meaningfully different codes.\n"
            "- **0.95** — conservative; keeps more codes but the list may still have "
            "a lot of redundancy."
        )
        st.slider(
            "Deduplication threshold", 0.70, 0.95,
            value=float(st.session_state.get("ita_g_dedup", 0.85)),
            step=0.05, key="ita_g_dedup",
        )

        st.divider()
        st.markdown("**Codes per chunk**")
        st.caption(
            "The AI reads your text in sections called *chunks*. "
            "This setting controls how many codes it extracts from each chunk. \n\n"
            "- **2 codes per chunk** — conservative; cleaner output, less noise. "
            "Good for short module reflections (a few sentences each).\n"
            "- **3 codes per chunk** — captures more ideas per section. "
            "Better for longer interview transcripts where students cover many topics."
        )
        st.select_slider(
            "Codes per chunk", options=[2, 3],
            value=int(st.session_state.get("ita_g_n_codes", 3)),
            key="ita_g_n_codes",
        )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4 — REVIEW & EDIT PROMPT
    # ═══════════════════════════════════════════════════════════════════
    elif step == STEPS[3]:
        st.markdown("### Step 4 — Review & Edit Prompt")
        st.caption(
            "These are the instructions the AI receives before it reads your data. "
            "The prompts are pre-filled with research-validated defaults. "
            "You can edit them to better suit your cohort's age group, language, or "
            "research focus — for example, simplifying the language for younger students "
            "or asking the AI to pay attention to specific AI concepts. "
            "**Save** stores your version for this session. "
            "**Restore Original** puts the default text back."
        )

        # ── System prompt ─────────────────────────────────────────────
        st.markdown("#### System prompt")
        st.caption(
            "Defines the AI's role and overall analysis approach — sent at the start "
            "of every phase. The default instructs the AI to act as a qualitative "
            "researcher analysing AI literacy responses from young learners."
        )

        # Show saved-vs-live status
        _sys_modified = (
            st.session_state.get("ita_g_sys_prompt", _ITA_SYSTEM_PROMPT)
            != st.session_state.get("ita_g_sys_saved", _ITA_SYSTEM_PROMPT)
        )
        if _sys_modified:
            st.info("✏️ Unsaved changes — click **Save** to apply to the next run.")

        st.text_area(
            "System prompt",
            key="ita_g_sys_prompt",
            height=180,
            label_visibility="collapsed",
        )

        col_save_s, col_restore_s, _ = st.columns([1, 1.4, 4])
        if col_save_s.button("💾 Save", key="ita_sys_save",
                             help="Apply this version to the next ITA run"):
            st.session_state["ita_g_sys_saved"] = st.session_state["ita_g_sys_prompt"]
            st.success("System prompt saved.")
        if col_restore_s.button("↺ Restore Original", key="ita_sys_restore",
                                help="Reset to the default research prompt"):
            # Set flag — actual write happens at top of next render cycle
            st.session_state["_ita_reset_sys"] = True
            st.rerun()

        st.divider()

        # ── Phase 2 prompt ────────────────────────────────────────────
        st.markdown("#### Phase 2 prompt template")
        st.caption(
            "Sent to the AI once per text chunk during Phase 2 (code extraction). "
            "`{text}` is automatically replaced with each participant's text at runtime — "
            "**keep `{text}` in place** or the pipeline will fail. "
            "You can adjust how many codes to extract or what to focus on."
        )

        _p2_modified = (
            st.session_state.get("ita_g_p2_prompt", _PHASE2_PROMPT)
            != st.session_state.get("ita_g_p2_saved", _PHASE2_PROMPT)
        )
        if _p2_modified:
            st.info("✏️ Unsaved changes — click **Save** to apply to the next run.")

        st.text_area(
            "Phase 2 prompt template",
            key="ita_g_p2_prompt",
            height=220,
            label_visibility="collapsed",
        )

        col_save_p, col_restore_p, _ = st.columns([1, 1.4, 4])
        if col_save_p.button("💾 Save", key="ita_p2_save",
                             help="Apply this version to the next ITA run"):
            st.session_state["ita_g_p2_saved"] = st.session_state["ita_g_p2_prompt"]
            st.success("Phase 2 prompt saved.")
        if col_restore_p.button("↺ Restore Original", key="ita_p2_restore",
                                help="Reset to the default Phase 2 prompt"):
            st.session_state["_ita_reset_p2"] = True
            st.rerun()

        st.divider()

        # ── Phase 3 / 4 prompt ────────────────────────────────────────
        st.markdown("#### Phase 3 & 4 prompt template")
        st.caption(
            "Sent to the AI once in Phase 3 (theme search) and once in Phase 4 "
            "(theme stability check at T=1.0). Both phases use the same prompt — "
            "Phase 4 simply reruns it with higher temperature to test robustness. "
            "\n\n"
            "**Methodology note (De Paoli, 2024):** This prompt asks the model to "
            "group codes into themes — corresponding to De Paoli's Phase 4 "
            "('grouping topics'). Deduplication of redundant codes (De Paoli's "
            "Phase 3) is handled automatically by the pipeline in Phase 2b using "
            "embedding similarity, which is more reliable than asking an LLM to "
            "judge uniqueness.\n\n"
            "**Traceability:** The prompt requires the model to return `code_indices` "
            "(linking each theme back to the original transcript chunks) and verbatim "
            "`quotes` from participants — fulfilling De Paoli's anti-hallucination "
            "requirement. **Do not remove** `{n_themes}` or `{codes_list}` placeholders "
            "or the pipeline will fail."
        )

        _p3_modified = (
            st.session_state.get("ita_g_p3_prompt", _PHASE3_PROMPT)
            != st.session_state.get("ita_g_p3_saved", _PHASE3_PROMPT)
        )
        if _p3_modified:
            st.info("✏️ Unsaved changes — click **Save** to apply to the next run.")

        st.text_area(
            "Phase 3 & 4 prompt template",
            key="ita_g_p3_prompt",
            height=280,
            label_visibility="collapsed",
        )

        col_save_p3, col_restore_p3, _ = st.columns([1, 1.4, 4])
        if col_save_p3.button("💾 Save", key="ita_p3_save",
                              help="Apply this version to the next ITA run"):
            st.session_state["ita_g_p3_saved"] = st.session_state["ita_g_p3_prompt"]
            st.success("Phase 3 & 4 prompt saved.")
        if col_restore_p3.button("↺ Restore Original", key="ita_p3_restore",
                                 help="Reset to the default Phase 3 & 4 prompt"):
            st.session_state["_ita_reset_p3"] = True
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5 — RUN ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    elif step == STEPS[4]:
        st.markdown("### Step 5 — Run Analysis")

        # ── Read sources from stable keys written by Step 1 ──────────
        # These survive step navigation; the widget keys (ita_g_src_*)
        # are wiped by Streamlit when Step 1 is not rendered.
        sources = []
        if st.session_state.get("ita_src_responses",  True):  sources.append("responses")
        if st.session_state.get("ita_src_persistent", False): sources.append("persistent")
        if st.session_state.get("ita_src_observer",   False): sources.append("observer")
        if st.session_state.get("ita_src_per_run",    False): sources.append("per_run")
        #per_run_files = st.session_state.get("ita_g_upload") #remove upload now in llm analysis, ITA
        per_run_files = st.session_state.get("ita_g_upload_store")

        _src_labels = {
            "responses":  "Module reflections (DB)",
            "persistent": "Interview transcripts (store)",
            "observer":   "Observer/instructor transcript(s) (store)",
            "per_run":    "Uploaded files (this run)",
        }
        if sources:
            st.caption(
                "**Sources from Step 1:** "
                + "  ·  ".join(_src_labels.get(s, s) for s in sources)
                + " — go back to Step 1 to change."
            )
        else:
            st.warning("No data sources selected — go back to Step 1.")
            return

        _ita_cohort_ids = st.session_state.get("ita_cohort_filter", [])
        st.caption(
            "**Cohort scope:** " + (", ".join(_ita_cohort_ids) if _ita_cohort_ids else "All cohorts (no filter)")
        )

        models = st.session_state.get("ita_g_models", [])

        temperature = float(st.session_state.get("ita_g_temp", 0.0))
        n_themes    = int(st.session_state.get("ita_g_n_themes", 5))
        n_codes     = int(st.session_state.get("ita_g_n_codes", 3))
        dedup_thr   = float(st.session_state.get("ita_g_dedup", 0.85))

        if not models:
            st.warning("No models selected — go back to Step 2.")
            return
        if not sources:
            st.warning("Select at least one data source — go back to Step 1.")
            return

        # ── Available counts ──────────────────────────────────────────
        st.divider()
        st.markdown("**How many texts to include in this run?**")
        st.caption(
            "The counts below show what is currently available in the database"
            + (f", scoped to **{', '.join(_ita_cohort_ids)}**" if _ita_cohort_ids else "")
            + ". Use a smaller number to keep costs low when testing a paid model — "
            "for example, 2 reflections + 1 interview is enough to confirm the "
            "full pipeline works end-to-end. Set to the full count for a production run."
        )

        _avail_counts = _count_available_sources(cohort_ids=_ita_cohort_ids or None)
        n_avail_ref = _avail_counts["reflections"]
        n_avail_int = _avail_counts["interviews"]
        n_avail_obs = _avail_counts["observer"]

        _cohort_suffix = f" ({'/'.join(_ita_cohort_ids)})" if _ita_cohort_ids else ""
        ca, cb, ce = st.columns(3)
        ca.metric(f"Reflections available{_cohort_suffix}", n_avail_ref)
        cb.metric(f"Interviews available{_cohort_suffix}",  n_avail_int)
        ce.metric(f"Observer/instructor transcripts available{_cohort_suffix}", n_avail_obs)
        if "observer" in sources and n_avail_obs > 0:
            st.caption(
                f"All {n_avail_obs} observer/instructor transcript(s) will be "
                "included automatically (no separate count control yet, "
                "unlike reflections/interviews above)."
            )

        cc, cd = st.columns(2)
        n_ref_use = cc.number_input(
            "Reflections to include",
            min_value=0,
            max_value=max(n_avail_ref, 1),
            value=min(int(st.session_state.get("ita_g_max_ref", 2)), n_avail_ref),
            step=1,
            key="ita_g_max_ref",
            help=(
                f"{n_avail_ref} available  ·  "
                "0 = skip reflections entirely  ·  "
                f"{n_avail_ref} = use all"
            ),
        )
        n_int_use = cd.number_input(
            "Interviews to include",
            min_value=0,
            max_value=max(n_avail_int, 1),
            value=min(int(st.session_state.get("ita_g_max_int", 1)), max(n_avail_int, 1)),
            step=1,
            key="ita_g_max_int",
            help=(
                f"{n_avail_int} available  ·  "
                "0 = skip interviews entirely  ·  "
                f"{n_avail_int} = use all"
            ),
        )
        n_total_texts = int(n_ref_use) + int(n_int_use)

        # ── Source / count consistency check ─────────────────────────
        # If the teacher asked for interviews but forgot to tick the interview
        # source in Step 1 (or vice versa), auto-correct and notify clearly.
        _src_auto_added = []
        if int(n_int_use) > 0 and "persistent" not in sources:
            sources.append("persistent")
            _src_auto_added.append("Interview transcripts (store)")
        if int(n_ref_use) > 0 and "responses" not in sources:
            sources.append("responses")
            _src_auto_added.append("Module reflections (DB)")
        if int(n_ref_use) == 0 and "responses" in sources:
            sources.remove("responses")
        if int(n_int_use) == 0 and "persistent" in sources:
            sources.remove("persistent")

        if _src_auto_added:
            st.info(
                "ℹ️ **Source automatically added:** "
                + ", ".join(_src_auto_added)
                + " — you requested texts from this source but it was not "
                "selected in Step 1. It has been added for this run. "
                "Go back to Step 1 to make this permanent."
            )

        # ── Settings summary ──────────────────────────────────────────
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Models",       ", ".join(m.title() for m in models))
        c2.metric("Temperature",  f"{temperature:.1f}")
        c3.metric("Themes",       n_themes)
        c4.metric("Reflections",  int(n_ref_use))
        c5.metric("Interviews",   int(n_int_use))

        # ── Cost estimate ─────────────────────────────────────────────
        est_lines = _ita_cost_estimate(n_total_texts, models)
        st.info(
            f"💰 **Estimated cost** ({int(n_ref_use)} reflection(s) + "
            f"{int(n_int_use)} interview(s) = {n_total_texts} text(s)): "
            + "  ·  ".join(est_lines)
        )

        # Warn if Groq selected and estimated tokens approach free tier daily limit
        if "groq" in models:
            # Rough estimate: ~700 input + ~400 output tokens per chunk,
            # ~2 chunks per text across 6 phases
            _est_tokens = n_total_texts * 2 * 1100
            if _est_tokens > 80_000:
                st.warning(
                    f"⚠️ **Groq daily token limit:** This run may use ~{_est_tokens:,} tokens. "
                    f"The free tier allows 100,000 tokens/day. "
                    f"Consider reducing the number of texts or running in smaller batches."
                )

        if n_total_texts == 0:
            st.warning("Set at least one reflection or interview above.")
            return

        if st.button("▶ Run ITA Pipeline", key="ita_g_run", type="primary"):
            _run_ita_pipeline(
                username=username, sources=sources, per_run_files=per_run_files,
                models=models, temperature=temperature,
                n_themes=n_themes, n_codes=n_codes, dedup_threshold=dedup_thr,
                max_reflections=int(n_ref_use),
                max_interviews=int(n_int_use),
                cohort_ids=_ita_cohort_ids,
                custom_sys_prompt=st.session_state.get("ita_g_sys_saved"),
                custom_p2_prompt=st.session_state.get("ita_g_p2_saved"),
                custom_p3_prompt=st.session_state.get("ita_g_p3_saved"),
            )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 6 — RESULTS
    # ═══════════════════════════════════════════════════════════════════
    elif step == STEPS[5]:
        st.markdown("### Step 6 — Results")
        runs = _ita_list_runs(created_by=username)
        if not runs:
            st.info("No ITA runs found. Complete Step 5 first."); return

        run_opts = {
            f"{r['model'].upper()} T={r['temperature']} — "
            f"{r['created_at'][:16]} [{r['status']}]": r["run_id"]
            for r in runs[:10]
        }
        # Default to the most recently completed run (set by pipeline on finish)
        _default_run_id  = st.session_state.get("ita_last_run_id")
        _default_run_key = next(
            (k for k, v in run_opts.items() if v == _default_run_id),
            list(run_opts.keys())[0]
        )
        selected = st.selectbox(
            "Select run", list(run_opts.keys()),
            index=list(run_opts.keys()).index(_default_run_key),
            key="ita_results_run",
        )
        run_id  = run_opts[selected]
        run_rec = _ita_get_run(run_id)
        st.caption(
            f"Model: **{_llm_display_name(run_rec['model'])}** | "
            f"T={run_rec['temperature']} | "
            f"Phase reached: {run_rec['phase_reached']}"
        )
        st.divider()

        view_tab = st.radio(
            "View", ["Codes", "Themes", "Phase 4 Review", "Report", "Compare Runs"],
            horizontal=True, key="ita_results_section",
        )
        st.divider()

        p2d = load_phase_result(run_id, 2)
        p3d = load_phase_result(run_id, 3)
        p4d = load_phase_result(run_id, 4)
        p5d = load_phase_result(run_id, 5)
        p6d = load_phase_result(run_id, 6)

        # ── Codes ──────────────────────────────────────────────────────
        if view_tab == "Codes":
            if not p2d: st.info("Phase 2 not yet run."); return
            codes   = p2d.get("codes", [])
            dedup   = p2d.get("dedup", {})
            n_raw   = p2d.get("n_codes_raw", len(codes))
            n_after = dedup.get("n_after", len(codes))
            n_rem   = dedup.get("n_removed", 0)
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Raw codes",   n_raw)
            cm2.metric("After dedup", n_after)
            cm3.metric("Removed",     n_rem,
                       delta=f"-{n_rem}" if n_rem else None,
                       delta_color="inverse")

            sub = st.radio(
                "Show",
                ["Raw codes", "After dedup", "Removed by dedup"],
                horizontal=True, key="ita_codes_sub",
            )
            st.divider()

            if sub == "Raw codes":
                if codes:
                    rows = [{"Chunk": c.get("chunk_index", ""),
                             "Participant": c.get("participant_id", ""),
                             "Code": c.get("name", ""),
                             "Description": c.get("description", "")[:100]}
                            for c in codes]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                else:
                    st.info("No raw codes found.")

            elif sub == "After dedup":
                dedup_codes = dedup.get("codes_dedup", [])
                if dedup_codes:
                    st.caption(
                        f"**{len(dedup_codes)} code(s) after deduplication** "
                        f"(threshold = {dedup.get('threshold', 0.85):.2f}). "
                        "Codes with the same name from **different participants** "
                        "are intentionally kept — they represent independent evidence "
                        "from separate people. Only codes from the *same or different* "
                        "participants whose name **and** description are highly similar "
                        "(above the threshold) are merged."
                    )
                    rows = [{"Chunk": c.get("chunk_index", ""),
                             "Participant": c.get("participant_id", ""),
                             "Code": c.get("name", ""),
                             "Description": c.get("description", "")[:120]}
                            for c in dedup_codes]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                else:
                    st.info("No deduplicated codes found.")

            elif sub == "Removed by dedup":
                removed = dedup.get("removed_codes", [])
                if removed:
                    st.caption(
                        f"**{len(removed)} code(s) removed** — each was semantically "
                        f"too similar (≥ {dedup.get('threshold', 0.85):.2f} cosine "
                        "similarity) to a code from an earlier chunk. "
                        "The earliest-chunk version is always kept (De Paoli "
                        "anti-hallucination: lowest chunk_index wins)."
                    )
                    rows = [{"Removed chunk": r.get("chunk_index", ""),
                             "Participant":   r.get("participant_id", ""),
                             "Removed code":  r.get("name", ""),
                             "Description":   r.get("description", "")[:100]}
                            for r in removed]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                else:
                    st.info(
                        "No codes were removed in deduplication. "
                        "If the 'Removed' count above is > 0 but this table is empty, "
                        "re-run the analysis — older runs pre-date this feature."
                    )

        # ── Themes ─────────────────────────────────────────────────────
        elif view_tab == "Themes":
            data   = p5d or p3d
            if not data: st.info("Phase 3 not yet run."); return
            themes = data.get("themes_defined") or data.get("themes", [])
            st.caption(
                f"{len(themes)} theme(s) identified. "
                "Each theme groups related codes under a shared idea."
            )
            for i, t in enumerate(themes):
                with st.expander(f"**{i+1}. {t.get('name', '')}**",
                                 expanded=(i == 0)):
                    st.markdown(t.get("summary", t.get("description", "")))
                    codes_in_theme = t.get("codes", [])
                    if codes_in_theme:
                        st.caption(f"Supporting codes ({len(codes_in_theme)}):")
                        for c in codes_in_theme[:10]:
                            lbl = c if isinstance(c, str) else c.get("name", str(c))
                            st.markdown(f"- {lbl}")

        # ── Phase 4 Review ─────────────────────────────────────────────
        elif view_tab == "Phase 4 Review":
            if not p3d or not p4d:
                st.info("Both Phase 3 and Phase 4 must be run first."); return
            t0 = p3d.get("themes", [])
            t1 = p4d.get("themes", [])
            st.markdown(
                f"**Phase 3 → {len(t0)} themes  |  Phase 4 (T=1.0) → {len(t1)} themes**"
            )
            st.caption(
                "Phase 4 re-runs the theme search at Temperature = 1.0 to check "
                "whether your themes hold up under variation. "
                "**Agreement scores above 0.80** suggest the themes are robust "
                "and not just an artefact of one particular run."
            )
            if t0 and t1:
                with st.spinner("Computing alignment…"):
                    aligned = _align_themes(t0, t1, "T=0", "T=1.0")
                st.dataframe(aligned, hide_index=True, width="stretch")
                if not aligned.empty:
                    st.metric("Mean agreement", f"{aligned['Agreement Score'].mean():.3f}")

        # ── Report ─────────────────────────────────────────────────────
        elif view_tab == "Report":
            if not p6d: st.info("Phase 6 not yet run."); return
            report = p6d.get("report_text", "")
            if report:
                st.markdown(report)
                st.download_button(
                    "📄 Download report (.txt)", data=report,
                    file_name=f"ita_report_{run_id[:8]}.txt",
                    mime="text/plain", key="ita_dl_report",
                )

        # ── Compare Runs ───────────────────────────────────────────────
        elif view_tab == "Compare Runs":
            completed = [r for r in runs if r["phase_reached"] >= 3]
            if len(completed) < 2:
                st.info("Need at least 2 completed runs to compare."); return
            st.caption(
                "Compare theme sets from two runs — e.g. different models or "
                "temperatures — to check how consistent the analysis is. "
                "High agreement means the themes are stable regardless of model choice."
            )
            opts = {
                f"{r['model'].upper()} T={r['temperature']} {r['created_at'][:10]}":
                r["run_id"] for r in completed[:8]
            }
            ca, cb = st.columns(2)
            ra = ca.selectbox("Run A", list(opts.keys()), key="ita_cmp_a")
            rb = cb.selectbox("Run B", list(opts.keys()),
                              index=min(1, len(opts) - 1), key="ita_cmp_b")
            if ra == rb: st.warning("Select two different runs."); return
            pa = load_phase_result(opts[ra], 3)
            pb = load_phase_result(opts[rb], 3)
            if pa and pb:
                with st.spinner("Computing agreement…"):
                    cmp = _compare_runs(
                        pa.get("themes", []), pb.get("themes", []), ra, rb
                    )
                if cmp.get("error"): st.error(cmp["error"]); return
                cc1, cc2 = st.columns(2)
                cc1.metric("Overall agreement", f"{cmp['overall_agreement']:.3f}")
                cc2.metric("Interpretation",    cmp["interpretation"])
                bm = pd.DataFrame(cmp.get("best_matches", []))
                if not bm.empty:
                    st.dataframe(bm, hide_index=True, width="stretch")


# -----------------------------------------------------------------------
# ITA — Expert mode (merged into guided — kept for backward compatibility)
# -----------------------------------------------------------------------

def _render_ita_expert(username: str, canonical_df: pd.DataFrame) -> None:
    """Expert mode merged into the unified guided interface."""
    _render_ita_guided(username, canonical_df)


# -----------------------------------------------------------------------
# ITA — Pipeline runner
# -----------------------------------------------------------------------

def _run_ita_pipeline(
    username, sources, per_run_files,
    models, temperature, n_themes, n_codes, dedup_threshold,
    max_reflections=None,
    max_interviews=None,
    max_transcripts=None,   # legacy fallback — use max_reflections/max_interviews instead
    custom_sys_prompt=None,
    custom_p2_prompt=None,
    custom_p3_prompt=None,
    cohort_ids=None,
):
    src_label = " + ".join(sources) if sources else "none"
    with st.spinner(f"Loading transcripts from: {src_label}..."):
        try:
            transcripts = _load_combined_transcripts(sources, per_run_files, cohort_ids=cohort_ids)
        except Exception as e:
            st.error(f"Could not load transcripts: {e}"); return

    if not transcripts:
        st.warning(f"No transcript data found in: {src_label}."); return

    # ── Per-type subsetting ──────────────────────────────────────────
    # Split loaded transcripts by source_types tag, apply limits, then
    # recombine — gives the teacher independent control over each type.
    if max_reflections is not None or max_interviews is not None:
        refs = [t for t in transcripts if "reflections" in t.get("source_types", [])]
        ints = [t for t in transcripts if "interview"   in t.get("source_types", [])]
        others = [t for t in transcripts
                  if "reflections" not in t.get("source_types", [])
                  and "interview"  not in t.get("source_types", [])]
        if max_reflections is not None:
            refs = refs[:max_reflections]
        if max_interviews is not None:
            ints = ints[:max_interviews]
        transcripts = refs + ints + others
        st.caption(
            f"Using {len(refs)} reflection(s) + {len(ints)} interview(s)"
            + (f" + {len(others)} other(s)" if others else "") + "."
        )
    elif max_transcripts and len(transcripts) > max_transcripts:
        # Legacy path
        transcripts = transcripts[:max_transcripts]
        st.caption(f"Using first {max_transcripts} text(s).")

    # Apply custom prompts by patching module-level variables before pipeline runs
    import core.analytics.llm.ita_pipeline as _ita_mod
    _orig_sys = getattr(_ita_mod, "SYSTEM_PROMPT", None)
    _orig_p2  = getattr(_ita_mod, "_PHASE2_PROMPT", None)
    _orig_p3  = getattr(_ita_mod, "_PHASE3_PROMPT", None)
    if custom_sys_prompt and custom_sys_prompt != _ITA_SYSTEM_PROMPT:
        _ita_mod.SYSTEM_PROMPT = custom_sys_prompt
    if custom_p2_prompt and custom_p2_prompt != _PHASE2_PROMPT:
        _ita_mod._PHASE2_PROMPT = custom_p2_prompt
    if custom_p3_prompt and custom_p3_prompt != _PHASE3_PROMPT:
        _ita_mod._PHASE3_PROMPT = custom_p3_prompt

    st.info(f"Loaded {len(transcripts)} participant(s). "
            f"Running {len(models)} model(s).")

    for model in models:
        st.markdown(f"---\n#### {_llm_display_name(model)}")
        run_id = _ita_create_run(
            model=model, temperature=temperature,
            source_type="+".join(sources),
            created_by=username,
            notes=f"n_themes={n_themes}, n_codes={n_codes}",
            cohort_scope=", ".join(cohort_ids) if cohort_ids else "All cohorts",
        )

        with st.status("Phase 1 — Chunking...", expanded=False) as s:
            chunks = run_phase1(transcripts, chunk_size=2500)
            s.update(label=f"Phase 1 ✅ — {len(chunks)} chunks", state="complete")

        with st.expander(
            f"🧩 AI Coding Process Insights — Phase 1: Chunking  "
            f"({len(chunks)} chunk(s) created)",
            expanded=False,
        ):
            st.caption(
                "The AI splits each participant's text into overlapping chunks "
                "so that long responses are not cut off mid-thought. "
                f"**{len(chunks)} chunk(s)** were created from "
                f"{len(transcripts)} participant text(s)."
            )

        with st.status(f"Phase 2 — Generating codes (T={temperature})...",
                       expanded=False) as s:
            p2 = run_phase2(chunks, model, temperature,
                            n_codes=n_codes, run_id=run_id)
            n_codes_raw = p2.get("n_codes_raw", 0)
            _p2_ok = n_codes_raw > 0
            s.update(
                label=f"Phase 2 {'✅' if _p2_ok else '⚠️'} — {n_codes_raw} codes",
                state="complete" if _p2_ok else "error",
            )

        with st.expander(
            f"🧩 AI Coding Process Insights — Phase 2: Code Extraction  "
            f"({'✅ ' + str(n_codes_raw) + ' codes extracted' if _p2_ok else '⚠️ 0 codes — see details below'})",
            expanded=not _p2_ok,
        ):
            st.caption(
                "The AI reads each chunk and extracts short descriptive labels "
                "called **codes** — concise phrases that capture a key idea in "
                "the text (e.g. 'uncertainty about AI decisions'). "
                f"**{n_codes_raw} code(s)** were extracted across all chunks."
            )
            if _p2_ok:
                st.caption("Sample codes (first 3):")
                st.json(p2.get("codes", [])[:3])
            else:
                st.warning("No codes were extracted. Details below.")
                _p2_errors = p2.get("errors")
                if _p2_errors:
                    st.caption("Errors captured inside the pipeline:")
                    st.json(_p2_errors) if isinstance(_p2_errors, (list, dict)) \
                        else st.code(str(_p2_errors))
                else:
                    st.caption("No error details stored — check ita_pipeline.run_phase2().")

        with st.status("Phase 2b — Deduplicating codes...", expanded=False) as s:
            dedup = run_phase2_dedup(p2["codes"], threshold=dedup_threshold)
            save_phase_result(run_id, 2, {**p2, "dedup": dedup})
            s.update(
                label=f"Phase 2b ✅ — "
                      f"{dedup['n_before']}→{dedup['n_after']} codes "
                      f"({dedup.get('n_removed', 0)} removed)",
                state="complete",
            )

        with st.expander(
            f"🧩 AI Coding Process Insights — Phase 2b: Deduplication  "
            f"({dedup.get('n_before', 0)} → {dedup.get('n_after', 0)} codes, "
            f"{dedup.get('n_removed', 0)} removed)",
            expanded=False,
        ):
            st.caption(
                "Different chunks often produce very similar codes. "
                "Deduplication uses semantic similarity to merge near-duplicates, "
                "keeping the code list clean before themes are built. "
                f"Threshold used: **{dedup_threshold}** "
                "(higher = stricter, keeps more codes)."
            )

        with st.status("Phase 3 — Identifying themes...", expanded=False) as s:
            p3 = run_phase3(dedup["codes_dedup"], model, temperature,
                            n_themes=n_themes, run_id=run_id)
            if p3.get("error"):
                s.update(label=f"Phase 3 ❌ — {p3['error']}", state="error")
            else:
                s.update(
                    label=f"Phase 3 ✅ — {p3['n_themes']} themes identified",
                    state="complete",
                )

        with st.expander(
            f"🧩 AI Coding Process Insights — Phase 3: Theme Identification  "
            f"({'❌ ' + str(p3.get('error','')) if p3.get('error') else '✅ ' + str(p3.get('n_themes', 0)) + ' theme(s) identified'})",
            expanded=bool(p3.get("error")),
        ):
            st.caption(
                "The AI groups related codes into broader **themes** — "
                "recurring patterns that cut across multiple participants and chunks. "
                f"**{p3.get('n_themes', 0)} theme(s)** were identified from "
                f"{dedup.get('n_after', 0)} deduplicated codes."
            )
            if p3.get("error"):
                st.error(p3["error"])

        if p3.get("error"):
            continue

        with st.status("Phase 4 — Reviewing themes (T=1.0)...", expanded=False) as s:
            p4 = run_phase4(dedup["codes_dedup"], model, temperature=1.0,
                            n_themes=n_themes, run_id=run_id)
            if p4.get("error"):
                s.update(label=f"Phase 4 ❌ — {p4['error']}", state="error")
            else:
                s.update(
                    label=f"Phase 4 ✅ — {p4['n_themes']} themes at T=1.0 "
                          f"(stability check)",
                    state="complete",
                )

        with st.expander(
            f"🧩 AI Coding Process Insights — Phase 4: Theme Stability Check  "
            f"({'❌ error' if p4.get('error') else str(p4.get('n_themes', 0)) + ' theme(s) at T=1.0'})",
            expanded=False,
        ):
            st.caption(
                "The same theme search is re-run at **Temperature = 1.0** "
                "(more creative/varied). If the themes found here closely match "
                "Phase 3 (agreement > 0.80), it confirms the themes are robust "
                "and not just an artefact of one particular run. "
                "You can compare these in Step 6 → Phase 4 Review."
            )
            if p4.get("error"):
                st.error(p4["error"])

        with st.status("Phase 5 — Defining themes...", expanded=False) as s:
            p5 = run_phase5(p3["themes"], dedup["codes_dedup"], model,
                            temperature, run_id=run_id)
            n_defined = len(p5.get("themes_defined", []))
            s.update(
                label=f"Phase 5 ✅ — {n_defined} themes defined with descriptions",
                state="complete",
            )

        with st.expander(
            f"🧩 AI Coding Process Insights — Phase 5: Theme Definition  "
            f"({n_defined} theme(s) defined)",
            expanded=False,
        ):
            st.caption(
                "Each theme is given a formal name, a one-paragraph description, "
                "and a list of the supporting codes that belong to it. "
                "These definitions form the basis of the final report."
            )

        with st.status("Phase 6 — Writing report...", expanded=False) as s:
            p6 = run_phase6(p5["themes_defined"], dedup["codes_dedup"],
                            model, temperature, run_id=run_id)
            if p6.get("error"):
                s.update(label=f"Phase 6 ❌ — {p6['error']}", state="error")
            else:
                n_chars = len(p6.get("report_text", ""))
                s.update(
                    label=f"Phase 6 ✅ — Report written ({n_chars:,} characters)",
                    state="complete",
                )

        with st.expander(
            f"🧩 AI Coding Process Insights — Phase 6: Report Generation  "
            f"({'❌ error' if p6.get('error') else str(len(p6.get('report_text',''))) + ' characters written'})",
            expanded=False,
        ):
            st.caption(
                "The AI synthesises all themes and supporting evidence into a "
                "structured qualitative research report suitable for inclusion "
                "in a study writeup. The full report is available in Step 6 → Report."
            )
            if p6.get("error"):
                st.error(p6["error"])

        st.session_state["ita_last_run_id"] = run_id
        st.session_state["ita_results_run_default"] = run_id

    # ── Pipeline complete — use flag to navigate, not direct key write ──
    st.success("✅ ITA complete!")
    st.session_state["_ita_goto_step6"] = True
    if st.button("➡ View Results (Step 6)", type="primary",
                 key="ita_goto_results"):
        st.rerun()

    # Restore original module-level prompts
    if _orig_sys is not None:
        _ita_mod.SYSTEM_PROMPT = _orig_sys
    if _orig_p2 is not None:
        _ita_mod._PHASE2_PROMPT = _orig_p2


# -----------------------------------------------------------------------
# ITA — Results
# -----------------------------------------------------------------------

def _render_ita_results(username: str) -> None:
    runs = _ita_list_runs(created_by=username)
    if not runs:
        st.info("No ITA runs found. Run the pipeline first."); return

    run_opts = {
        f"{r['model'].upper()} T={r['temperature']} — "
        f"{r['created_at'][:16]} [{r['status']}]": r["run_id"]
        for r in runs[:10]
    }
    selected = st.selectbox("Select run", list(run_opts.keys()),
                            key="ita_results_run")
    run_id  = run_opts[selected]
    run_rec = _ita_get_run(run_id)
    st.caption(f"Model: **{_llm_display_name(run_rec['model'])}** | "
               f"T={run_rec['temperature']} | Phase reached: {run_rec['phase_reached']}")

    section = st.radio("View", ["Codes", "Themes", "Phase 4 Review", "Report",
                                 "Compare Runs"],
                        horizontal=True, key="ita_results_section")
    st.divider()

    p2d = load_phase_result(run_id, 2)
    p3d = load_phase_result(run_id, 3)
    p4d = load_phase_result(run_id, 4)
    p5d = load_phase_result(run_id, 5)
    p6d = load_phase_result(run_id, 6)

    if section == "Codes":
        if not p2d: st.info("Phase 2 not yet run."); return
        codes = p2d.get("codes", [])
        dedup = p2d.get("dedup", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw codes",   p2d.get("n_codes_raw", len(codes)))
        c2.metric("After dedup", dedup.get("n_after", len(codes)))
        c3.metric("Removed",     dedup.get("n_removed", 0))
        if codes:
            import json as _jc
            rows = [{"Chunk": c.get("chunk_index",""),
                     "Participant": c.get("participant_id",""),
                     "Code": c.get("name",""),
                     "Description": c.get("description","")[:100]}
                    for c in codes]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    elif section == "Themes":
        data = p5d or p3d
        if not data: st.info("Phase 3 not yet run."); return
        themes = data.get("themes_defined") or data.get("themes", [])
        for i, t in enumerate(themes):
            with st.expander(f"**{i+1}. {t.get('name','')}**", expanded=(i==0)):
                st.markdown(t.get("summary", t.get("description", "")))

    elif section == "Phase 4 Review":
        if not p3d or not p4d:
            st.info("Both Phase 3 and Phase 4 must be run first."); return
        t0 = p3d.get("themes", []); t1 = p4d.get("themes", [])
        st.markdown(f"**Phase 3 → {len(t0)} themes | Phase 4 (T=1.0) → {len(t1)} themes**")
        if t0 and t1:
            with st.spinner("Computing alignment..."):
                aligned = _align_themes(t0, t1, "T=0", "T=1.0")
            st.dataframe(aligned, hide_index=True, width="stretch")
            if not aligned.empty:
                st.metric("Mean agreement", f"{aligned['Agreement Score'].mean():.3f}")

    elif section == "Report":
        if not p6d: st.info("Phase 6 not yet run."); return
        report = p6d.get("report_text", "")
        if report:
            st.markdown(report)
            st.download_button("📄 Download report (.txt)", data=report,
                               file_name=f"ita_report_{run_id[:8]}.txt",
                               mime="text/plain", key="ita_dl_report")

    elif section == "Compare Runs":
        completed = [r for r in runs if r["phase_reached"] >= 3]
        if len(completed) < 2:
            st.info("Need at least 2 completed runs."); return
        opts = {f"{r['model'].upper()} T={r['temperature']} {r['created_at'][:10]}":
                r["run_id"] for r in completed[:8]}
        c1, c2 = st.columns(2)
        ra = c1.selectbox("Run A", list(opts.keys()), key="ita_cmp_a")
        rb = c2.selectbox("Run B", list(opts.keys()),
                          index=min(1, len(opts)-1), key="ita_cmp_b")
        if ra == rb: st.warning("Select two different runs."); return
        pa = load_phase_result(opts[ra], 3)
        pb = load_phase_result(opts[rb], 3)
        if pa and pb:
            with st.spinner("Computing agreement..."):
                cmp = _compare_runs(pa.get("themes",[]), pb.get("themes",[]), ra, rb)
            if cmp.get("error"): st.error(cmp["error"]); return
            c1, c2 = st.columns(2)
            c1.metric("Overall agreement", f"{cmp['overall_agreement']:.3f}")
            c2.metric("Interpretation",    cmp["interpretation"])
            bm = pd.DataFrame(cmp.get("best_matches", []))
            if not bm.empty:
                st.dataframe(bm, hide_index=True, width="stretch")


# -----------------------------------------------------------------------
# DTA section
# -----------------------------------------------------------------------

def _render_dta_section(username: str, canonical_df: pd.DataFrame) -> None:
    view = st.radio("View",
                    ["⚙️ Run Analysis", "📊 Results", "🎓 Learning Objectives"],
                    horizontal=True, key="dta_view_radio")
    st.divider()
    if view == "⚙️ Run Analysis":
        _render_dta_run_panel(username, canonical_df)
    elif view == "📊 Results":
        _render_dta_results_panel(username)
    else:
        _render_dta_lo_panel(username)


def _render_dta_run_panel(username: str, canonical_df: pd.DataFrame) -> None:
    # ── Session-state defaults for DTA prompts ─────────────────────────
    _dta_prompt_defaults = {
        "dta_sys_prompt":   DTA_SYSTEM_PROMPT,
        "dta_sys_saved":    DTA_SYSTEM_PROMPT,
        "dta_p2_prompt":    _DTA_PHASE2_PROMPT,
        "dta_p2_saved":     _DTA_PHASE2_PROMPT,
        "dta_lo_prompt":    _DTA_LO_PROMPT,
        "dta_lo_saved":     _DTA_LO_PROMPT,
        "dta_p5_prompt":    _DTA_PHASE5_PROMPT,
        "dta_p5_saved":     _DTA_PHASE5_PROMPT,
    }
    for _k, _d in _dta_prompt_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _d
    # Reset flags — applied before widgets render
    for _flag, _key, _default in [
        ("_dta_reset_sys", "dta_sys_prompt", DTA_SYSTEM_PROMPT),
        ("_dta_reset_sys", "dta_sys_saved",  DTA_SYSTEM_PROMPT),
        ("_dta_reset_p2",  "dta_p2_prompt",  _DTA_PHASE2_PROMPT),
        ("_dta_reset_p2",  "dta_p2_saved",   _DTA_PHASE2_PROMPT),
        ("_dta_reset_lo",  "dta_lo_prompt",  _DTA_LO_PROMPT),
        ("_dta_reset_lo",  "dta_lo_saved",   _DTA_LO_PROMPT),
        ("_dta_reset_p5",  "dta_p5_prompt",  _DTA_PHASE5_PROMPT),
        ("_dta_reset_p5",  "dta_p5_saved",   _DTA_PHASE5_PROMPT),
    ]:
        if st.session_state.pop(_flag, False):
            st.session_state[_key] = _default

    col1, col2 = st.columns(2)
    with col1:
        _avail = get_available_models(check_keys=True)
        st.markdown("**Models:**")
        dta_models = []
        mc1, mc2, mc3, mc4 = st.columns(4)
        if mc1.checkbox("Claude", value=_avail.get("claude", False),
                        disabled=not _avail.get("claude", False), key="dta_claude"):
            dta_models.append("claude")
        if mc2.checkbox("Gemini", value=False,
                        disabled=not _avail.get("gemini", False), key="dta_gemini"):
            dta_models.append("gemini")
        if mc3.checkbox("GPT", value=False,
                        disabled=not _avail.get("gpt", False), key="dta_gpt"):
            dta_models.append("gpt")
        if mc4.checkbox("Groq (free ✅)", value=_avail.get("groq", False),
                        disabled=not _avail.get("groq", False), key="dta_groq"):
            dta_models.append("groq")
    with col2:
        dta_temp = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                             format="%.1f", key="dta_temp")

    st.divider()
    sources, per_run_files = _source_checkboxes("dta")
    dta_cohort_ids = _cohort_filter_multiselect("dta")

    st.divider()
    st.markdown("**Construct groups:**")
    _grp_opts = {
        "messaging_perception":       "Messaging Perception (CCCES)",
        "individual_characteristics": "Individual Characteristics (CCCES)",
        "cognitive_engagement":       "Cognitive Engagement (SCES)",
        "motivation":                 "Motivation (SIMS)",
        "ai_understanding":           "AI Understanding (AI-CI)",
        "ai_misconceptions":          "AI Misconceptions (AIM-F)",
    }
    dta_groups = []
    gcols = st.columns(3)
    for i, (gk, gl) in enumerate(_grp_opts.items()):
        if gcols[i % 3].checkbox(gl, value=True, key=f"dta_grp_{gk}"):
            dta_groups.append(gk)

    with st.expander("📋 Codebook preview", expanded=False):
        for gk, gl in _grp_opts.items():
            if gk not in dta_groups: continue
            st.markdown(f"**{gl}**")
            for cn, cd in _DTA_CODEBOOK.items():
                if cd["group"] != gk: continue
                st.caption(f"  {cn.replace('_',' ').title()}: "
                           + " · ".join(f'"{i}"' for i in cd["indicators"][:2]) + " …")

    st.divider()
    show_stream = st.checkbox(
        "Show live coding detail (prompt + response per construct)",
        value=False, key="dta_show_stream",
        help="Tick BEFORE clicking Run DTA to avoid aborting the pipeline."
    )

    if not dta_models:
        st.warning("Select at least one model.")
        return
    if not sources:
        st.warning("Select at least one data source.")
        return

    # ── Per-type text selection ───────────────────────────────────────
    st.divider()
    st.markdown("**How many texts to include in this run?**")
    st.caption(
        "Showing what is currently available"
        + (f", scoped to **{', '.join(dta_cohort_ids)}**" if dta_cohort_ids else "")
        + ". Use smaller numbers to test with a paid model before committing to a full run — "
        "even 2 reflections + 1 interview is enough to verify the pipeline and output quality."
    )

    _dta_counts = _count_available_sources(cohort_ids=dta_cohort_ids or None)
    n_dta_avail_ref = _dta_counts["reflections"]
    n_dta_avail_int = _dta_counts["interviews"]
    n_dta_avail_obs = _dta_counts["observer"]

    _dta_cohort_suffix = f" ({'/'.join(dta_cohort_ids)})" if dta_cohort_ids else ""
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric(f"Reflections available{_dta_cohort_suffix}", n_dta_avail_ref)
    dc2.metric(f"Interviews available{_dta_cohort_suffix}",  n_dta_avail_int)
    dc3.metric(f"Observer/instructor transcripts available{_dta_cohort_suffix}", n_dta_avail_obs)
    if "observer" in sources and n_dta_avail_obs > 0:
        st.caption(
            f"All {n_dta_avail_obs} observer/instructor transcript(s) will be "
            "included automatically (no separate count control yet, "
            "unlike reflections/interviews above)."
        )

    di1, di2 = st.columns(2)
    dta_n_ref = di1.number_input(
        "Reflections to include",
        min_value=0,
        max_value=max(n_dta_avail_ref, 1),
        value=min(int(st.session_state.get("dta_max_ref", 2)), n_dta_avail_ref),
        step=1,
        key="dta_max_ref",
        help=f"{n_dta_avail_ref} available  ·  0 = skip  ·  {n_dta_avail_ref} = use all",
    )
    dta_n_int = di2.number_input(
        "Interviews to include",
        min_value=0,
        max_value=max(n_dta_avail_int, 1),
        value=min(int(st.session_state.get("dta_max_int", 1)), max(n_dta_avail_int, 1)),
        step=1,
        key="dta_max_int",
        help=f"{n_dta_avail_int} available  ·  0 = skip  ·  {n_dta_avail_int} = use all",
    )
    n_dta_total = int(dta_n_ref) + int(dta_n_int)

    # ── Cost estimates ────────────────────────────────────────────────
    st.divider()
    est_dta = _dta_cost_estimate(max(n_dta_total, 1), len(dta_groups), dta_models)
    est_lo  = _dta_cost_estimate(max(n_dta_total, 1), len(_DTA_LO), dta_models)

    st.info(
        f"💰 **Estimated cost — Run DTA** "
        f"({int(dta_n_ref)} reflection(s) + {int(dta_n_int)} interview(s) × "
        f"{len(dta_groups)} construct(s)): " + "  ·  ".join(est_dta)
    )
    st.info(
        f"💰 **Estimated cost — Run LO Analysis** "
        f"({n_dta_total} text(s) × {len(_DTA_LO)} learning objective(s)): "
        + "  ·  ".join(est_lo)
    )

    if n_dta_total == 0:
        st.warning("Set at least one reflection or interview above.")
        return

    # ── Review & Edit Prompts ─────────────────────────────────────────
    with st.expander("🔍 Review & edit prompts (optional)", expanded=False):
        st.caption(
            "These are the exact instructions sent to the AI for each phase. "
            "They are pre-filled with research-validated defaults. "
            "Edit to adjust for your cohort's language, age, or research focus. "
            "**Save** applies your version to the next run. "
            "**Restore** resets to the default."
        )

        def _dta_prompt_widget(label, help_text, sess_key, saved_key, reset_flag,
                               default, height=160):
            modified = st.session_state.get(sess_key, default) != st.session_state.get(saved_key, default)
            st.markdown(f"**{label}**")
            st.caption(help_text)
            if modified:
                st.info("✏️ Unsaved changes — click Save to apply.")
            st.text_area(label, key=sess_key, height=height, label_visibility="collapsed")
            c1, c2, _ = st.columns([1, 1.4, 4])
            if c1.button("💾 Save", key=f"dta_save_{sess_key}"):
                st.session_state[saved_key] = st.session_state[sess_key]
                st.success("Saved.")
            if c2.button("↺ Restore", key=f"dta_restore_{sess_key}"):
                st.session_state[reset_flag] = True
                st.rerun()
            st.divider()

        _dta_prompt_widget(
            "System prompt",
            "Defines the AI's role and analysis approach. Injected into every DTA phase call.",
            "dta_sys_prompt", "dta_sys_saved", "_dta_reset_sys", DTA_SYSTEM_PROMPT, height=150,
        )
        _dta_prompt_widget(
            "Phase 2 prompt (construct coding)",
            "Sent once per participant × construct. "
            "Must keep `{participant_id}`, `{construct_name}`, `{definition}`, "
            "`{analytic_focus}`, `{indicators}`, `{text}`.",
            "dta_p2_prompt", "dta_p2_saved", "_dta_reset_p2", _DTA_PHASE2_PROMPT, height=280,
        )
        _dta_prompt_widget(
            "Learning Objectives prompt",
            "Sent once per participant × module × LO. "
            "Must keep `{participant_id}`, `{module_id}`, `{module_title}`, "
            "`{lo_index}`, `{lo_text}`, `{indicators}`, `{text}`.",
            "dta_lo_prompt", "dta_lo_saved", "_dta_reset_lo", _DTA_LO_PROMPT, height=240,
        )
        _dta_prompt_widget(
            "Phase 5 prompt (narrative summary)",
            "Sent once per run to generate the academic summary. Must keep `{table_text}`.",
            "dta_p5_prompt", "dta_p5_saved", "_dta_reset_p5", _DTA_PHASE5_PROMPT, height=180,
        )

    col_r, col_lo = st.columns(2)
    run_dta = col_r.button("▶ Run DTA",         key="dta_run_btn", type="primary")
    run_lo  = col_lo.button("▶ Run LO Analysis", key="lo_run_btn")

    if run_dta:
        _execute_dta_run(
            username=username, sources=sources, per_run_files=per_run_files,
            models=dta_models, temperature=dta_temp,
            construct_groups=dta_groups,
            show_stream=st.session_state.get("dta_show_stream", False),
            max_reflections=int(dta_n_ref),
            max_interviews=int(dta_n_int),
            custom_sys_prompt=st.session_state.get("dta_sys_saved"),
            custom_p2_prompt=st.session_state.get("dta_p2_saved"),
            custom_p5_prompt=st.session_state.get("dta_p5_saved"),
            cohort_ids=dta_cohort_ids,
        )
    if run_lo:
        _execute_lo_run(
            username=username, models=dta_models, temperature=dta_temp,
            max_reflections=int(dta_n_ref), max_interviews=int(dta_n_int),
            custom_sys_prompt=st.session_state.get("dta_sys_saved"),
            custom_lo_prompt=st.session_state.get("dta_lo_saved"),
        )


def _execute_dta_run(username, sources, per_run_files,
                     models, temperature, construct_groups,
                     show_stream=False,
                     max_reflections=None,
                     max_interviews=None,
                     custom_sys_prompt=None,
                     custom_p2_prompt=None,
                     custom_p5_prompt=None,
                     cohort_ids=None):
    src_label = " + ".join(sources)
    with st.spinner(f"Loading data from: {src_label}..."):
        try:
            transcripts = _load_combined_transcripts(sources, per_run_files, cohort_ids=cohort_ids)
        except Exception as e:
            st.error(f"Could not load data: {e}"); return

    if not transcripts:
        st.warning(f"No transcript data found in: {src_label}."); return

    # Per-type subsetting
    if max_reflections is not None or max_interviews is not None:
        refs   = [t for t in transcripts if "reflections" in t.get("source_types", [])]
        ints   = [t for t in transcripts if "interview"   in t.get("source_types", [])]
        others = [t for t in transcripts
                  if "reflections" not in t.get("source_types", [])
                  and "interview"  not in t.get("source_types", [])]
        if max_reflections is not None:
            refs = refs[:max_reflections]
        if max_interviews is not None:
            ints = ints[:max_interviews]
        transcripts = refs + ints + others
        st.caption(
            f"Using {len(refs)} reflection(s) + {len(ints)} interview(s)"
            + (f" + {len(others)} other(s)" if others else "") + "."
        )

    constructs_to_run = {k: v for k, v in _DTA_CODEBOOK.items()
                         if v["group"] in construct_groups}
    st.info(f"Loaded {len(transcripts)} participant(s). "
            f"Analysing {len(constructs_to_run)} constructs × {len(models)} model(s).")
    st.caption(f"Estimated API calls: "
               f"{len(transcripts) * len(constructs_to_run) * len(models)}")

    import json as _j
    from datetime import datetime as _dt

    import core.analytics.llm.dta_pipeline as _dta_mod
    if custom_sys_prompt and custom_sys_prompt != DTA_SYSTEM_PROMPT:
        _dta_mod.DTA_SYSTEM_PROMPT = custom_sys_prompt
    if custom_p2_prompt and custom_p2_prompt != _DTA_PHASE2_PROMPT:
        _dta_mod._DTA_PHASE2_PROMPT = custom_p2_prompt
    if custom_p5_prompt and custom_p5_prompt != _DTA_PHASE5_PROMPT:
        _dta_mod._DTA_PHASE5_PROMPT = custom_p5_prompt
    sys_p = custom_sys_prompt or DTA_SYSTEM_PROMPT

    for model in models:
        st.markdown(f"---\n#### {_llm_display_name(model)}")
        run_id = _dta_create_run(
            model=model, temperature=temperature,
            source_type="+".join(sources),
            construct_groups=construct_groups,
            created_by=username,
            cohort_scope=", ".join(cohort_ids) if cohort_ids else "All cohorts",
        )
        st.caption(f"Run ID: `{run_id[:8]}...`")

        n_total  = len(transcripts) * len(constructs_to_run)
        progress = st.progress(0, text="Phase 1 — Chunking...")

        # Phase 1
        with st.status("Phase 1 — Preparing transcripts...",
                       expanded=False) as s1:
            total_chars = sum(len(t.get("content","")) for t in transcripts)
            s1.update(
                label=f"Phase 1 ✅ — {len(transcripts)} participant(s) "
                      f"({total_chars:,} characters)",
                state="complete",
            )

        # Phase 2 — per-construct streaming
        stream_area = st.container() if show_stream else None
        p2_results  = []
        step        = 0
        now_s       = _dt.utcnow().isoformat()

        for transcript in transcripts:
            pid  = transcript.get("participant_id", "unknown")
            text = str(transcript.get("content", "")).strip()
            if not text: continue

            for cname, cdef in constructs_to_run.items():
                ind_block = "\n".join(('  - "' + i + '"') for i in cdef["indicators"])
                prompt = _DTA_PHASE2_PROMPT.format(
                    construct_name=cname.replace("_"," ").title(),
                    definition=cdef["definition"],
                    analytic_focus=", ".join(cdef["analytic_focus"]),
                    indicators=ind_block,
                    text=text[:3000],
                )
                response = _llm_call_model(
                    model, prompt, system=sys_p,
                    temperature=temperature, max_tokens=1500,
                )
                raw_resp = response.get("text", "")
                result = {
                    "participant_id":     pid,
                    "construct_name":     cname,
                    "construct_group":    cdef["group"],
                    "group_label":        cdef["group_label"],
                    "model":              model,
                    "temperature":        temperature,
                    "created_at":         now_s,
                    "error":              response.get("error"),
                    "evidence_count":     0,
                    "valence_positive":   0,
                    "valence_negative":   0,
                    "valence_neutral":    0,
                    "instances":          [],
                    "raw_prompt":         prompt,
                    "raw_response":       raw_resp,
                    "matched_indicators": [],
                }
                if not response.get("error"):
                    parsed = _parse_dta_json(raw_resp)
                    if parsed:
                        insts = parsed.get("instances", [])
                        result["evidence_count"]    = parsed.get("evidence_count", len(insts))
                        result["instances"]         = insts
                        result["valence_positive"]  = sum(1 for i in insts if i.get("valence")=="positive")
                        result["valence_negative"]  = sum(1 for i in insts if i.get("valence")=="negative")
                        result["valence_neutral"]   = sum(1 for i in insts if i.get("valence")=="neutral")
                        result["matched_indicators"] = _detect_matched_indicators(
                            text, cdef["indicators"], insts
                        )
                    else:
                        result["error"] = f"Parse failed: {raw_resp[:60]}"

                p2_results.append(result)
                step += 1
                ev   = result["evidence_count"]
                icon = "🟢" if ev > 0 else "⚪"
                progress.progress(
                    min(step / max(n_total, 1), 1.0),
                    text=f"Phase 2 — {pid} / "
                         f"{cname.replace('_',' ')} {icon} {ev}"
                )
                if show_stream and stream_area:
                    matched = result.get("matched_indicators", [])
                    with stream_area.expander(
                        f"{icon} **{pid}** / "
                        f"**{cname.replace('_',' ').title()}** — {ev} instance(s)",
                        expanded=False,
                    ):
                        tp, tr, ti = st.tabs(["Prompt","Raw Response","Matched Indicators"])
                        with tp: st.code(prompt, language=None)
                        with tr: st.code(raw_resp or "(no response)", language=None)
                        with ti:
                            if matched:
                                for m in matched: st.caption(f"matched: {m}")
                            else:
                                st.caption("No indicator phrases directly matched.")

        progress.progress(1.0, text="Phase 2 complete")
        save_dta_results(run_id, p2_results)
        n_ev = sum(r.get("evidence_count",0) for r in p2_results)
        st.caption(f"Phase 2 — {len(p2_results)} cells coded, {n_ev} evidence instances.")

        # Phase 3
        with st.status("Phase 3 — Building evidence matrix...", expanded=False) as s3:
            p3 = run_dta_phase3(p2_results)
            s3.update(label="Phase 3 ✅ — Evidence matrix built", state="complete")

        # Phase 4 — re-examine zero-evidence at T=1.0
        with st.status("Phase 4 — Reviewing zero-evidence constructs (T=1.0)...",
                       expanded=False) as s4:
            zero_keys = set(r["construct_name"] for r in p2_results
                            if r.get("evidence_count",0)==0 and not r.get("error"))
            if not zero_keys:
                s4.update(label="Phase 4 ✅ — All constructs had evidence; no review needed",
                          state="complete")
            else:
                p4_index = {}
                for transcript in transcripts:
                    pid  = transcript.get("participant_id","unknown")
                    text = str(transcript.get("content","")).strip()
                    if not text: continue
                    for cname, cdef in constructs_to_run.items():
                        if cname not in zero_keys: continue
                        ind_block = "\n".join(('  - "'+i+'"') for i in cdef["indicators"])
                        prompt4 = _DTA_PHASE2_PROMPT.format(
                            construct_name=cname.replace("_"," ").title(),
                            definition=cdef["definition"],
                            analytic_focus=", ".join(cdef["analytic_focus"]),
                            indicators=ind_block, text=text[:3000],
                        )
                        r4 = _llm_call_model(model, prompt4, system=sys_p,
                                             temperature=1.0, max_tokens=1500)
                        raw4 = r4.get("text","")
                        parsed4 = _parse_dta_json(raw4) if not r4.get("error") else None
                        if parsed4:
                            insts4 = parsed4.get("instances",[])
                            if insts4:
                                p4_index[(pid, cname)] = {
                                    "evidence_count":   parsed4.get("evidence_count", len(insts4)),
                                    "instances":        insts4,
                                    "valence_positive": sum(1 for i in insts4 if i.get("valence")=="positive"),
                                    "valence_negative": sum(1 for i in insts4 if i.get("valence")=="negative"),
                                    "valence_neutral":  sum(1 for i in insts4 if i.get("valence")=="neutral"),
                                    "raw_prompt":       prompt4,
                                    "raw_response":     raw4,
                                    "phase4_recovery":  True,
                                }

                recovered = 0
                for i, r in enumerate(p2_results):
                    key = (r["participant_id"], r["construct_name"])
                    if key in p4_index:
                        p2_results[i] = {**r, **p4_index[key]}
                        recovered += 1

                save_dta_results(run_id, p2_results)
                p3 = run_dta_phase3(p2_results)  # rebuild with recoveries
                s4.update(
                    label=f"Phase 4 ✅ — {len(zero_keys)} reviewed, "
                          f"{recovered} recovered at T=1.0",
                    state="complete",
                )

        # Phase 5
        with st.status("Phase 5 — Generating narrative summary...", expanded=False) as s5:
            p5 = run_dta_phase5(p3, p2_results, model=model,
                                temperature=temperature, run_id=run_id)
            if p5.get("error"):
                s5.update(label=f"Phase 5 ❌ — {p5['error']}", state="error")
            else:
                s5.update(label="Phase 5 ✅ — Report generated", state="complete")

        st.session_state["dta_last_run_id"] = run_id
        st.success("DTA complete — view results in the Results tab.")


def _execute_lo_run(username, models, temperature,
                    max_reflections=None, max_interviews=None,
                    custom_sys_prompt=None,
                    custom_lo_prompt=None):
    import sqlite3 as _sq3, re as _re3
    from pathlib import Path as _P3
    #db3 = next((p/"responses.db" for p in _P3(__file__).resolve().parents #fix is to replace both next(...) path searches in this function with a direct env var lookup.
    #            if (p/"responses.db").exists()), None)
    db3 = _get_responses_db_path()
    if not db3: st.error("responses.db not found."); return
    conn3 = _sq3.connect(db3)
    rows3 = conn3.execute(
        "SELECT user_id, instrument_name, response_value FROM responses "
        "WHERE instrument_name LIKE '%module_reflections%' "
        "AND response_value IS NOT NULL"
    ).fetchall()
    conn3.close()
    lo_trans = []
    for uid, iname, rval in rows3:
        if not str(rval).strip(): continue
        m = _re3.match(r"module_?(\d+)", iname, _re3.I)
        if m:
            lo_trans.append({"participant_id": uid,
                             "module_id": f"module_{m.group(1)}",
                             "content": str(rval)})
    if not lo_trans: st.warning("No module reflection notes found."); return

    # Apply limit — LO analysis uses reflections only
    if max_reflections is not None and max_reflections < len(lo_trans):
        lo_trans = lo_trans[:max_reflections]
        st.caption(f"Using first {max_reflections} participant-module reflection(s).")
    if not lo_trans: st.warning("No module reflection notes found."); return
    st.info(f"{len(lo_trans)} participant-module combinations.")
    for model in models:
        st.markdown(f"---\n#### {_llm_display_name(model)}")
        run_id = _dta_create_run(model=model, temperature=temperature,
                                 source_type="lo_analysis",
                                 construct_groups=["learning_objectives"],
                                 created_by=username)
        with st.status(f"Running LO analysis ({len(lo_trans)} participant-modules)...",
                       expanded=False) as slo:
            lo_results = run_lo_analysis(lo_trans, model, temperature, run_id=run_id)
            achieved   = sum(r.get("evidence_present",0) for r in lo_results)
            slo.update(label=f"LO Analysis ✅ — {achieved}/{len(lo_results)} objectives with evidence",
                       state="complete")
        st.session_state["dta_lo_last_run_id"] = run_id
        st.success("LO analysis complete — view in Learning Objectives tab.")


# -----------------------------------------------------------------------
# DTA Results panel
# -----------------------------------------------------------------------

def _render_dta_results_panel(username: str) -> None:
    runs = [r for r in _dta_list_runs(created_by=username)
            if r.get("construct_groups","[]") != '["learning_objectives"]']
    if not runs:
        st.info("No DTA runs found. Run the analysis first."); return

    run_opts = {
        f"{r['model'].upper()} T={r['temperature']} — "
        f"{r['created_at'][:16]} [{r['status']}]": r["run_id"]
        for r in runs[:10]
    }
    sel   = st.selectbox("Select run", list(run_opts.keys()), key="dta_results_run")
    run_id = run_opts[sel]
    run_rec = next(r for r in runs if r["run_id"] == run_id)
    st.caption(f"Model: **{_llm_display_name(run_rec['model'])}** | "
               f"T={run_rec['temperature']} | Source: {run_rec.get('source_type','')}")
    st.divider()

    df = load_dta_results(run_id)
    if df.empty: st.info("No results for this run."); return

    view = st.radio("View as",
                    ["Construct Table", "Evidence Heatmap", "Participant Profiles"],
                    horizontal=True, key="dta_result_view")

    if view == "Construct Table":
        _render_dta_construct_table(df)
    elif view == "Evidence Heatmap":
        _render_dta_heatmap(df)
    else:
        _render_dta_participant_profiles(df)

    # Audit trail
    st.divider()
    import json as _aj, io as _aio
    with st.expander("📥 Download audit trail", expanded=False):
        audit = []
        for _, row in df.iterrows():
            matched = row.get("matched_indicators", [])
            if isinstance(matched, str):
                import json as _jj
                try: matched = _jj.loads(matched)
                except: matched = []
            audit.append({
                "participant_id":     row["participant_id"],
                "construct_name":     row["construct_name"],
                "evidence_count":     int(row.get("evidence_count",0)),
                "matched_indicators": matched,
                "instances":          row.get("instances",[]),
                "raw_prompt":         row.get("raw_prompt",""),
                "raw_response":       row.get("raw_response",""),
            })
        import json as _ajj
        col_j, col_c = st.columns(2)
        col_j.download_button("📄 Full audit (.json)",
                              data=_ajj.dumps(audit, indent=2, ensure_ascii=False),
                              file_name=f"dta_audit_{run_id[:8]}.json",
                              mime="application/json", key="dta_audit_dl")
        import pandas as _apd
        col_c.download_button("📊 Summary (.csv)",
                              data=_apd.DataFrame([{
                                  "participant": r["participant_id"],
                                  "construct":   r["construct_name"],
                                  "evidence":    r["evidence_count"],
                                  "n_matched":   len(r["matched_indicators"]),
                              } for r in audit]).to_csv(index=False),
                              file_name=f"dta_summary_{run_id[:8]}.csv",
                              mime="text/csv", key="dta_summary_dl")


def _render_dta_construct_table(df: pd.DataFrame) -> None:
    summary = df.groupby(["construct_group","construct_name"]).agg(
        total_evidence=("evidence_count","sum"),
        n_with_evidence=("evidence_count", lambda x: (x>0).sum()),
        positive=("valence_positive","sum"),
        negative=("valence_negative","sum"),
        neutral=("valence_neutral","sum"),
    ).reset_index()
    for gk in summary["construct_group"].unique():
        gl = _DTA_GROUPS.get(gk, gk)
        st.markdown(f"**{gl}**")
        grp = summary[summary["construct_group"]==gk].drop(
            columns=["construct_group"]
        ).copy()
        grp["construct_name"] = grp["construct_name"].str.replace("_"," ").str.title()
        grp.columns = ["Construct","Total Evidence","Participants",
                       "Positive","Negative","Neutral"]
        st.dataframe(grp, hide_index=True, width="stretch")
        st.divider()


def _render_dta_heatmap(df: pd.DataFrame) -> None:
    if not _HAS_PLOTLY:
        st.info("Install plotly for heatmap."); return
    import plotly.express as px
    pivot = df.pivot_table(index="participant_id", columns="construct_name",
                           values="evidence_count", aggfunc="sum", fill_value=0)
    pivot.columns = [c.replace("_"," ").title() for c in pivot.columns]
    fig = px.imshow(pivot, title="Evidence Count Heatmap",
                    color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(height=max(300, len(pivot)*30+100),
                      margin=dict(t=50,b=10,l=10,r=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, width='stretch')


def _render_dta_participant_profiles(df: pd.DataFrame) -> None:
    import json as _pj
    participants = sorted(df["participant_id"].unique())
    pid  = st.selectbox("Select participant", participants, key="dta_profile_pid")
    pdata = df[df["participant_id"]==pid]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total evidence",         int(pdata["evidence_count"].sum()))
    c2.metric("Constructs with evidence",int((pdata["evidence_count"]>0).sum()))
    c3.metric("Positive",               int(pdata["valence_positive"].sum()))
    c4.metric("Negative",               int(pdata["valence_negative"].sum()))

    show_t = st.checkbox("Show prompt and raw response",
                         value=False, key="dta_profile_transparency")
    st.divider()

    for gk in pdata["construct_group"].unique():
        st.markdown(f"**{_DTA_GROUPS.get(gk, gk)}**")
        for _, row in pdata[pdata["construct_group"]==gk].iterrows():
            cname = row["construct_name"].replace("_"," ").title()
            ev    = int(row["evidence_count"])
            p4tag = " [P4]" if row.get("phase4_recovery") else ""
            icon  = "🟢" if ev > 0 else "⚪"
            matched = row.get("matched_indicators",[])
            if isinstance(matched, str):
                try: matched = _pj.loads(matched)
                except: matched = []

            with st.expander(f"{icon} {cname}{p4tag} — {ev} instance(s)",
                             expanded=False):
                insts = row.get("instances",[])
                if isinstance(insts, str):
                    try: insts = _pj.loads(insts)
                    except: insts = []
                if insts:
                    for inst in insts:
                        v = inst.get("valence","neutral")
                        vlbl = {"positive":"Positive","negative":"Negative",
                                "neutral":"Neutral"}.get(v, v)
                        st.markdown(f"**[{vlbl}]** {inst.get('quote','')}")
                        if inst.get("explanation"):
                            st.caption(f"Explanation: {inst['explanation']}")
                else:
                    st.caption("No evidence found.")
                if matched:
                    st.markdown("**Matched indicator phrases:**")
                    for m in matched: st.caption(f"  matched: {m}")
                if show_t:
                    tp, tr = st.tabs(["Prompt sent","Raw response"])
                    with tp: st.code(row.get("raw_prompt","(not stored)"), language=None)
                    with tr: st.code(row.get("raw_response","(not stored)"), language=None)


# -----------------------------------------------------------------------
# DTA — Learning Objectives panel
# -----------------------------------------------------------------------

def _render_dta_lo_panel(username: str) -> None:
    lo_runs = [r for r in _dta_list_runs(created_by=username)
               if r.get("construct_groups","") == '["learning_objectives"]']
    if not lo_runs:
        st.info("No LO analysis found. Go to Run Analysis and click Run LO Analysis.")
        return

    opts = {f"{r['model'].upper()} T={r['temperature']} — {r['created_at'][:16]}":
            r["run_id"] for r in lo_runs[:5]}
    sel    = st.selectbox("Select LO run", list(opts.keys()), key="dta_lo_run")
    run_id = opts[sel]

    try:
        import sqlite3 as _sq4
        from pathlib import Path as _P4
        #db4 = next((p/"responses.db" for p in _P4(__file__).resolve().parents #fix is to replace both next(...) path searches in this function with a direct env var lookup. 
        #            if (p/"responses.db").exists()), None)
        db4 = _get_responses_db_path()
        if not db4: st.error("DB not found."); return
        conn4 = _sq4.connect(db4)
        rows4 = conn4.execute(
            "SELECT participant_id, module_id, lo_index, lo_text, "
            "evidence_present, evidence_quote FROM dta_lo_results "
            "WHERE run_id=? ORDER BY participant_id, module_id, lo_index",
            (run_id,)
        ).fetchall()
        conn4.close()
    except Exception as e:
        st.error(f"Could not load LO results: {e}"); return

    if not rows4: st.info("No LO results stored."); return

    lo_df = pd.DataFrame(rows4, columns=[
        "participant_id","module_id","lo_index",
        "lo_text","evidence_present","evidence_quote"
    ])

    lo_view = st.radio("View", ["By Module","By Participant"],
                       horizontal=True, key="dta_lo_view")
    st.divider()

    if lo_view == "By Module":
        mods = sorted(lo_df["module_id"].unique())
        sel_mod = st.selectbox("Module", mods, key="dta_lo_mod",
                               format_func=lambda m:
                               f"{m.replace('_',' ').title()} — "
                               f"{_DTA_LO.get(m,{}).get('title','')}")
        mdf = lo_df[lo_df["module_id"]==sel_mod]
        if _HAS_PLOTLY:
            import plotly.express as px
            pivot = mdf.pivot_table(index="participant_id", columns="lo_index",
                                    values="evidence_present", aggfunc="max",
                                    fill_value=0)
            pivot.columns = [f"LO {c}" for c in pivot.columns]
            fig = px.imshow(pivot, title=f"LO Achievement — {sel_mod}",
                            color_continuous_scale=[[0,"white"],[1,"#2ECC71"]],
                            aspect="auto")
            fig.update_layout(height=300, margin=dict(t=50,b=10))
            st.plotly_chart(fig, width='stretch')
        for lo_idx in sorted(mdf["lo_index"].unique()):
            lrow = mdf[mdf["lo_index"]==lo_idx]
            lo_text = lrow.iloc[0]["lo_text"]
            n_ach   = int(lrow["evidence_present"].sum())
            with st.expander(f"LO {lo_idx}: {lo_text[:70]}... — {n_ach}/{len(lrow)} achieved",
                             expanded=False):
                for _, r in lrow.iterrows():
                    icon = "✅" if r["evidence_present"] else "⚪"
                    st.markdown(f"{icon} **{r['participant_id']}**")
                    if r["evidence_quote"]:
                        st.caption(f'"{r["evidence_quote"]}"')
    else:
        pids = sorted(lo_df["participant_id"].unique())
        sel_pid = st.selectbox("Participant", pids, key="dta_lo_pid")
        pdf = lo_df[lo_df["participant_id"]==sel_pid]
        total_ach = int(pdf["evidence_present"].sum())
        pct = round(total_ach/max(len(pdf),1)*100)
        c1,c2,c3 = st.columns(3)
        c1.metric("LOs with evidence", f"{total_ach}/{len(pdf)}")
        c2.metric("Achievement rate",  f"{pct}%")
        c3.metric("Modules covered",   pdf["module_id"].nunique())
        for mid in sorted(pdf["module_id"].unique()):
            mdf2   = pdf[pdf["module_id"]==mid]
            mtitle = _DTA_LO.get(mid,{}).get("title",mid)
            n_mod  = int(mdf2["evidence_present"].sum())
            with st.expander(f"{mid.replace('_',' ').title()} — {mtitle} "
                             f"({n_mod}/{len(mdf2)} LOs)", expanded=False):
                for _, r in mdf2.iterrows():
                    icon = "✅" if r["evidence_present"] else "⚪"
                    st.markdown(f"{icon} LO {r['lo_index']}: {r['lo_text'][:80]}...")
                    if r["evidence_quote"]:
                        st.caption(f'"{r["evidence_quote"]}"')


# =======================================================================
# REPORT GENERATION TAB
# =======================================================================

def _render_report_tab(
    username: str,
    canonical_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    cohort_map: dict,
) -> None:
    """Tab 5 — Report Generation."""

    st.subheader("📄 Report Generation")
    st.caption(
        "Generate PDF reports for each analysis section. "
        "Each report captures the current filtered dataset. "
        "Run the relevant analysis first, then come here to download."
    )

    # ── Section selector ─────────────────────────────────────────────────────
    if "report_section_radio" not in st.session_state:
        st.session_state["report_section_radio"] = "i. Basic Statistics"

    rep_section = st.radio(
        "**Report section:**",
        options=[
            "i. Basic Statistics",
            "ii. Inferential Statistics",
            "iii. IRT Analysis",
            "iv. LLM Analysis",
            "v. Competency Progression",
            "vi. Full Programme Report",
            "vii. Instruments & References",
        ],
        horizontal=True,
        key="report_section_radio",
    )
    _RS_COLORS = {
        "i. Basic Statistics":       ("#0077BB", "#E6F3FB"),
        "ii. Inferential Statistics":("#EE7733", "#FFF3E6"),
        "iii. IRT Analysis":         ("#009E73", "#E6F7F1"),
        "iv. LLM Analysis":          ("#CC79A7", "#F9EEF5"),
        "v. Competency Progression": ("#888888", "#F0F0F0"),
        "vi. Full Programme Report":       ("#333333", "#F5F5F5"),
        "vii. Instruments & References":   ("#56B4E9", "#E8F4FB"),
    }
    _rc, _rbg = _RS_COLORS.get(rep_section, ("#333","#F8F8F8"))
    st.markdown(
        f"<div style='background:{_rbg};border-left:5px solid {_rc};"
        f"border-radius:6px;padding:0.4rem 1rem;margin:0.3rem 0;'>"
        f"<strong style='color:{_rc};'>▶ {rep_section}</strong></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    if rep_section == "i. Basic Statistics":
        _report_basic_stats(canonical_df, demographics_df, cohort_map)

    elif rep_section == "ii. Inferential Statistics":
        _report_inferential(canonical_df)

    elif rep_section == "iii. IRT Analysis":
        _report_irt(canonical_df)

    elif rep_section == "iv. LLM Analysis":
        _report_llm()

    elif rep_section == "v. Competency Progression":
        _report_cpi(canonical_df)

    elif rep_section == "vi. Full Programme Report":
        _report_full_programme(
            canonical_df, demographics_df, cohort_map, username
        )

    elif rep_section == "vii. Instruments & References":
        _report_instruments_references()



# -----------------------------------------------------------------------
# vii. Instruments & References
# -----------------------------------------------------------------------

def _report_instruments_references() -> None:
    """Instruments used and full references for Basics4AI scales."""
    st.markdown("### vii. Instruments & References")
    st.caption(
        "Full citations for all adapted instruments used in the Basics4AI "
        "programme evaluation. Cite these sources when reporting survey results."
    )

    # ── Instruments overview ─────────────────────────────────────────
    with st.expander("📋 Instruments used in this study", expanded=True):
        st.markdown("""
**SCCCES — Situational Conceptual Change Cognitive Engagement Scale**
Measures learners' cognitive engagement with instructional messaging across
seven constructs: Engagement with Task, Effort & Persistence, Experience of Flow,
Coherency, Plausibility, Credibility, and Comprehensibility of Messaging,
Attention, Culture, and Personal Relevance. Likert scale 1–4.
Items Q9_1, Q9_2, Q10_1, Q10_2 are reverse-scored (5−x) before computing construct means.

---

**SIMS — Situational Motivation Scale**
Measures motivation type in four constructs: Intrinsic Motivation, Identified
Regulation, External Regulation, and Amotivation. Likert scale 1–4.
Items Q4_1–Q4_4 (External Regulation) and Q5_1–Q5_3 (Amotivation) are
reverse-scored so that all four constructs read in the same direction:
**higher mean = better self-determined motivation**.

---

**AIM-F — AI Misconceptions Framework (adapted)**
8-item True/False and Yes/No assessment measuring common misconceptions
about AI systems (learning autonomy, emotions, anthropomorphism, etc.).
Source: Basics4AI programme instrument.

---

**AI-CI — AI Conceptual Inventory (20-item)**
MCQ assessment of AI conceptual understanding across classification,
decision trees, supervised/unsupervised learning, bias, and prediction.
See Appendix 1 of the published instrument for full item listings.
        """)

    st.divider()

    # ── References ───────────────────────────────────────────────────
    st.markdown("#### References")

    _REFS = [
        {
            "tag": "SIMS",
            "color": "#E6F3FB",
            "border": "#0077BB",
            "citation": (
                "Guay, F., Vallerand, R. J., & Blanchard, C. (2000). "
                "On the Assessment of Situational Intrinsic and Extrinsic Motivation: "
                "The Situational Motivation Scale (SIMS). "
                "*Motivation and Emotion, 24*(3), 175–213. "
                "https://doi.org/10.1023/a:1005614228250"
            ),
            "note": "Adapted for Basics4AI: External Regulation and Amotivation items reverse-scored.",
        },
        {
            "tag": "SCES",
            "color": "#E6F7F1",
            "border": "#009E73",
            "citation": (
                "Rotgans, J. I., & Schmidt, H. G. (2011). "
                "Cognitive engagement in the problem-based learning classroom. "
                "*Advances in Health Sciences Education, 16*(4), 465–479. "
                "https://doi.org/10.1007/s10459-011-9272-9"
            ),
            "note": "Engagement with Task, Effort & Persistence, and Experience of Flow subscales.",
        },
        {
            "tag": "CCCES",
            "color": "#F9EEF5",
            "border": "#CC79A7",
            "citation": (
                "Heddy, B. C., Taasoobshirazi, G., Chancey, J. B., & Danielson, R. W. (2018). "
                "Developing and Validating a Conceptual Change Cognitive Engagement Instrument. "
                "*Frontiers in Education, 3*(43), 1–9. "
                "https://doi.org/10.3389/feduc.2018.00043"
            ),
            "note": (
                "Coherency, Plausibility, Credibility, Comprehensibility of Messaging, "
                "Attention, Culture, and Personal Relevance subscales."
            ),
        },
        {
            "tag": "IRT",
            "color": "#E6F7F1",
            "border": "#009E73",
            "citation": (
                "De Ayala, R. J. (2009). "
                "*The Theory and Practice of Item Response Theory*. "
                "Guilford Press."
            ),
            "note": "Rasch (1PL), 2PL, and GRM models used in the IRT Analysis tab.",
        },
        {
            "tag": "ITA",
            "color": "#FFF3E6",
            "border": "#EE7733",
            "citation": (
                "Braun, V., & Clarke, V. (2006). "
                "Using thematic analysis in psychology. "
                "*Qualitative Research in Psychology, 3*(2), 77–101. "
                "https://doi.org/10.1191/1478088706qp063oa"
            ),
            "note": "Inductive Thematic Analysis (ITA) methodology used in the LLM Analysis tab.",
        },
        {
            "tag": "De Paoli",
            "color": "#FFF3E6",
            "border": "#EE7733",
            "citation": (
                "De Paoli, S. (2024). "
                "Performing an inductive thematic analysis of semi-structured interviews "
                "with a Large Language Model. "
                "*Applied AI Letters, 5*(1), e129. "
                "https://doi.org/10.1002/ail2.129"
            ),
            "note": "LLM-assisted ITA pipeline methodology (Phases 1–6) used in the LLM Analysis tab.",
        },
        {
            "tag": "SDT",
            "color": "#E6F3FB",
            "border": "#0077BB",
            "citation": (
                "Deci, E. L., & Ryan, R. M. (1985). "
                "*Intrinsic Motivation and Self-Determination in Human Behavior*. "
                "Plenum. https://doi.org/10.1007/978-1-4899-2271-7"
            ),
            "note": "Self-Determination Theory underpinning SIMS construct interpretation.",
        },
    ]

    for ref in _REFS:
        _r_color  = ref["color"]
        _r_border = ref["border"]
        _r_tag    = ref["tag"]
        _r_cite   = ref["citation"]
        _r_note   = ref["note"]
        st.markdown(
            f"<div style='background:{_r_color};border-left:4px solid "
            f"{_r_border};border-radius:6px;padding:0.5rem 1rem;"
            f"margin:0.5rem 0;'>"
            f"<span style='font-size:11px;font-weight:500;color:{_r_border}'>"
            f"[{_r_tag}]</span><br>"
            f"{_r_cite}<br>"
            f"<span style='font-size:12px;color:#555;font-style:italic;'>"
            f"↳ {_r_note}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "To export this reference list to PDF, use the Full Programme Report "
        "generator in section vi."
    )

# -----------------------------------------------------------------------
# PDF builder — shared utility
# -----------------------------------------------------------------------

def _build_pdf(sections: list, title: str) -> bytes:
    """
    Build a PDF report from a list of content sections.

    Parameters
    ----------
    sections : list of dicts, each with keys:
        heading : str
        body    : str (plain text, may include \n)
        table   : pd.DataFrame or None
        caption : str or None
    title : str  — document title (header)

    Returns bytes (PDF content)
    """
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, PageBreak,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18, spaceAfter=6,
        textColor=colors.HexColor("#0077BB"),
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=13, spaceBefore=12, spaceAfter=4,
        textColor=colors.HexColor("#0077BB"),
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, spaceAfter=4, leading=13,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#555555"),
        spaceAfter=6, leading=11,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=7.5, textColor=colors.HexColor("#888888"),
        spaceBefore=8, leading=10,
    )

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Basics4AI Programme", styles["Heading2"]))
    story.append(Paragraph(title, title_style))
    from datetime import datetime as _dt
    story.append(Paragraph(
        f"Generated: {_dt.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
        caption_style
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#0077BB")))
    story.append(Spacer(1, 0.4*cm))

    # ── Sections ───────────────────────────────────────────────────────────
    for sec in sections:
        heading = sec.get("heading", "")
        body    = sec.get("body", "")
        df      = sec.get("table")
        caption = sec.get("caption")

        if heading:
            story.append(Paragraph(heading, h1_style))

        if body:
            for line in body.split("\n"):
                line = line.strip()
                if line:
                    story.append(Paragraph(line, body_style))

        if df is not None and not df.empty:
            # Convert dataframe to reportlab table.
            #
            # Fixed 2026-08-09: a plain string in a platypus Table cell is
            # NOT auto-wrapped by reportlab -- a well-known gotcha -- so any
            # value wider than its column (a long participant ID, a long
            # description, or Task F's new comma-joined multi-module scope
            # string like "module_1,module_2,...") silently overflowed into
            # the next column or past the page margin instead of wrapping.
            # Equal fixed-width columns made it worse for any table with a
            # naturally-long column. Now every cell is wrapped in a
            # Paragraph (which DOES wrap) and column widths are weighted by
            # each column's average content length instead of split evenly.
            col_headers = list(df.columns)
            data_rows   = df.astype(str).values.tolist()

            page_w = A4[0] - 4*cm
            _avg_lens = [
                max(
                    (len(str(col_headers[i])) + sum(len(row[i]) for row in data_rows))
                    / (1 + len(data_rows)),
                    3,
                )
                for i in range(len(col_headers))
            ]
            _min_w = 1.6 * cm
            col_widths = [max(page_w * (l / sum(_avg_lens)), _min_w) for l in _avg_lens]
            # The min-width floor above can push the total past page_w for
            # tables with many columns -- rescale so it always fits exactly.
            _scale = page_w / sum(col_widths)
            col_widths = [w * _scale for w in col_widths]

            _tbl_header_style = ParagraphStyle(
                "TblHeader", parent=styles["Normal"],
                fontSize=8, leading=10, textColor=colors.white,
                fontName="Helvetica-Bold",
            )
            _tbl_body_style = ParagraphStyle(
                "TblBody", parent=styles["Normal"],
                fontSize=7.5, leading=9,
            )
            tbl_data = (
                [[Paragraph(str(h), _tbl_header_style) for h in col_headers]]
                + [
                    [Paragraph(str(cell), _tbl_body_style) for cell in row]
                    for row in data_rows
                ]
            )

            tbl = Table(tbl_data, colWidths=col_widths, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#0077BB")),
                ("VALIGN",      (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.HexColor("#F7FBFF"), colors.white]),
                ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
                ("TOPPADDING",  (0,0), (-1,-1), 3),
                ("BOTTOMPADDING",(0,0),(-1,-1), 3),
                ("LEFTPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.2*cm))

        if caption:
            story.append(Paragraph(caption, caption_style))

        story.append(Spacer(1, 0.3*cm))

    # ── Footer disclaimer ──────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#CCCCCC")))
    story.append(Paragraph(
        "Instruments: CCCES (Dole & Sinatra, 1998), SIMS (Deci & Ryan, 1985), "
        "AI-CI, AIM-F. "
        "Qualitative analysis: Braun & Clarke (2006); De Paoli (2024); "
        "Bingham (2023). "
        "LLM-assisted thematic analysis is exploratory and does not establish "
        "formal procedures for I-/D-TA with LLMs.",
        disclaimer_style,
    ))

    doc.build(story)
    return buf.getvalue()


# -----------------------------------------------------------------------
# i. Basic Statistics report
# -----------------------------------------------------------------------

def _report_basic_stats(
    canonical_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    cohort_map: dict,
) -> None:
    st.markdown("### i. Basic Statistics Report")
    st.caption("Generates a PDF snapshot of participant summary, assessment scores, and survey construct means.")

    # Seed checkbox defaults so they persist across reruns
    _BS_OPTIONS = ["Participant summary", "Assessment scores", "Survey construct means"]
    bs_selected = st.multiselect(
        "Include sections:",
        options=_BS_OPTIONS,
        default=_BS_OPTIONS,
        key="rpt_bs_multiselect",
    )

    if st.button("📄 Generate Basic Statistics PDF", key="rpt_bs_gen", type="primary"):
        selected = st.session_state.get("rpt_bs_multiselect", _BS_OPTIONS)
        include_participant = "Participant summary"   in selected
        include_assessment  = "Assessment scores"    in selected
        include_survey      = "Survey construct means" in selected

        with st.spinner("Building PDF…"):
            sections = []
            sections.append({
                "heading": "Basic Statistics Report",
                "body":    f"Dataset: {len(demographics_df)} participants | {canonical_df['module_id'].nunique()} modules",
            })

            # Participant summary
            if include_participant:
                from core.analytics.descriptive.descriptive_stats import participant_summary
                summ = participant_summary(demographics_df, cohort_map)
                sections.append({
                    "heading": "1. Participant Summary",
                    "body":    f"Total students: {summ['total']}",
                    "table":   summ["by_grade"].rename(columns={"grade":"Grade","n":"N","pct":"%","grade_level":"Level"})[["Level","Grade","N","%"]],
                    "caption": "Table 1. Distribution of participants by grade level.",
                })
                if summ.get("by_cohort") is not None:
                    sections.append({
                        "heading": "",
                        "table":   summ["by_cohort"].rename(columns={"cohort_id":"Cohort","n":"N","pct":"%"}),
                        "caption": "Table 2. Distribution by cohort.",
                    })

            # Assessment scores
            if include_assessment:
                from core.analytics.descriptive.score_aggregator import (
                    compute_assessment_scores, summarize_scores
                )
                asc = compute_assessment_scores(canonical_df)
                if not asc.empty:
                    for inst_key in sorted(asc["instrument_key"].unique()):
                        label = _ASSESSMENT_LABELS.get(inst_key, inst_key)
                        asc_f = asc[asc["instrument_key"] == inst_key]
                        summ_asc = summarize_scores(asc_f)
                        if not summ_asc.empty:
                            row = summ_asc.iloc[0]
                            sections.append({
                                "heading": f"2. Assessment — {label}",
                                "body":    (
                                    f"N = {int(row.get('n_users',0))}  |  "
                                    f"Mean = {row.get('mean_pct',0):.1f}%  |  "
                                    f"Median = {row.get('median_pct',0):.1f}%"
                                ),
                                "table":   asc_f[["user_id","pct_correct"]].rename(
                                    columns={"user_id":"Student","pct_correct":"% Correct"}
                                ).round(1),
                                "caption": f"Table: Individual scores for {label}.",
                            })

            # Survey construct means
            if include_survey:
                from core.analytics.descriptive.score_aggregator import (
                    compute_construct_means, aggregate_construct_means, summarize_scores
                )
                cm = compute_construct_means(canonical_df)
                if not cm.empty:
                    for base_key in ["b4ai_sccces_survey", "b4ai_sims_survey"]:
                        label = _SURVEY_LABELS.get(base_key, base_key)
                        cm_s = cm[cm["instrument_key"].apply(
                            lambda k: k == base_key or k.endswith("_" + base_key)
                        )]
                        if cm_s.empty:
                            continue
                        agg = aggregate_construct_means(cm_s)
                        summ_cm = summarize_scores(agg)
                        if not summ_cm.empty:
                            disp = summ_cm[["construct","n_users","mean_score","median_score"]].copy()
                            disp.columns = ["Construct","N","Mean","Median"]
                            for c in ["Mean","Median"]:
                                disp[c] = disp[c].apply(lambda x: f"{x:.2f}" if x is not None else "—")
                            sections.append({
                                "heading": f"3. Survey — {label}",
                                "body":    "Aggregate means across all modules (Likert 1–4 scale).",
                                "table":   disp,
                                "caption": f"Table: Construct means for {label}. Scale: 1=Strongly disagree, 4=Strongly agree.",
                            })

            pdf_bytes = _build_pdf(sections, "Basic Statistics Report")
            st.download_button(
                "⬇️ Download Basic Statistics Report (PDF)",
                data=pdf_bytes,
                file_name="b4ai_basic_statistics.pdf",
                mime="application/pdf",
                key="rpt_bs_dl",
            )
            st.success("PDF ready — click the button above to download.")


# -----------------------------------------------------------------------
# ii. Inferential Statistics report
# -----------------------------------------------------------------------

def _report_inferential(canonical_df: pd.DataFrame) -> None:
    st.markdown("### ii. Inferential Statistics Report")
    st.caption("Generates a PDF of all computed inferential tests for the current filtered dataset.")

    _INF_OPTIONS = ["Pre vs Post", "Between Groups", "Across Modules"]
    inf_selected = st.multiselect(
        "Include sections:",
        options=_INF_OPTIONS,
        default=_INF_OPTIONS,
        key="rpt_inf_multiselect",
    )

    if st.button("📄 Generate Inferential Statistics PDF", key="rpt_inf_gen", type="primary"):
        inf_sel     = st.session_state.get("rpt_inf_multiselect", _INF_OPTIONS)
        inc_prepost = "Pre vs Post"     in inf_sel
        inc_between = "Between Groups"  in inf_sel
        inc_across  = "Across Modules"  in inf_sel

        with st.spinner("Running tests and building PDF…"):
            sections = [{"heading": "Inferential Statistics Report", "body": "α = 0.05 for all tests."}]

            # Pre vs Post
            if inc_prepost:
                pairs = [
                    ("AI Misconceptions",    "precourse_pre_ai_misconceptions_assessment",  "postcourse_post_ai_misconceptions_assessment"),
                    ("AI Conceptual Inventory", "precourse_pre_aici_assessment", "postcourse_post_aici_assessment"),
                ]
                for label, pre_k, post_k in pairs:
                    try:
                        r = run_paired_comparison(canonical_df, pre_k, post_k, alpha=0.05)
                        if r.get("error"):
                            continue
                        sections.append({
                            "heading": f"Pre vs Post — {label}",
                            "body":    (
                                f"Pre mean: {r['pre_mean']:.2f}  |  "
                                f"Post mean: {r['post_mean']:.2f}  |  "
                                f"Difference: {r['mean_diff']:+.2f}\n"
                                f"Cohen's d = {r['cohens_d']:.3f} ({r['effect_size_label']})  |  "
                                f"p = {r['t_p_value']:.4f}  |  "
                                f"{'Significant' if r['significant'] else 'Not significant'} at α=0.05"
                            ),
                        })
                    except Exception:
                        pass

            # Between Groups
            if inc_between:
                bg_instruments = [
                    ("AI Misconceptions Pre",  "precourse_pre_ai_misconceptions_assessment"),
                    ("AI Misconceptions Post", "postcourse_post_ai_misconceptions_assessment"),
                ]
                for label, key in bg_instruments:
                    for group_col in ["grade", "gender"]:
                        if group_col not in canonical_df.columns:
                            continue
                        try:
                            r = run_between_groups(canonical_df, key, group_col=group_col, alpha=0.05)
                            if r.get("error"):
                                continue
                            sections.append({
                                "heading": f"Between Groups — {label} by {group_col.title()}",
                                "body":    (
                                    f"F = {r['f_stat']:.3f}  |  "
                                    f"η² = {r['eta_squared']:.4f} ({r['effect_size_label']})  |  "
                                    f"p = {r['anova_p']:.4f}  |  "
                                    f"Kruskal-Wallis p = {r['kruskal_p']:.4f}"
                                ),
                            })
                        except Exception:
                            pass

            # Across Modules
            if inc_across:
                # MCQ
                try:
                    r = run_repeated_measures(canonical_df, "content_mcq_assessment", construct=None)
                    if not r.get("error"):
                        mbt = r.get("means_by_time", {})
                        mbt_df = pd.DataFrame([
                            {"Module": k, "Mean % Correct": round(v, 2)}
                            for k, v in sorted(mbt.items())
                        ]) if mbt else None
                        sections.append({
                            "heading": "Across Modules — MCQ Content Knowledge",
                            "body":    (
                                f"Friedman χ² = {r['friedman_stat']:.4f}  |  "
                                f"p = {r['p_value']:.4f}  |  "
                                f"Kendall's W = {r['kendalls_w']:.4f} ({r['effect_size_label']})"
                            ),
                            "table":   mbt_df,
                            "caption": "Mean % correct per module across all students.",
                        })
                except Exception:
                    pass

                # Survey constructs
                sccces_constructs = [
                    "engagement_with_task", "effort_and_persistence", "experience_of_flow",
                    "coherency_of_messaging", "plausibility_of_messaging",
                ]
                sims_constructs = ["intrinsic_motivation", "identified_regulation",
                                   "external_regulation", "amotivation"]
                for survey_key, constructs in [
                    ("b4ai_sccces_survey", sccces_constructs),
                    ("b4ai_sims_survey",   sims_constructs),
                ]:
                    survey_label = _SURVEY_LABELS.get(survey_key, survey_key)
                    summary_rows = []
                    for construct in constructs:
                        try:
                            r = run_repeated_measures(canonical_df, survey_key, construct=construct)
                            if r.get("error"):
                                continue
                            summary_rows.append({
                                "Construct":    construct.replace("_"," ").title(),
                                "Friedman χ²":  f"{r['friedman_stat']:.3f}",
                                "p-value":      f"{r['p_value']:.4f}",
                                "Kendall's W":  f"{r['kendalls_w']:.4f}",
                                "Effect":       r["effect_size_label"],
                                "Significant":  "Yes" if r["significant"] else "No",
                            })
                        except Exception:
                            pass
                    if summary_rows:
                        sections.append({
                            "heading": f"Across Modules — {survey_label}",
                            "body":    "Friedman test per construct across 7 modules.",
                            "table":   pd.DataFrame(summary_rows),
                            "caption": "Friedman test results for all constructs. α = 0.05.",
                        })

            pdf_bytes = _build_pdf(sections, "Inferential Statistics Report")
            st.download_button(
                "⬇️ Download Inferential Statistics Report (PDF)",
                data=pdf_bytes,
                file_name="b4ai_inferential_statistics.pdf",
                mime="application/pdf",
                key="rpt_inf_dl",
            )
            st.success("PDF ready.")


# -----------------------------------------------------------------------
# iii. IRT Analysis report
# -----------------------------------------------------------------------

def _report_irt(canonical_df: pd.DataFrame) -> None:
    st.markdown("### iii. IRT Analysis Report")
    st.caption("Generates a PDF of IRT model results for the selected instrument.")

    if not _IRT_AVAILABLE:
        st.error("IRT library (girth) not available — cannot generate IRT report.")
        return

    col1, col2 = st.columns(2)
    with col1:
        irt_report_type = st.selectbox(
            "Instrument type",
            ["Binary Assessments", "Survey (GRM)"],
            key="rpt_irt_type",
        )
    with col2:
        if irt_report_type == "Binary Assessments":
            irt_model = st.selectbox("Model", ["Rasch (1PL)", "2PL (requires n≥50)"],
                                     key="rpt_irt_model")
        else:
            irt_survey = st.selectbox("Survey", ["SCCCES", "SIMS"], key="rpt_irt_survey")
            irt_construct = st.selectbox(
                "Construct",
                ["engagement_with_task","effort_and_persistence","experience_of_flow",
                 "intrinsic_motivation","identified_regulation","external_regulation","amotivation"],
                key="rpt_irt_construct",
            )

    if st.button("📄 Generate IRT Report (PDF)", key="rpt_irt_gen", type="primary"):
        with st.spinner("Fitting model and building PDF…"):
            sections = [{"heading": "IRT Analysis Report", "body": ""}]
            try:
                if irt_report_type == "Binary Assessments":
                    # Run Rasch on all binary instruments
                    for inst_key in [
                        "precourse_pre_ai_misconceptions_assessment",
                        "postcourse_post_ai_misconceptions_assessment",
                        "precourse_pre_aici_assessment",
                        "postcourse_post_aici_assessment",
                    ] + [f"module{n}_content_mcq_assessment" for n in discover_all_module_numbers()]:
                        try:
                            mat, item_ids = build_binary_response_matrix(canonical_df, inst_key)
                            if len(mat) < 3:
                                continue
                            if "2PL" in irt_model:
                                res = run_2pl_model(mat, item_ids)
                            else:
                                res = run_rasch_model(mat, item_ids)
                            if res.get("error"):
                                continue
                            label = _ASSESSMENT_LABELS.get(inst_key, inst_key)
                            params = res.get("item_params", pd.DataFrame())
                            sections.append({
                                "heading": f"IRT — {label}",
                                "body":    (
                                    f"Model: {res['model_type']}  |  "
                                    f"N persons: {res['n_persons']}  |  "
                                    f"N items: {res['n_items']}  |  "
                                    f"AIC: {res.get('aic','—')}"
                                ),
                                "table":   params.round(3) if not params.empty else None,
                                "caption": "Item parameters estimated by IRT model.",
                            })
                        except Exception:
                            pass
                else:
                    survey_key = ("b4ai_sccces_survey" if irt_survey == "SCCCES"
                                  else "b4ai_sims_survey")
                    lmat, litem_ids = build_likert_response_matrix(
                        canonical_df, survey_key, irt_construct
                    )
                    res = run_grm_model(lmat, litem_ids)
                    if not res.get("error"):
                        params = res.get("item_params", pd.DataFrame())
                        sections.append({
                            "heading": f"GRM — {irt_construct.replace('_',' ').title()}",
                            "body":    (
                                f"Model: {res['model_type']}  |  "
                                f"N persons: {res['n_persons']}  |  "
                                f"N items: {res['n_items']}"
                            ),
                            "table":   params.round(3) if not params.empty else None,
                            "caption": "Item parameters from Graded Response Model.",
                        })

                pdf_bytes = _build_pdf(sections, "IRT Analysis Report")
                st.download_button(
                    "⬇️ Download IRT Report (PDF)",
                    data=pdf_bytes,
                    file_name="b4ai_irt_analysis.pdf",
                    mime="application/pdf",
                    key="rpt_irt_dl",
                )
                st.success("PDF ready.")
            except Exception as e:
                st.error(f"IRT report error: {e}")


# -----------------------------------------------------------------------
# iv. LLM Analysis report
# -----------------------------------------------------------------------

def _report_llm() -> None:
    st.markdown("### iv. LLM Analysis Report")

    col_ita, col_dta = st.columns(2)

    # ITA sub-report
    with col_ita:
        st.markdown("**Inductive Thematic Analysis (ITA)**")
        if not _LLM_AVAILABLE:
            st.warning("LLM modules not available.")
        else:
            ita_runs = _ita_list_runs()
            completed = [r for r in ita_runs if r.get("phase_reached", 0) >= 6]
            if not completed:
                st.info("No completed ITA runs found. Run ITA first.")
            else:
                ita_opts = {
                    f"{r['model'].upper()} T={r['temperature']} {r['created_at'][:10]}"
                    f" — {r.get('cohort_scope') or 'All cohorts'}":
                    r["run_id"] for r in completed[:8]
                }
                sel_ita = st.selectbox("Select ITA run", list(ita_opts.keys()),
                                       key="rpt_ita_run")
                if st.button("📄 Generate ITA PDF", key="rpt_ita_gen", type="primary"):
                    with st.spinner("Building ITA report PDF…"):
                        run_id = ita_opts[sel_ita]
                        run    = _ita_get_run(run_id)
                        p6     = load_phase_result(run_id, 6)
                        p5     = load_phase_result(run_id, 5)
                        p2     = load_phase_result(run_id, 2)

                        sections = [{
                            "heading": "Inductive Thematic Analysis Report",
                            "body":    (
                                f"Model: {_llm_display_name(run['model'])}  |  "
                                f"Temperature: {run['temperature']}  |  "
                                f"Source: {run.get('source_type','—')}  |  "
                                f"Cohort scope: {run.get('cohort_scope') or 'All cohorts'}"
                            ),
                        }]

                        if p6 and p6.get("report_text"):
                            sections.append({
                                "heading": "Phase 6 — Analytical Report",
                                "body":    p6["report_text"],
                            })

                        # codes is needed both for the per-theme quotes below and
                        # the Phase 2 table further down -- compute it once here.
                        codes = (p2 or {}).get("codes", [])

                        themes = (p5 or {}).get("themes_defined") or (p5 or {}).get("themes", [])
                        if themes:
                            # Phase 6's own prompt only asks the LLM to include
                            # quotes "where available" in free-flowing prose --
                            # a soft instruction the model doesn't reliably
                            # honor. Fixed 2026-08-09: deterministically list up
                            # to 3 attributed, verbatim quotes per theme in the
                            # PDF itself (the same "most significant" cap Phase
                            # 6 already uses when building its own prompt), so
                            # citation doesn't depend on the narrative prose
                            # choosing to include them.
                            try:
                                from core.analytics.llm.ita_pipeline import _get_theme_codes
                            except ImportError:
                                _get_theme_codes = None

                            for i, t in enumerate(themes):
                                _body = t.get("summary", t.get("description", ""))
                                if _get_theme_codes and codes:
                                    _t_codes = _get_theme_codes(t, codes)
                                    _quotes = [
                                        (c.get("quote", "").strip(), c.get("participant_id", ""))
                                        for c in _t_codes if c.get("quote", "").strip()
                                    ][:3]
                                    if _quotes:
                                        _body += "\n\nRepresentative quotes:\n" + "\n".join(
                                            f'  "{q}" — {pid or "participant"}' for q, pid in _quotes
                                        )
                                sections.append({
                                    "heading": f"Theme {i+1}: {t.get('name','')}",
                                    "body":    _body,
                                })

                        if p2:
                            dedup = p2.get("dedup", {})
                            if codes:
                                codes_df = pd.DataFrame([
                                    {"Participant": c.get("participant_id",""),
                                     "Code Name":   c.get("name",""),
                                     # Was hard-truncated to [:80] chars, which
                                     # read as a broken mid-sentence cutoff once
                                     # _build_pdf() started wrapping cells
                                     # properly (fixed 2026-08-09) -- the PDF
                                     # table now wraps full text correctly, so
                                     # there's no longer a reason to pre-truncate
                                     # the underlying data itself.
                                     "Description": c.get("description","")}
                                    for c in codes[:50]
                                ])
                                sections.append({
                                    "heading": "Phase 2 — Initial Codes (first 50)",
                                    "body":    (
                                        f"Raw codes: {p2.get('n_codes_raw', len(codes))}  |  "
                                        f"After deduplication: {dedup.get('n_after', len(codes))}"
                                    ),
                                    "table":   codes_df,
                                    "caption": "Initial codes generated in Phase 2.",
                                })

                        pdf_bytes = _build_pdf(sections, "ITA Report")
                        st.download_button(
                            "⬇️ Download ITA Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"b4ai_ita_{run_id[:8]}.pdf",
                            mime="application/pdf",
                            key="rpt_ita_dl",
                        )
                        st.success("PDF ready.")

    # DTA sub-report
    with col_dta:
        st.markdown("**Deductive Thematic Analysis (DTA)**")
        if not _LLM_AVAILABLE:
            st.warning("LLM modules not available.")
        else:
            dta_runs = [r for r in _dta_list_runs()
                        if r.get("construct_groups","[]") != '["learning_objectives"]']
            if not dta_runs:
                st.info("No DTA runs found. Run DTA first.")
            else:
                dta_opts = {
                    f"{r['model'].upper()} T={r['temperature']} {r['created_at'][:10]}"
                    f" — {r.get('cohort_scope') or 'All cohorts'}":
                    r["run_id"] for r in dta_runs[:8]
                }
                sel_dta = st.selectbox("Select DTA run", list(dta_opts.keys()),
                                       key="rpt_dta_run")
                if st.button("📄 Generate DTA PDF", key="rpt_dta_gen", type="primary"):
                    with st.spinner("Building DTA report PDF…"):
                        run_id = dta_opts[sel_dta]
                        run_rec = next(r for r in dta_runs if r["run_id"] == run_id)
                        df_dta  = load_dta_results(run_id)

                        sections = [{
                            "heading": "Deductive Thematic Analysis Report",
                            "body":    (
                                f"Model: {_llm_display_name(run_rec['model'])}  |  "
                                f"Temperature: {run_rec['temperature']}  |  "
                                f"Source: {run_rec.get('source_type','—')}  |  "
                                f"Cohort scope: {run_rec.get('cohort_scope') or 'All cohorts'}"
                            ),
                        }]

                        if not df_dta.empty:
                            # Construct-level summary
                            summary = df_dta.groupby(
                                ["construct_group","construct_name"]
                            ).agg(
                                total_evidence=("evidence_count","sum"),
                                n_participants=("evidence_count", lambda x: (x>0).sum()),
                                positive=("valence_positive","sum"),
                                negative=("valence_negative","sum"),
                            ).reset_index()
                            summary["construct_name"] = (
                                summary["construct_name"].str.replace("_"," ").str.title()
                            )
                            summary.columns = ["Group","Construct","Evidence","Participants",
                                               "Positive","Negative"]
                            sections.append({
                                "heading": "Evidence Summary by Construct",
                                "body":    f"Total constructs analysed: {len(summary)}",
                                "table":   summary,
                                "caption": "Evidence count and valence per construct across all participants.",
                            })

                            # Per-participant summary
                            part_summ = df_dta.groupby("participant_id").agg(
                                total_evidence=("evidence_count","sum"),
                                constructs_found=("evidence_count", lambda x: (x>0).sum()),
                                positive=("valence_positive","sum"),
                            ).reset_index()
                            part_summ.columns = ["Participant","Total Evidence",
                                                 "Constructs Found","Positive Instances"]
                            sections.append({
                                "heading": "Participant-Level Evidence Summary",
                                "table":   part_summ,
                                "caption": "Evidence found per participant across all constructs.",
                            })

                        pdf_bytes = _build_pdf(sections, "DTA Report")
                        st.download_button(
                            "⬇️ Download DTA Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"b4ai_dta_{run_id[:8]}.pdf",
                            mime="application/pdf",
                            key="rpt_dta_dl",
                        )
                        st.success("PDF ready.")


# -----------------------------------------------------------------------
# vi. Full Programme Report
# -----------------------------------------------------------------------

def _report_full_programme(
    canonical_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    cohort_map: dict,
    username: str,
) -> None:
    st.markdown("### vi. Full Programme Report")
    st.caption(
        "Generates a single PDF combining all available analysis sections. "
        "Run each analysis tab first to populate all sections."
    )

    _FULL_OPTIONS = (
        ["Basic Statistics", "Inferential Statistics"] +
        (["IRT Analysis"] if _IRT_AVAILABLE else []) +
        (["ITA (latest run)", "DTA (latest run)"] if _LLM_AVAILABLE else [])
    )
    full_selected = st.multiselect(
        "Include sections:",
        options=_FULL_OPTIONS,
        default=_FULL_OPTIONS,
        key="rpt_full_multiselect",
    )

    cohort_name = st.text_input(
        "Cohort / study name for report header",
        value=list(cohort_map.values())[0] if cohort_map else "Basics4AI",
        key="rpt_full_cohort",
    )

    if st.button("📄 Generate Full Programme Report (PDF)",
                 key="rpt_full_gen", type="primary"):
        full_sel = st.session_state.get("rpt_full_multiselect", _FULL_OPTIONS)
        inc_bs  = "Basic Statistics"      in full_sel
        inc_inf = "Inferential Statistics" in full_sel
        inc_irt = "IRT Analysis"          in full_sel
        inc_ita = "ITA (latest run)"      in full_sel
        inc_dta = "DTA (latest run)"      in full_sel

        with st.spinner("Compiling all sections — this may take a moment…"):
            import io as _io2

            all_sections = [{
                "heading": f"Basics4AI Programme Report — {cohort_name}",
                "body":    (
                    f"Participants: {len(demographics_df)}  |  "
                    f"Modules: {canonical_df['module_id'].nunique()}  |  "
                    f"Generated by: {username}"
                ),
            }]

            def _safe_add(label, fn):
                try:
                    fn(all_sections)
                except Exception as e:
                    all_sections.append({
                        "heading": label,
                        "body":    f"Could not generate this section: {e}",
                    })

            if inc_bs:
                def _add_bs(secs):
                    from core.analytics.descriptive.score_aggregator import (
                        compute_assessment_scores, compute_construct_means,
                        aggregate_construct_means, summarize_scores,
                    )
                    secs.append({"heading": "─── BASIC STATISTICS ───", "body": ""})
                    asc = compute_assessment_scores(canonical_df)
                    if not asc.empty:
                        for inst_key in sorted(asc["instrument_key"].unique())[:4]:
                            label = _ASSESSMENT_LABELS.get(inst_key, inst_key)
                            asc_f = asc[asc["instrument_key"] == inst_key]
                            summ  = summarize_scores(asc_f)
                            if not summ.empty:
                                row = summ.iloc[0]
                                secs.append({
                                    "heading": f"Assessment: {label}",
                                    "body":    f"N={int(row.get('n_users',0))}  Mean={row.get('mean_pct',0):.1f}%  Median={row.get('median_pct',0):.1f}%",
                                })
                _safe_add("Basic Statistics", _add_bs)

            if inc_inf:
                def _add_inf(secs):
                    secs.append({"heading": "─── INFERENTIAL STATISTICS ───", "body": ""})
                    for label, pre_k, post_k in [
                        ("AI Misconceptions",
                         "precourse_pre_ai_misconceptions_assessment",
                         "postcourse_post_ai_misconceptions_assessment"),
                    ]:
                        r = run_paired_comparison(canonical_df, pre_k, post_k)
                        if not r.get("error"):
                            secs.append({
                                "heading": f"Pre vs Post — {label}",
                                "body":    (
                                    f"Pre={r['pre_mean']:.2f}  Post={r['post_mean']:.2f}  "
                                    f"d={r['cohens_d']:.3f} ({r['effect_size_label']})  "
                                    f"p={r['t_p_value']:.4f}"
                                ),
                            })
                _safe_add("Inferential Statistics", _add_inf)

            if inc_ita and _LLM_AVAILABLE:
                def _add_ita(secs):
                    secs.append({"heading": "─── ITA ANALYSIS ───", "body": ""})
                    runs = [r for r in _ita_list_runs() if r.get("phase_reached",0)>=6]
                    if runs:
                        r   = runs[0]
                        p6  = load_phase_result(r["run_id"], 6)
                        if p6 and p6.get("report_text"):
                            secs.append({
                                "heading": f"ITA — {_llm_display_name(r['model'])} T={r['temperature']}",
                                "body":    p6["report_text"][:2000] + ("…" if len(p6["report_text"]) > 2000 else ""),
                            })
                _safe_add("ITA", _add_ita)

            if inc_dta and _LLM_AVAILABLE:
                def _add_dta(secs):
                    secs.append({"heading": "─── DTA ANALYSIS ───", "body": ""})
                    runs = [r for r in _dta_list_runs()
                            if r.get("construct_groups","[]") != '["learning_objectives"]']
                    if runs:
                        r  = runs[0]
                        df = load_dta_results(r["run_id"])
                        if not df.empty:
                            summ = df.groupby("construct_name").agg(
                                evidence=("evidence_count","sum"),
                                participants=("evidence_count", lambda x: (x>0).sum()),
                            ).reset_index()
                            summ["construct_name"] = summ["construct_name"].str.replace("_"," ").str.title()
                            summ.columns = ["Construct","Total Evidence","Participants with Evidence"]
                            secs.append({
                                "heading": f"DTA Summary — {_llm_display_name(r['model'])}",
                                "table":   summ,
                                "caption": "Evidence counts across all constructs and participants.",
                            })
                _safe_add("DTA", _add_dta)

            pdf_bytes = _build_pdf(all_sections, f"Full Programme Report — {cohort_name}")
            st.download_button(
                "⬇️ Download Full Programme Report (PDF)",
                data=pdf_bytes,
                file_name=f"b4ai_full_report_{cohort_name.replace(' ','_')}.pdf",
                mime="application/pdf",
                key="rpt_full_dl",
            )
            st.success("Full programme report PDF ready.")
