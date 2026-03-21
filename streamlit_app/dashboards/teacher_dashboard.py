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

import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from core.analytics.datasets.canonical_loader import load_canonical_data
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
        _DTA_PHASE2_PROMPT, _parse_dta_json,
        _detect_matched_indicators,
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
       for n in range(1, 8)},
}

_SURVEY_LABELS: Dict[str, str] = {
    "b4ai_sccces_survey": "SCCCES (Conceptual Change)",
    "b4ai_sims_survey":   "SIMS (Motivation)",
}

# Maps module_id (canonical) → display label
_MODULE_LABELS: Dict[str, str] = {
    **{f"module_{n}": f"Module {n}" for n in range(1, 8)},
    "global":       "Global (Pre/Post)",
    "demographics": "Demographics",
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
        st.plotly_chart(fig, width='stretch')
    else:
        st.bar_chart(values.dropna())


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


_QUESTION_MAP = {
    'Q10_1': ('SCCCES', 'Culture', 'While going through this module, I thought about whether the topic conflicts with my culture (for example, religion or family values).'),
    'Q10_2': ('SCCCES', 'Culture', 'While going through this module, I thought about whether I agreed with the topic conflicts based on my culture (for example, religion or family values).'),
    'Q11_1': ('SCCCES', 'Personal Relevance', 'While going through this module, I thought about how this topic relates to things I like or care about.'),
    'Q11_2': ('SCCCES', 'Personal Relevance', 'While going through this module, I thought about how the information could be useful to me.'),
    'Q11_3': ('SCCCES', 'Personal Relevance', 'While going through this module, I thought about how the activities would be helpful to my personal goals.'),
    'Q2_1': ('SCCCES', 'Engagement With Task', 'I was engaged with the topic at hand.'),
    'Q2_2': ('SIMS', 'Intrinsic Motivation', 'Because I like doing this activity'),
    'Q2_3': ('SIMS', 'Intrinsic Motivation', 'Because I feel good when doing this activity'),
    'Q3_1': ('SCCCES', 'Effort And Persistence', 'I put in a lot of effort.'),
    'Q3_2': ('SCCCES', 'Effort And Persistence', 'I wish we could still continue with the work for a while.'),
    'Q3_3': ('SIMS', 'Identified Regulation', 'Because this activity will help me later'),
    'Q4_1': ('SCCCES', 'Experience Of Flow', 'I was so involved that I forgot everything around me.'),
    'Q4_2': ('SIMS', 'External Regulation', 'Because I have no choice'),
    'Q4_3': ('SIMS', 'External Regulation', 'Because I do not want to get in trouble'),
    'Q4_4': ('SIMS', 'External Regulation', 'Because I feel I have to do it'),
    'Q5_1': ('SCCCES', 'Coherency Of Messaging', 'While going through this module, I thought about whether the information was well organized.'),
    'Q5_2': ('SCCCES', 'Coherency Of Messaging', 'While going through this module, I considered whether the information was easy to understand.'),
    'Q5_3': ('SCCCES', 'Coherency Of Messaging', 'While going through this module, I thought about whether the information flowed well.'),
    'Q6_1': ('SCCCES', 'Plausibility Of Messaging', 'While going through this module, I thought about whether the information was believable.'),
    'Q6_2': ('SCCCES', 'Plausibility Of Messaging', 'While going through this module, I thought about whether the information was reasonable.'),
    'Q7_1': ('SCCCES', 'Credibility Of Messaging', 'While going through this module, I thought about whether the source of the information was trustworthy.'),
    'Q7_2': ('SCCCES', 'Credibility Of Messaging', 'While going through this module, I thought about whether the source of the information was believable.'),
    'Q8_1': ('SCCCES', 'Comprehensibility Of Messaging', 'While going through this module, I thought about whether the information presented was easy to follow.'),
    'Q8_2': ('SCCCES', 'Comprehensibility Of Messaging', 'While going through this module, I thought about whether the information was clear.'),
    'Q9_1': ('SCCCES', 'Attention', 'I was having trouble paying attention during the module.'),
    'Q9_2': ('SCCCES', 'Attention', 'I was distracted by other thoughts during the module.'),
}

# Reverse map: instrument -> sorted list of (qid, construct, text) for guide
def _build_question_guide(instrument: str) -> list:
    rows = []
    for qid, (inst, con, txt) in sorted(_QUESTION_MAP.items()):
        if inst == instrument:
            rows.append({"Question ID": qid, "Construct": con, "Question Text": txt})
    return rows


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
            options=["Question", "Student"],
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
            pct_by_q.columns = ["Question", "% Correct", "N Students"]
            st.dataframe(
                pct_by_q, hide_index=True, width="stretch",
                column_config={
                    "Question":   st.column_config.TextColumn("Question",  width="small"),
                    "% Correct":  st.column_config.NumberColumn("% Correct", format="%.1f", width="small"),
                    "N Students": st.column_config.NumberColumn("N Students", format="%d",   width="small"),
                }
            )

    else:
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
        "scale_low":  "Learners reported no external pressure — they chose the activity freely.",
        "scale_mid":  "Learners felt some external pressure but also had some personal buy-in.",
        "scale_high": "Learners reported doing the activity primarily because they had to, not by choice.",
        "reverse_coded": True,
        "reverse_note": "⚠️ Reverse-coded construct: higher scores indicate more externally controlled (less self-determined) motivation. A mean above 2.5 warrants attention.",
    },
    "amotivation": {
        "label": "Amotivation",
        "definition": "A lack of motivation — feeling no reason to do the activity and disconnected from outcomes.",
        "analytic_focus": ["disengagement", "helplessness", "lack of purpose"],
        "scale_low":  "Learners showed no signs of disengagement — they had clear reasons for participating.",
        "scale_mid":  "Learners showed some motivational uncertainty or occasional disengagement.",
        "scale_high": "Learners felt disconnected from the activity, saw no purpose, and showed signs of helplessness.",
        "reverse_coded": True,
        "reverse_note": "⚠️ Reverse-coded construct: higher scores indicate greater disengagement and lack of motivation. A mean above 2.0 is a concern worth investigating.",
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

    # Summary table across users for each construct
    summary_cm = summarize_scores(cm_filtered)

    if not summary_cm.empty:
        st.markdown("**Summary Statistics per Construct**")
        display_summary = summary_cm[["construct","n_users","mean_score","median_score","mode_score"]].copy()
        display_summary.columns = ["Construct", "N", "Mean", "Median", "Mode"]
        for col in ["Mean", "Median", "Mode"]:
            display_summary[col] = display_summary[col].apply(
                lambda x: f"{x:.2f}" if x is not None else "—"
            )
        st.dataframe(
            display_summary,
            hide_index=True, width="stretch",
            column_config={
                "Construct": st.column_config.TextColumn("Construct", width="medium"),
                "N":         st.column_config.NumberColumn("N", format="%d", width="small"),
                "Mean":      st.column_config.TextColumn("Mean", width="small"),
                "Median":    st.column_config.TextColumn("Median", width="small"),
                "Mode":      st.column_config.TextColumn("Mode", width="small"),
            }
        )

        # Reverse-coding alert for any selected construct
        rev_coded = [c for c in selected_constructs
                     if _CONSTRUCT_DEFINITIONS.get(c, {}).get("reverse_coded")]
        if rev_coded:
            for rc in rev_coded:
                note = _CONSTRUCT_DEFINITIONS[rc].get("reverse_note", "")
                if note:
                    st.warning(note)

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
                st.divider()

    # View toggle: per-question or per-student
    survey_item_view = st.radio(
        "Chart view",
        options=["By Question Item", "By Student (distribution)"],
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
                # Rename by name (not position) to avoid order-dependent bugs
                display_items = item_means[
                    ["question_id", "construct", "mean_score", "n_students"]
                ].copy().rename(columns={
                    "question_id": "Question",
                    "construct":   "Construct",
                    "mean_score":  "Mean Score",
                    "n_students":  "N Students",
                })
                import streamlit as _st2
                st.dataframe(
                    display_items.reset_index(drop=True),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Question":   st.column_config.TextColumn("Question",  width="small"),
                        "Construct":  st.column_config.TextColumn("Construct", width="medium"),
                        "Mean Score": st.column_config.NumberColumn("Mean Score", format="%.2f", width="small"),
                        "N Students": st.column_config.NumberColumn("N Students", format="%d",   width="small"),
                    }
                )
                # ─── Question guide ───────────────────────────────────────────
                _inst_key = "SIMS" if "sims" in selected_survey_base else "SCCCES"
                _guide_rows = _build_question_guide(_inst_key)
                if _guide_rows:
                    import pandas as _gpd
                    with st.expander(f"📋 Question ID guide ({_inst_key})", expanded=False):
                        st.dataframe(
                            _gpd.DataFrame(_guide_rows),
                            hide_index=True, width="stretch",
                            column_config={
                                "Question ID": st.column_config.TextColumn("Question ID", width="small"),
                                "Construct":   st.column_config.TextColumn("Construct",   width="medium"),
                                "Question Text": st.column_config.TextColumn("Question Text", width="large"),
                            }
                        )

    else:
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

def _render_result_card(result: dict, score_label: str = "% Correct") -> None:
    """Render a single test result as a clean card."""
    if result.get("error"):
        st.error(f"Could not compute: {result['error']}")
        return

    sig   = result.get("significant", False)
    alpha = result.get("alpha", 0.05)
    sig_badge = "✅ Significant" if sig else "— Not significant"

    # Main stats row
    cols = st.columns(4)
    if "pre_mean" in result:
        cols[0].metric(f"Pre mean",  f"{result['pre_mean']:.1f}")
        cols[1].metric(f"Post mean", f"{result['post_mean']:.1f}",
                       delta=f"{result['mean_diff']:+.1f}")
        cols[2].metric("Cohen's d",
                       f"{result['cohens_d']:.3f} ({result['effect_size_label']})")
        cols[3].metric("Paired t p-value",
                       f"{result['t_p_value']:.4f}  {sig_badge}")
    elif "f_stat" in result:
        cols[0].metric("F statistic",  f"{result['f_stat']:.4f}")
        cols[1].metric("ANOVA p-value",f"{result['anova_p']:.4f}  {sig_badge}")
        cols[2].metric("η² (eta²)",
                       f"{result['eta_squared']:.4f} ({result['effect_size_label']})")
        cols[3].metric("Kruskal-Wallis p", f"{result['kruskal_p']:.4f}")
    elif "friedman_stat" in result:
        cols[0].metric("Friedman χ²",  f"{result['friedman_stat']:.4f}")
        cols[1].metric("p-value",      f"{result['p_value']:.4f}  {sig_badge}")
        cols[2].metric("Kendall's W",
                       f"{result['kendalls_w']:.4f} ({result['effect_size_label']})")
        cols[3].metric("N subjects",   str(result.get("n_subjects","")))

        # ── RM ANOVA-style SS/df/MS/F table ──────────────────────────────────
        # Reconstruct from Friedman χ² and Kendall's W
        # χ² = k(n-1)W  →  SS_conditions = χ²·MS_error (approx)
        # Using the relationship: F ≈ χ²/(k-1) / ((n·k - χ² - k)/(k-1)(n-1))
        _n  = result.get("n_subjects", 0)
        _tp = result.get("time_points", [])
        _k  = len(_tp) if _tp else 1
        _W  = result.get("kendalls_w", 0)
        _chi = result.get("friedman_stat", 0)
        if _n > 1 and _k > 1:
            _df_cond  = _k - 1
            _df_subj  = _n - 1
            _df_err   = (_k - 1) * (_n - 1)
            # SS from Kendall's W: SS_conditions = 12*n*W*SS_ranks/(k(k²-1))
            # Simpler approximation from χ²:
            _SS_cond  = round(_chi * (_k - 1) / _k, 3) if _k > 0 else None
            _SS_err   = round((_n * _k * (_k + 1) / 12) - _chi / (_k - 1), 3) if _k > 1 else None
            _MS_cond  = round(_SS_cond / _df_cond, 3) if _SS_cond and _df_cond else None
            _MS_err   = round(_SS_err  / _df_err,  3) if _SS_err  and _df_err  else None
            _F_val    = round(_MS_cond / _MS_err,  3) if _MS_cond and _MS_err and _MS_err != 0 else None

            # All columns must be same type for Arrow — use strings throughout
            def _fmt(v):
                return f"{v:.3f}" if v is not None else "—"
            _rm_table = pd.DataFrame([
                {"Source": "Conditions (Time)",
                 "SS": _fmt(_SS_cond), "df": str(_df_cond),
                 "MS": _fmt(_MS_cond), "F":  _fmt(_F_val)},
                {"Source": "Subjects",
                 "SS": "—", "df": str(_df_subj), "MS": "—", "F": "—"},
                {"Source": "Error",
                 "SS": _fmt(_SS_err),  "df": str(_df_err),
                 "MS": _fmt(_MS_err),  "F": ""},
            ])
            with st.expander("📊 RM ANOVA-style source table", expanded=False):
                st.dataframe(
                    _rm_table,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        c: st.column_config.Column(c, width="small")
                        for c in _rm_table.columns
                    }
                )
                st.caption(
                    "SS and MS are approximated from Friedman χ² and Kendall's W. "
                    "The Friedman test is non-parametric and does not produce exact "
                    "SS values — treat these as indicative, not exact ANOVA decomposition."
                )

        # ── Means by time point ───────────────────────────────────────────────
        mbt = result.get("means_by_time", {})
        sbt = result.get("stds_by_time",  {})
        if mbt:
            _means_df = pd.DataFrame([
                {"Module": tp,
                 "Mean":   round(mbt[tp], 3) if tp in mbt else "—",
                 "SD":     round(sbt[tp], 3) if tp in sbt else "—"}
                for tp in sorted(mbt.keys())
            ])
            with st.expander("📈 Means by module / time point", expanded=False):
                st.dataframe(_means_df, hide_index=True, width="stretch")

    # Plain-language interpretation
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

    # Wilcoxon row (if present)
    if result.get("wilcoxon_stat") is not None:
        st.caption(
            f"Wilcoxon signed-rank: W={result['wilcoxon_stat']}, "
            f"p={result['wilcoxon_p']:.4f}"
        )
        with st.expander("ℹ️ What is the Wilcoxon test?", expanded=False):
            st.markdown(_STAT_HELP["wilcoxon"])

    # Power panel
    if result.get("low_n_warning"):
        with st.expander(
            "⚠️  Low-N Warning + Sample Size Guidance", expanded=True
        ):
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
            is_paired = "pre_mean" in result
            is_repeated = "friedman_stat" in result
            if is_paired:
                pwr_df = pd.DataFrame([
                    {"Cohen's d": "Small (0.2)",
                     "Practical meaning": "Subtle",
                     "N for 80% power": 199,
                     "N for 95% power": 327},
                    {"Cohen's d": "Medium (0.5)",
                     "Practical meaning": "Moderate — recommended minimum",
                     "N for 80% power": 34,
                     "N for 95% power": 55},
                    {"Cohen's d": "Large (0.8)",
                     "Practical meaning": "Substantial",
                     "N for 80% power": 15,
                     "N for 95% power": 24},
                ])
            elif is_repeated:
                pwr_df = pd.DataFrame([
                    {"Kendall's W": "Small (0.1)",
                     "Practical meaning": "Weak consistency",
                     "N for 80% power": ">200",
                     "N for 95% power": ">300"},
                    {"Kendall's W": "Medium (0.3)",
                     "Practical meaning": "Moderate — recommended minimum",
                     "N for 80% power": "~52",
                     "N for 95% power": "~85"},
                    {"Kendall's W": "Large (0.5)",
                     "Practical meaning": "Strong consistency",
                     "N for 80% power": "~21",
                     "N for 95% power": "~34"},
                ])
            else:
                pwr_df = pd.DataFrame([
                    {"Cohen's f": "Small (0.10)",
                     "η²": "≈0.01",
                     "Practical meaning": "Subtle",
                     "N per group (80%)": 322,
                     "N per group (95%)": 527},
                    {"Cohen's f": "Medium (0.25)",
                     "η²": "≈0.06",
                     "Practical meaning": "Moderate — recommended minimum",
                     "N per group (80%)": 52,
                     "N per group (95%)": 85},
                    {"Cohen's f": "Large (0.40)",
                     "η²": "≈0.14",
                     "Practical meaning": "Substantial",
                     "N per group (80%)": 21,
                     "N per group (95%)": 34},
                ])
            st.dataframe(pwr_df, hide_index=True, width="stretch")
            st.caption(
                f"With your planned n ≈ 90, you will have adequate power to "
                f"detect medium-to-large effects. The observed post-hoc power "
                f"at n = {n_shown} is shown below for reference only — "
                f"it should not be used to justify your sample size."
            )
            power_val = result.get("power_achieved")
            if power_val is not None:
                st.caption(
                    f"Observed post-hoc power (reference only): "
                    f"**{power_val*100:.1f}%** at n = {n_shown}"
                )
    # Group means table for between-groups
    if "group_means" in result:
        rows = [
            {"Group": g,
             "N": result["n_per_group"][g],
             f"Mean {score_label}": f"{m:.2f}",
             "SD": f"{result['group_stds'].get(g, 0):.2f}"}
            for g, m in result["group_means"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # Means-by-time table for repeated measures
    if "means_by_time" in result:
        rows = [
            {"Module": t, f"Mean {score_label}": f"{m:.4f}",
             "SD": f"{result['stds_by_time'].get(t,0):.4f}"}
            for t, m in result["means_by_time"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


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
# Placeholder tabs (Phases 2–5)
# -----------------------------------------------------------------------

def _render_placeholder(title: str, phase: int, bullets: list) -> None:
    st.markdown(f"### {title}")
    st.info(f"🔧 Coming in **Phase {phase}**")
    st.markdown("**Planned capabilities:**")
    for b in bullets:
        st.markdown(f"- {b}")


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
        "🔬 IRT Analysis",
        "🤖 LLM Analysis",
        "📉 Competency Progression",
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
        "🔬 IRT Analysis":           ("#009E73", "#E6F7F1"),
        "🤖 LLM Analysis":           ("#CC79A7", "#F9EEF5"),
        "📉 Competency Progression": ("#888888", "#F0F0F0"),
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
        _render_placeholder(
            "Competency Progression Index", phase=5,
            bullets=[
                "Composite index: misconception improvement + conceptual gain + MCQ score",
                "Per-student progression trajectory across modules",
                "Cohort-level CPI distribution and benchmarks",
                "Customizable construct weights",
            ]
        )

    elif active_tab == "📄 Report Generation":
        _render_report_tab(username, filtered_canonical, filtered_demographics, cohort_map)

# -----------------------------------------------------------------------
# Tab 2 — IRT Analysis
# -----------------------------------------------------------------------

def _render_irt_tab(canonical_df: pd.DataFrame) -> None:
    """Tab 2 — IRT Analysis using mirt via rpy2."""

    st.subheader("🔬 IRT Analysis")

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

# -----------------------------------------------------------------------
# Multi-source loader (shared by ITA and DTA)
# -----------------------------------------------------------------------

def _load_combined_transcripts(
    sources: list,
    per_run_files=None,
) -> list:
    """
    Load and merge transcripts from one or more sources.
    sources: list containing any of "responses", "persistent", "per_run"
    When a participant appears in multiple sources their texts are concatenated.
    """
    import sqlite3 as _sq, re as _re
    from pathlib import Path as _Pth
    from collections import defaultdict

    combined = defaultdict(lambda: {"content": [], "source_types": []})

    for source_key in sources:
        if source_key == "responses":
            db = next(
                (p / "responses.db" for p in _Pth(__file__).resolve().parents
                 if (p / "responses.db").exists()), None
            )
            if not db:
                continue
            conn = _sq.connect(db)
            rows = conn.execute(
                "SELECT user_id, response_value FROM responses "
                "WHERE instrument_name LIKE '%module_reflections%' "
                "AND response_value IS NOT NULL"
            ).fetchall()
            conn.close()
            for uid, rval in rows:
                if rval and str(rval).strip():
                    combined[uid]["content"].append(str(rval))
                    if "reflections" not in combined[uid]["source_types"]:
                        combined[uid]["source_types"].append("reflections")

        elif source_key == "persistent":
            try:
                trans = load_for_analysis(source="persistent", source_type="interview")
                for t in (trans or []):
                    pid  = t.get("participant_id", "unknown")
                    text = str(t.get("content", "")).strip()
                    if text:
                        combined[pid]["content"].append(text)
                        if "interview" not in combined[pid]["source_types"]:
                            combined[pid]["source_types"].append("interview")
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


def _source_checkboxes(prefix: str, default_reflections: bool = True):
    """Render 3-way source checkboxes. Returns (sources_list, per_run_files)."""
    st.markdown("**Data sources** (select one or more):")
    c1, c2, c3 = st.columns(3)
    use_ref = c1.checkbox("Module reflections (DB)",      value=default_reflections,
                          key=f"{prefix}_src_ref",
                          help="End-of-module reflection notes from responses.db")
    use_int = c2.checkbox("Interview transcripts (store)", value=False,
                          key=f"{prefix}_src_int",
                          help="Semi-structured transcripts uploaded via Admin dashboard")
    use_upl = c3.checkbox("Upload now (this run only)",    value=False,
                          key=f"{prefix}_src_upl",
                          help="Upload transcript files fresh for this run")

    sources = []
    if use_ref: sources.append("responses")
    if use_int: sources.append("persistent")
    if use_upl: sources.append("per_run")

    per_run_files = None
    if use_upl:
        per_run_files = st.file_uploader(
            "Upload transcript files (.vtt, .txt, .pdf)",
            type=["vtt", "txt", "pdf"],
            accept_multiple_files=True,
            key=f"{prefix}_upload",
        )
    return sources, per_run_files


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
        mode = st.radio("**Mode:**", ["🧭 Guided", "⚙️ Expert"],
                        horizontal=True, key="llm_mode_radio")
        _mc, _mbg = ("#CC79A7","#F9EEF5") if "Guided" in mode else ("#EE7733","#FFF3E6")
        st.markdown(
            f"<div style='background:{_mbg};border-left:4px solid {_mc};"
            f"border-radius:5px;padding:0.3rem 0.8rem;margin:0.3rem 0;'>"
            f"<strong style='color:{_mc};'>Mode: {mode}</strong></div>",
            unsafe_allow_html=True,
        )
        st.divider()
        if "Guided" in mode:
            _render_ita_guided(username, canonical_df)
        else:
            _render_ita_expert(username, canonical_df)
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
# ITA — Guided mode
# -----------------------------------------------------------------------

def _render_ita_guided(username: str, canonical_df: pd.DataFrame) -> None:
    # Seed defaults on first load
    _avail = get_available_models(check_keys=True)
    for _k, _d in [
        ("ita_g_claude", _avail.get("claude", False)),
        ("ita_g_gemini", False), ("ita_g_gpt", False),
        ("ita_g_temp", 0.0), ("ita_g_n_themes", 5),
        ("ita_g_n_codes", 3), ("ita_g_dedup", 0.85),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _d

    STEPS = ["1 — Data Source", "2 — Models & Temperature",
             "3 — Theme Settings", "4 — Review Prompt",
             "5 — Run Analysis", "6 — Results"]
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

    if step == STEPS[0]:
        st.markdown("### Step 1 — Select Data Source")
        sources, per_run_files = _source_checkboxes("ita_g")
        if not sources:
            st.warning("Select at least one source.")
        else:
            n_ref = 0
            if "responses" in sources:
                import sqlite3 as _sq2
                from pathlib import Path as _P2
                db2 = next((p / "responses.db" for p in _P2(__file__).resolve().parents
                            if (p / "responses.db").exists()), None)
                if db2:
                    conn2 = _sq2.connect(db2)
                    n_ref = conn2.execute(
                        "SELECT COUNT(DISTINCT user_id) FROM responses "
                        "WHERE instrument_name LIKE '%module_reflections%'"
                    ).fetchone()[0]
                    conn2.close()
            if "persistent" in sources:
                n_int = get_transcript_count("interview")
                st.metric("Interviews in store", n_int)
            if "responses" in sources:
                st.metric("Students with reflection notes", n_ref)

    elif step == STEPS[1]:
        st.markdown("### Step 2 — Models & Temperature")
        _avail2 = get_available_models(check_keys=True)
        st.markdown("**Select models:**")
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Claude", value=_avail2.get("claude", False),
                    disabled=not _avail2.get("claude", False), key="ita_g_claude")
        c2.checkbox("Gemini", value=False,
                    disabled=not _avail2.get("gemini", False), key="ita_g_gemini")
        c3.checkbox("GPT",    value=False,
                    disabled=not _avail2.get("gpt", False),    key="ita_g_gpt")
        st.divider()
        st.slider("Temperature (T)", 0.0, 2.0, 0.0, 0.1,
                  key="ita_g_temp", format="%.1f")

    elif step == STEPS[2]:
        st.markdown("### Step 3 — Theme Settings")
        st.slider("Number of themes (Phase 3)", 3, 15, 5, 1, key="ita_g_n_themes")
        st.slider("Deduplication threshold", 0.70, 0.95, 0.85, 0.05,
                  key="ita_g_dedup")
        st.select_slider("Codes per chunk", [2, 3], value=3, key="ita_g_n_codes")

    elif step == STEPS[3]:
        st.markdown("### Step 4 — Review Prompt")
        st.text_area("System prompt (fixed)", value=_ITA_SYSTEM_PROMPT,
                     height=150, disabled=True)
        st.text_area("Phase 2 prompt template",
                     value=_PHASE2_PROMPT.replace("{text}", "[TRANSCRIPT CHUNK]"),
                     height=200, disabled=True)

    elif step == STEPS[4]:
        st.markdown("### Step 5 — Run Analysis")
        sources, per_run_files = _source_checkboxes("ita_s5")

        models = [m for m, k in [("claude","ita_g_claude"),
                                   ("gemini","ita_g_gemini"),
                                   ("gpt","ita_g_gpt")]
                  if st.session_state.get(k, False)]
        temperature = float(st.session_state.get("ita_g_temp", 0.0))
        n_themes    = int(st.session_state.get("ita_g_n_themes", 5))
        n_codes     = int(st.session_state.get("ita_g_n_codes", 3))
        dedup_thr   = float(st.session_state.get("ita_g_dedup", 0.85))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Models", len(models) if models else "None")
        c2.metric("Temperature", f"{temperature:.1f}")
        c3.metric("Themes", n_themes)
        c4.metric("Sources", len(sources) if sources else "None")

        if not models:
            st.warning("No models selected — go to Step 2.")
            return
        if not sources:
            st.warning("Select at least one data source.")
            return

        if st.button("▶ Run ITA Pipeline", key="ita_g_run", type="primary"):
            _run_ita_pipeline(
                username=username, sources=sources, per_run_files=per_run_files,
                models=models, temperature=temperature,
                n_themes=n_themes, n_codes=n_codes, dedup_threshold=dedup_thr,
            )

    elif step == STEPS[5]:
        st.markdown("### Step 6 — Results")
        _render_ita_results(username)


# -----------------------------------------------------------------------
# ITA — Expert mode
# -----------------------------------------------------------------------

def _render_ita_expert(username: str, canonical_df: pd.DataFrame) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        sources, per_run_files = _source_checkboxes("ita_e")
    with col2:
        st.markdown("**Models:**")
        _avail = get_available_models(check_keys=True)
        e_models = []
        for m, lbl in [("claude","Claude"),("gemini","Gemini"),("gpt","GPT")]:
            if st.checkbox(lbl, value=_avail.get(m, False),
                           disabled=not _avail.get(m, False), key=f"ita_e_{m}"):
                e_models.append(m)

    c1, c2, c3, c4 = st.columns(4)
    e_temp     = c1.number_input("Temperature", 0.0, 2.0, 0.0, 0.1,
                                  format="%.1f", key="ita_e_temp")
    e_n_themes = int(c2.number_input("Themes", 3, 20, 5, 1, key="ita_e_n_themes"))
    e_n_codes  = int(c3.number_input("Codes/chunk", 2, 4, 3, 1, key="ita_e_n_codes"))
    e_dedup    = c4.number_input("Dedup", 0.70, 0.99, 0.85, 0.05,
                                  format="%.2f", key="ita_e_dedup")

    if not e_models:
        st.warning("Select at least one model.")
        return
    if not sources:
        st.warning("Select at least one data source.")
        return

    if st.button("▶ Run ITA Pipeline", key="ita_e_run", type="primary"):
        _run_ita_pipeline(
            username=username, sources=sources, per_run_files=per_run_files,
            models=e_models, temperature=e_temp,
            n_themes=e_n_themes, n_codes=e_n_codes, dedup_threshold=e_dedup,
        )
    st.divider()
    _render_ita_results(username)


# -----------------------------------------------------------------------
# ITA — Pipeline runner
# -----------------------------------------------------------------------

def _run_ita_pipeline(
    username, sources, per_run_files,
    models, temperature, n_themes, n_codes, dedup_threshold,
):
    src_label = " + ".join(sources) if sources else "none"
    with st.spinner(f"Loading transcripts from: {src_label}..."):
        try:
            transcripts = _load_combined_transcripts(sources, per_run_files)
        except Exception as e:
            st.error(f"Could not load transcripts: {e}"); return

    if not transcripts:
        st.warning(f"No transcript data found in: {src_label}."); return

    st.info(f"Loaded {len(transcripts)} participant(s). "
            f"Running {len(models)} model(s).")

    for model in models:
        st.markdown(f"---\n#### {_llm_display_name(model)}")
        run_id = _ita_create_run(
            model=model, temperature=temperature,
            source_type="+".join(sources),
            created_by=username,
            notes=f"n_themes={n_themes}, n_codes={n_codes}",
        )

        with st.status("Phase 1 — Chunking...", expanded=False) as s:
            chunks = run_phase1(transcripts, chunk_size=2500)
            s.update(label=f"Phase 1 ✅ — {len(chunks)} chunks", state="complete")

        with st.status(f"Phase 2 — Generating codes (T={temperature})...",
                       expanded=False) as s:
            p2 = run_phase2(chunks, model, temperature,
                            n_codes=n_codes, run_id=run_id)
            s.update(label=f"Phase 2 ✅ — {p2.get('n_codes_raw',0)} codes",
                     state="complete")

        with st.status("Phase 2b — Deduplicating...", expanded=False) as s:
            dedup = run_phase2_dedup(p2["codes"], threshold=dedup_threshold)
            save_phase_result(run_id, 2, {**p2, "dedup": dedup})
            s.update(label=f"Phase 2b ✅ — {dedup['n_before']}→{dedup['n_after']} codes",
                     state="complete")

        with st.status(f"Phase 3 — Searching themes...", expanded=False) as s:
            p3 = run_phase3(dedup["codes_dedup"], model, temperature,
                            n_themes=n_themes, run_id=run_id)
            if p3.get("error"):
                s.update(label=f"Phase 3 ❌ — {p3['error']}", state="error")
            else:
                s.update(label=f"Phase 3 ✅ — {p3['n_themes']} themes",
                         state="complete")

        if p3.get("error"):
            continue

        with st.status("Phase 4 — Reviewing themes (T=1.0)...", expanded=False) as s:
            p4 = run_phase4(dedup["codes_dedup"], model, temperature=1.0,
                            n_themes=n_themes, run_id=run_id)
            if p4.get("error"):
                s.update(label=f"Phase 4 ❌ — {p4['error']}", state="error")
            else:
                s.update(label=f"Phase 4 ✅ — {p4['n_themes']} themes at T=1.0",
                         state="complete")

        with st.status("Phase 5 — Defining themes...", expanded=False) as s:
            p5 = run_phase5(p3["themes"], dedup["codes_dedup"], model,
                            temperature, run_id=run_id)
            s.update(label=f"Phase 5 ✅ — {len(p5.get('themes_defined',[]))} themes defined",
                     state="complete")

        with st.status("Phase 6 — Writing report...", expanded=False) as s:
            p6 = run_phase6(p5["themes_defined"], dedup["codes_dedup"],
                            model, temperature, run_id=run_id)
            if p6.get("error"):
                s.update(label=f"Phase 6 ❌ — {p6['error']}", state="error")
            else:
                s.update(label=f"Phase 6 ✅ — {len(p6.get('report_text',''))} chars",
                         state="complete")

        st.session_state["ita_last_run_id"] = run_id
        st.success("ITA complete — view results in Step 6 / Results tab.")


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
    col1, col2 = st.columns(2)
    with col1:
        _avail = get_available_models(check_keys=True)
        st.markdown("**Models:**")
        dta_models = []
        mc1, mc2, mc3 = st.columns(3)
        if mc1.checkbox("Claude", value=_avail.get("claude", False),
                        disabled=not _avail.get("claude", False), key="dta_claude"):
            dta_models.append("claude")
        if mc2.checkbox("Gemini", value=False,
                        disabled=not _avail.get("gemini", False), key="dta_gemini"):
            dta_models.append("gemini")
        if mc3.checkbox("GPT", value=False,
                        disabled=not _avail.get("gpt", False), key="dta_gpt"):
            dta_models.append("gpt")
    with col2:
        dta_temp = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                             format="%.1f", key="dta_temp")

    st.divider()
    sources, per_run_files = _source_checkboxes("dta")

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

    col_r, col_lo = st.columns(2)
    run_dta = col_r.button("▶ Run DTA",         key="dta_run_btn", type="primary")
    run_lo  = col_lo.button("▶ Run LO Analysis", key="lo_run_btn")

    if run_dta:
        _execute_dta_run(
            username=username, sources=sources, per_run_files=per_run_files,
            models=dta_models, temperature=dta_temp,
            construct_groups=dta_groups,
            show_stream=st.session_state.get("dta_show_stream", False),
        )
    if run_lo:
        _execute_lo_run(username=username, models=dta_models, temperature=dta_temp)


def _execute_dta_run(username, sources, per_run_files,
                     models, temperature, construct_groups,
                     show_stream=False):
    src_label = " + ".join(sources)
    with st.spinner(f"Loading data from: {src_label}..."):
        try:
            transcripts = _load_combined_transcripts(sources, per_run_files)
        except Exception as e:
            st.error(f"Could not load data: {e}"); return

    if not transcripts:
        st.warning(f"No transcript data found in: {src_label}."); return

    constructs_to_run = {k: v for k, v in _DTA_CODEBOOK.items()
                         if v["group"] in construct_groups}
    st.info(f"Loaded {len(transcripts)} participant(s). "
            f"Analysing {len(constructs_to_run)} constructs × {len(models)} model(s).")
    st.caption(f"Estimated API calls: "
               f"{len(transcripts) * len(constructs_to_run) * len(models)}")

    import json as _j
    from datetime import datetime as _dt

    try:
        sys_p = _ITA_SYSTEM_PROMPT
    except Exception:
        sys_p = "You are a qualitative research assistant."

    for model in models:
        st.markdown(f"---\n#### {_llm_display_name(model)}")
        run_id = _dta_create_run(
            model=model, temperature=temperature,
            source_type="+".join(sources),
            construct_groups=construct_groups,
            created_by=username,
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


def _execute_lo_run(username, models, temperature):
    import sqlite3 as _sq3, re as _re3
    from pathlib import Path as _P3
    db3 = next((p/"responses.db" for p in _P3(__file__).resolve().parents
                if (p/"responses.db").exists()), None)
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
        db4 = next((p/"responses.db" for p in _P4(__file__).resolve().parents
                    if (p/"responses.db").exists()), None)
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
        "vi. Full Programme Report": ("#333333", "#F5F5F5"),
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
        st.info(
            "🔧 Competency Progression Index is under development. "
            "This section will provide composite index scores per student "
            "combining misconception improvement, conceptual gain, and MCQ scores. "
            "Reports will be available once the index is implemented."
        )

    elif rep_section == "vi. Full Programme Report":
        _report_full_programme(
            canonical_df, demographics_df, cohort_map, username
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
            # Convert dataframe to reportlab table
            col_headers = list(df.columns)
            data_rows   = df.astype(str).values.tolist()
            tbl_data    = [col_headers] + data_rows

            # Column widths — distribute across page
            page_w = A4[0] - 4*cm
            col_w  = page_w / max(len(col_headers), 1)
            col_widths = [col_w] * len(col_headers)

            tbl = Table(tbl_data, colWidths=col_widths, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#0077BB")),
                ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
                ("FONTSIZE",    (0,0), (-1,0),  8),
                ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
                ("ALIGN",       (0,0), (-1,-1), "LEFT"),
                ("FONTSIZE",    (0,1), (-1,-1), 7.5),
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
                    ] + [f"module{n}_content_mcq_assessment" for n in range(1,8)]:
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
                    f"{r['model'].upper()} T={r['temperature']} {r['created_at'][:10]}":
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
                                f"Source: {run.get('source_type','—')}"
                            ),
                        }]

                        if p6 and p6.get("report_text"):
                            sections.append({
                                "heading": "Phase 6 — Analytical Report",
                                "body":    p6["report_text"],
                            })

                        themes = (p5 or {}).get("themes_defined") or (p5 or {}).get("themes", [])
                        if themes:
                            for i, t in enumerate(themes):
                                sections.append({
                                    "heading": f"Theme {i+1}: {t.get('name','')}",
                                    "body":    t.get("summary", t.get("description","")),
                                })

                        if p2:
                            codes = p2.get("codes", [])
                            dedup = p2.get("dedup", {})
                            if codes:
                                codes_df = pd.DataFrame([
                                    {"Participant": c.get("participant_id",""),
                                     "Code Name":   c.get("name",""),
                                     "Description": c.get("description","")[:80]}
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
                    f"{r['model'].upper()} T={r['temperature']} {r['created_at'][:10]}":
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
                                f"Source: {run_rec.get('source_type','—')}"
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
