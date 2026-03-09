"""
streamlit_app/dashboards/teacher_dashboard.py
Teacher Dashboard — Analytics (Architecturally Aligned)
"""

import streamlit as st
import sqlite3
import pandas as pd
import yaml
from pathlib import Path

from core.analytics.datasets.dataset_builder import DatasetBuilder
from core.analytics.quantitative.aggregation_engine import DatasetAggregator
from core.analytics.quantitative.reliability_engine import ReliabilityEngine
from core.analytics.quantitative.irt_engine import IRTEngine
from core.analytics.quantitative.competency_engine import CompetencyEngine

# ✅ QUALITATIVE LAYER (NEW)
from core.analytics.qualitative.engine import QualitativeLLMEngine
from core.analytics.qualitative.storage import QualitativeStore
from core.analytics.qualitative.contracts import LLMPromptContract

from utils.module_normalizer import normalize_module_id

# Agreement layer
from core.analytics.qualitative.agreement_engine import AgreementEngine
from core.analytics.qualitative.agreement_visualizer import (
    plot_icc_summary,
    plot_bland_altman,
    plot_construct_correlation,
)

# Hybrid analytics layer
from core.analytics.hybrid.hybrid_engine import HybridCompetencyEngine

# Report generation layer
from core.analytics.reports.learning_report_engine import LearningReportEngine

# Report export layer
from core.analytics.reports.report_exporter import LearningReportExporter


# ----------------------------------------------------------
# DATABASE PATHS
# ----------------------------------------------------------
RESPONSES_DB_PATH = Path("responses.db")
QUAL_DB_PATH = Path("qualitative_ratings.db")

# ----------------------------------------------------------
# DB LOADER
# ----------------------------------------------------------
def load_responses():
    if not RESPONSES_DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(RESPONSES_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM responses", conn)
    conn.close()
    return df

# ----------------------------------------------------------
# YAML LOADER
# ----------------------------------------------------------
def load_yaml_directory(path: Path):
    instruments = {}
    scoring = {}

    for file in path.glob("*.yaml"):

        # ----------------------------------
        # EXCLUDE qualitative prompt YAMLs
        # ----------------------------------
        if "qualitative" in file.name:
            continue

        with open(file, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))

        for data in docs:
            if not isinstance(data, dict):
                continue

            key = data.get("instrument_key")
            if not key:
                continue

            if (
                "correct_answers" in data
                or "default_scale" in data
                or data.get("scoring_type") is not None
                or data.get("type") in ["binary", "likert"]
            ):
                if "scoring_type" not in data and "type" in data:
                    data["scoring_type"] = data["type"]
                scoring[key] = data
            else:
                instruments[key] = data

    return instruments, scoring

# ----------------------------------------------------------
# STREAMLIT SAFE DF
# ----------------------------------------------------------
def sanitize_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_object_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].apply(
                lambda x: "" if x is None else str(x)
            )
        elif pd.api.types.is_integer_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype(float)
        elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
    return df_clean

# ----------------------------------------------------------
# PROMPT CONTRACT LOADER (MULTI-DOCUMENT YAML SUPPORT)
# ----------------------------------------------------------
def load_prompt_contract(yaml_path: Path, instrument_key: str) -> LLMPromptContract:
    """
    Loads the LLMPromptContract for the given instrument_key
    from a multi-document YAML file (--- separators allowed).
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    matched = None
    for doc in docs:
        if doc.get("instrument_key") == instrument_key:
            matched = doc
            break

    if matched is None:
        raise ValueError(f"No YAML config found for instrument {instrument_key}")

    return LLMPromptContract(
        theory_block=matched["theory_block"],
        constructs=matched["constructs"],
        model_name=matched["model_name"],
    )

# ----------------------------------------------------------
# DASHBOARD VIEW
# ----------------------------------------------------------
def show_teacher_dashboard(username: str):
    st.title("📊 Teacher Analytics Dashboard")
    st.caption(f"Welcome, {username}")
    st.divider()

    responses_df = load_responses()
    if responses_df.empty:
        st.info("No response data available yet.")
        return

    instruments_dict, scoring_dict = load_yaml_directory(Path("streamlit_app/surveys"))

    builder = DatasetBuilder(
        responses_df=responses_df,
        instruments_dict=instruments_dict,
        scoring_dict=scoring_dict,
        demographics_df=None,
        filter_spec=None,
    )

    canonical_df = builder.build().convert_dtypes()

    # ---------- dtype stabilization ----------
    for col in canonical_df.columns:
        if pd.api.types.is_object_dtype(canonical_df[col]):
            canonical_df[col] = canonical_df[col].astype(str).fillna("")

    for dt_col in ["submitted_at", "completed_at"]:
        if dt_col in canonical_df.columns:
            canonical_df[dt_col] = pd.to_datetime(canonical_df[dt_col], errors="coerce")

    # -----------------------------
    # DEBUG — Canonical Dataset Check
    # -----------------------------
    with st.expander("DEBUG — Canonical Dataset Check", expanded=False):
        st.write("Columns:", list(canonical_df.columns))
        st.write("DTypes:", canonical_df.dtypes)
        # Check for mixed types
        for col in canonical_df.columns:
            types_in_col = canonical_df[col].map(type).value_counts()
            if len(types_in_col) > 1:
                st.write(f"Column '{col}' has mixed types:")
                st.write(types_in_col)

    # --------------------------------------------------
    # MODULE NORMALIZATION
    # --------------------------------------------------
    canonical_df["module_key"] = canonical_df["module_id"].apply(normalize_module_id)

    # -----------------------------
    # COMPETENCY ENGINE & CPI
    # -----------------------------
    try:
        competency_engine = CompetencyEngine(
            canonical_df,
            module_column="module_key",
            dataset_hash=canonical_df.attrs.get("dataset_hash"),
        )
        st.success("✅ CompetencyEngine initialized with normalized modules")
    except Exception as e:
        st.error("❌ CompetencyEngine initialization failed")
        st.exception(e)
        return
        


    # --------------------------------------------------
    # DATASET AGGREGATOR & RELIABILITY ENGINE
    # --------------------------------------------------
    aggregator = DatasetAggregator(
        canonical_df,
        dataset_hash=canonical_df.attrs.get("dataset_hash"),
    )
    reliability_engine = ReliabilityEngine(canonical_df)

    # --------------------------------------------------
    # COMPETENCY PROGRESSION INDEX (CPI)
    # --------------------------------------------------
    st.subheader("Competency Progression Index (DEBUG)")
    try:
        cpi_df = competency_engine.competency_progression_index()
        if cpi_df.empty:
            st.info("CPI produced empty dataframe.")
        else:
            st.dataframe(sanitize_dataframe_for_streamlit(cpi_df), width="stretch")
            with st.expander("DEBUG — CPI Summary"):
                st.write("Rows:", len(cpi_df))
                st.write("Progression Levels:")
                progression_counts = cpi_df["progression_level"].value_counts()
                progression_counts_safe = progression_counts.apply(lambda x: str(x))
                st.dataframe(progression_counts_safe)
    except Exception as e:
        st.error("❌ CPI computation failed")
        st.exception(e)
        
    if "cpi_df" not in locals():
        st.error("CPI unavailable — stopping analytics pipeline.")
        return

    #-----HYBRID COMPETENCY INDEX
    st.divider()
    st.subheader("Hybrid Competency Index (CPI+)")

    #---DEBUG for schema drift
    with st.expander("DEBUG — CPI Schema", expanded=False):
        st.write("CPI Columns:", list(cpi_df.columns))
        st.write("Sample:")
        st.dataframe(cpi_df.head())

    hybrid_engine = HybridCompetencyEngine(cpi_df)

    hybrid_df = hybrid_engine.hybrid_cpi()
    
    # --------------------------------------------------
    # DEBUG — Hybrid CPI Structural Validation
    # --------------------------------------------------
    # This debug block verifies that the Hybrid CPI dataframe
    # contains the minimum schema required for reliability
    # estimation and downstream diagnostics.
    #
    # It catches the three most common silent failures:
    # 1. Missing modality columns
    # 2. Constant-score collapse (std = 0)
    # 3. Schema drift between hybrid_engine and dashboard
    # --------------------------------------------------

    with st.expander("DEBUG — Hybrid CPI Structural Check", expanded=False):

        required_cols = [
            "user_id",
            "competency",
            "cpi_quant",
            "cpi_qual",
            "cpi_hybrid"
        ]

        missing_cols = [c for c in required_cols if c not in hybrid_df.columns]

        if missing_cols:
            st.error("Hybrid CPI schema mismatch detected.")
            st.write("Missing columns:", missing_cols)
        else:
            st.success("Hybrid CPI schema validated.")

        # ---------- modality variance check ----------
        st.write("Variance Diagnostics:")

        if "cpi_quant" in hybrid_df.columns:
            st.write(
                "Quant variance:",
                hybrid_df["cpi_quant"].var()
            )

        if "cpi_qual" in hybrid_df.columns:
            st.write(
                "Qual variance:",
                hybrid_df["cpi_qual"].var()
            )

        # ---------- row alignment check ----------
        if "cpi_quant" in hybrid_df.columns and "cpi_qual" in hybrid_df.columns:

            aligned_rows = hybrid_df.dropna(
                subset=["cpi_quant", "cpi_qual"]
            )

            st.write("Aligned rows (quant + qual):", len(aligned_rows))
            st.write("Total rows:", len(hybrid_df))
    
    
    
    with st.expander("DEBUG — Hybrid Engine", expanded=False):
        st.write("CPI rows:", len(cpi_df))
        st.write("Hybrid rows:", len(hybrid_df))
        st.write("Columns:", list(hybrid_df.columns))
        st.write("Null counts:")
        st.write(hybrid_df.isna().sum())

    if hybrid_df.empty:
        st.info("Hybrid CPI not available (missing qualitative data).")
    else:
        st.dataframe(
            sanitize_dataframe_for_streamlit(hybrid_df),
            width="stretch",
        )

    # ==================================================
    # PHASE-2 VALIDATION DIAGNOSTICS
    # ==================================================
    st.divider()
    st.subheader("Hybrid CPI Validation Diagnostics")

    if hybrid_df.empty:
        st.info("Diagnostics unavailable — Hybrid CPI not computed.")
    else:

        # --------------------------------------------------
        # 1. DISTRIBUTION CHECK
        # --------------------------------------------------
        st.markdown("### CPI+ Distribution")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Mean (should ≈0)",
            round(hybrid_df["cpi_hybrid"].mean(), 3)
        )

        col2.metric(
            "Std Dev (should ≈1)",
            round(hybrid_df["cpi_hybrid"].std(), 3)
        )

        col3.metric(
            "Observations",
            len(hybrid_df)
        )

        # Histogram
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.hist(hybrid_df["cpi_hybrid"], bins=20)
        plt.title("Hybrid CPI Distribution")
        plt.xlabel("CPI+")
        plt.ylabel("Frequency")
        st.pyplot(fig)

        # --------------------------------------------------
        # 2. QUANT ↔ QUAL CORRELATION
        # --------------------------------------------------
        st.markdown("### Modality Alignment")

        aligned = hybrid_df.dropna(subset=["cpi_quant", "cpi_qual"])

        if len(aligned) > 5:

            corr = aligned["cpi_quant"].corr(aligned["cpi_qual"])

            st.metric(
                "Quant ↔ Qual Correlation",
                round(corr, 3)
            )

            # Scatter plot
            fig = plt.figure()
            plt.scatter(aligned["cpi_quant"], aligned["cpi_qual"])
            plt.xlabel("Quantitative CPI")
            plt.ylabel("Qualitative CPI")
            plt.title("Quantitative vs Qualitative Alignment")
            st.pyplot(fig)

        else:
            st.info("Not enough aligned data to compute correlation.")

        # --------------------------------------------------
        # 3. MODALITY DOMINANCE
        # --------------------------------------------------
        st.markdown("### Modality Contribution")

        aligned = hybrid_df.dropna(subset=["cpi_quant", "cpi_qual"])

        if len(aligned) > 5:

            corr_quant = aligned["cpi_hybrid"].corr(aligned["z_quant"])
            corr_qual = aligned["cpi_hybrid"].corr(aligned["z_qual"])

            contrib_df = pd.DataFrame({
                "modality": ["quantitative", "qualitative"],
                "correlation_with_hybrid": [corr_quant, corr_qual],
            })

            st.dataframe(contrib_df)

            fig = plt.figure()
            plt.bar(
                contrib_df["modality"],
                contrib_df["correlation_with_hybrid"]
            )
            plt.title("Modality Influence on CPI+")
            st.ylabel("Correlation with CPI+")
            st.pyplot(fig)

        else:
            st.info("Insufficient data to estimate modality dominance.")

    # ==================================================
    # HYBRID RELIABILITY ESTIMATION (CPI+)
    # ==================================================
    st.divider()
    st.subheader("🔬 Hybrid Reliability Estimation (CPI+)")

    if hybrid_df.empty:
        st.info("Hybrid reliability metrics unavailable — CPI+ not computed.")
    else:
        import numpy as np
        import matplotlib.pyplot as plt

        # -----------------------------
        # 0️⃣ Preview indicators
        # -----------------------------
        st.markdown("### DEBUG: Indicators Overview")
        st.dataframe(hybrid_df[["cpi_quant", "cpi_qual"]].head(10))
        st.write("Columns available:", list(hybrid_df.columns))
        st.write("Null counts per column:")
        st.write(hybrid_df[["cpi_quant", "cpi_qual"]].isna().sum())



        # -----------------------------
        # 1️⃣ Composite Reliability (approx.)
        # -----------------------------
        indicators = hybrid_df[["cpi_quant", "cpi_qual"]].dropna()
        if len(indicators) < 2:
            st.warning("Not enough indicators for composite reliability.")
        else:
            total_var = indicators.sum(axis=1).var(ddof=1)
            sum_item_vars = indicators.var(ddof=1).sum()
            composite_reliability = 1 - (sum_item_vars / total_var) if total_var != 0 else np.nan
            st.metric("Composite Reliability (CR / Omega Total Approx.)", round(composite_reliability, 3))
            st.write("DEBUG — Individual variances per indicator:")
            st.write(indicators.var(ddof=1))
            st.write("DEBUG — Total variance of sum:", total_var)


        # -----------------------------
        # 2️⃣ Omega Hierarchical (ωh) using pingouin
        # -----------------------------
        try:
            import pingouin as pg
            omega_res = pg.omega(indicators)
            st.metric("Omega Hierarchical (ωh)", round(omega_res["omega_h"], 3))
        except ImportError:
            st.info("Install `pingouin` for Omega hierarchical computation: pip install pingouin")
        except Exception as e:
            st.warning(f"Omega computation failed: {e}")

        # -----------------------------
        # 3️⃣ Multimodal Measurement Stability
        # -----------------------------
        if len(indicators) > 1:
            corr = indicators["cpi_quant"].corr(indicators["cpi_qual"])
            st.metric("Quant ↔ Qual Stability (r)", round(corr, 3))

            # Scatter plot
            fig, ax = plt.subplots()
            ax.scatter(indicators["cpi_quant"], indicators["cpi_qual"])
            ax.set_xlabel("Quantitative CPI")
            ax.set_ylabel("Qualitative CPI")
            ax.set_title("Multimodal Stability: Quant vs Qual")
            st.pyplot(fig)
        else:
            st.info("Insufficient data to compute multimodal stability.")


    ## ----- LEARNING REPORT (PHASE 3)
    st.divider()
    st.subheader("📘 Generate Learning Report")

    if st.button("Generate Learning Report"):

        report_engine = LearningReportEngine(
            cpi_df=cpi_df,
            hybrid_df=hybrid_df,
        )

        st.session_state["learning_report"] = report_engine.build()
        st.session_state["learning_report"] = report_engine.build()
        report = st.session_state["learning_report"]

        if report is None:
            st.warning("Generate report before exporting.")
            return


        # -----------------------------
        # DEBUG PANEL
        # -----------------------------
        with st.expander("DEBUG — Report Diagnostics", expanded=False):
            st.json(report.diagnostics)

        # -----------------------------
        # Cohort Summary
        # -----------------------------
        st.markdown("### Cohort Summary")

        if report.cohort_summary.empty:
            st.info("No cohort summary available.")
        else:
            st.dataframe(
                sanitize_dataframe_for_streamlit(
                    report.cohort_summary
                ),
                width="stretch",
            )

        # -----------------------------
        # Competency Ranking
        # -----------------------------
        st.markdown("### Competency Ranking")

        if report.competency_summary.empty:
            st.info("No competency ranking available.")
        else:
            st.dataframe(
                sanitize_dataframe_for_streamlit(
                    report.competency_summary
                ),
                width="stretch",
            )

        # -----------------------------
        # Quant ↔ Qual Alignment
        # -----------------------------
        st.markdown("### Modality Alignment (Quant vs Qual)")

        if report.modality_alignment.empty:
            st.info("Insufficient overlap to compute correlations.")
        else:
            st.dataframe(
                sanitize_dataframe_for_streamlit(
                    report.modality_alignment
                ),
                width="stretch",
            )

    #----Export reports
    st.markdown("### Export Report")

    if st.button("Export Publication Report"):

        report = st.session_state.get("learning_report")

        if report is None:
            st.warning("Generate report before exporting.")
            return

        exporter = LearningReportExporter(report)
        outputs = exporter.export_all()

        with st.expander("DEBUG — Export Outputs", expanded=False):
            st.json(outputs)

        st.success("Report exported successfully.")


    # --------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Responses", len(canonical_df))
    col2.metric("Unique Students", canonical_df["user_id"].nunique())
    col3.metric("Instruments Used", canonical_df["instrument_key"].nunique())
    st.divider()

    # --------------------------------------------------
    # INSTRUMENT METRICS
    # --------------------------------------------------
    st.subheader("Mean Scores per Instrument")
    st.dataframe(sanitize_dataframe_for_streamlit(aggregator.instrument_level_metrics()), width="stretch")
    st.divider()

    # --------------------------------------------------
    # MODULE METRICS
    # --------------------------------------------------
    st.subheader("Mean Scores per Module")
    st.dataframe(sanitize_dataframe_for_streamlit(aggregator.module_level_metrics()), width="stretch")
    st.divider()

    # --------------------------------------------------
    # RELIABILITY
    # --------------------------------------------------
    st.subheader("Reliability (Cronbach's Alpha)")
    reliability = reliability_engine.reliability_per_instrument()
    if reliability.empty:
        st.info("No instruments with sufficient items.")
    else:
        st.dataframe(sanitize_dataframe_for_streamlit(reliability), width="stretch")
    st.divider()

    # --------------------------------------------------
    # STUDENT ACTIVITY
    # --------------------------------------------------
    st.subheader("Student Activity")
    st.dataframe(sanitize_dataframe_for_streamlit(aggregator.student_activity()), width="stretch")
    st.divider()

    # ==================================================
    # ✅ QUALITATIVE CODING (YAML-DRIVEN)
    # ==================================================
    st.subheader("Qualitative Coding (Experimental)")
    st.caption("Runs LLM coding on reflection responses and stores ratings in SQLite.")

    if st.button("Run LLM Coding on Reflections"):
        try:
            with st.spinner("Running qualitative coding..."):
                reflections = canonical_df[canonical_df["instrument_key"] == "module_reflections"]

                if reflections.empty:
                    st.warning("No reflection responses found.")
                else:
                    yaml_path = Path("streamlit_app/surveys/qualitative_prompts.yaml")
                    contract = load_prompt_contract(yaml_path, instrument_key="module_reflections")

                    llm_client = st.session_state.get("llm_client")
                    if llm_client is None:
                        st.error("LLM client not configured in session_state.")
                        return

                    engine = QualitativeLLMEngine(api_key=llm_client.api_key, contract=contract)

                    rows = []
                    for _, row in reflections.iterrows():
                        rows.append({
                            "student_id": str(row["user_id"]),
                            "module_id": str(row["module_id"]),
                            "question_id": str(row["question_id"]),
                            "question_text": row.get("question_text", ""),
                            "response_text": row.get("response_text", ""),
                        })

                    ratings = engine.rate_batch(rows)
                    store = QualitativeStore(db_path=QUAL_DB_PATH)
                    store.save(ratings)
                    st.success(f"✅ Stored {len(ratings)} qualitative ratings.")

        except Exception as e:
            st.error("Qualitative coding failed.")
            st.exception(e)

    st.divider()


    # ==================================================
    # AGREEMENT ANALYTICS
    # ==================================================
    st.divider()
    st.subheader("📊 Agreement Analytics (Human ↔ LLM)")

    if st.button("Compute Agreement Metrics"):

        try:
            with st.spinner("Computing agreement analytics..."):

                agreement_engine = AgreementEngine()

                agreement_df = agreement_engine.compute_agreement()
                aligned_df = agreement_engine.build_alignment()

                if agreement_df.empty:
                    st.warning("No aligned human + LLM ratings found.")
                else:
                    st.markdown("### Agreement Summary")
                    st.dataframe(
                        sanitize_dataframe_for_streamlit(agreement_df),
                        width="stretch",
                    )

                    # -----------------------------------------
                    # ICC BAR CHART
                    # -----------------------------------------
                    fig_icc = plot_icc_summary(agreement_df)
                    if fig_icc:
                        st.pyplot(fig_icc)

                    # -----------------------------------------
                    # HEATMAP
                    # -----------------------------------------
                    fig_heat = plot_construct_correlation(aligned_df)
                    if fig_heat:
                        st.pyplot(fig_heat)

                    # -----------------------------------------
                    # BLAND–ALTMAN SELECTOR
                    # -----------------------------------------
                    st.markdown("### Bland–Altman Diagnostics")

                    construct = st.selectbox(
                        "Select Construct",
                        agreement_df["construct"].tolist(),
                    )

                    fig_ba = plot_bland_altman(aligned_df, construct)
                    if fig_ba:
                        st.pyplot(fig_ba)

        except Exception as e:
            st.error("Agreement analytics failed.")
            st.exception(e)



    # --------------------------------------------------
    # IRT
    # --------------------------------------------------
    st.subheader("Item Response Theory")
    irt_engine = IRTEngine(canonical_df)
    eligible = irt_engine.get_eligible_instruments()
    if not eligible:
        st.info("No instruments eligible for IRT.")
        return

    selected = st.selectbox("Select Instrument for IRT", eligible)

    # DEBUG — IRT DATA INSPECTION
    with st.expander("DEBUG — IRT Dataset"):
        st.write("Selected instrument:", selected)
        irt_subset = canonical_df[canonical_df["instrument_key"] == selected]
        st.write("Rows:", len(irt_subset))
        st.write("Unique students:", irt_subset["user_id"].nunique())
        st.write("Unique items:", irt_subset["question_id"].nunique())
        st.write("Score distribution:")
        st.write(irt_subset["item_score"].value_counts(dropna=False))

    if irt_subset["user_id"].nunique() < 20:
        st.warning("IRT requires ≥20 students. Current dataset too small for stable estimation.")
    else:
        results = irt_engine.run(selected)
        if results:
            st.markdown("### Item Parameters")
            st.dataframe(sanitize_dataframe_for_streamlit(results["item_parameters"]), width="stretch")
            st.markdown("### Student Ability Estimates")
            st.dataframe(sanitize_dataframe_for_streamlit(results["person_parameters"]), width="stretch")

    # DEBUG — Module Alignment Check
    with st.expander("DEBUG — Module Alignment Check"):
        st.write("Dataset module_ids:")
        st.write(sorted(canonical_df["module_id"].unique()))
        canonical_df["module_key"] = canonical_df["module_id"].apply(normalize_module_id)
        st.write("Ontology module keys:")
        st.write(sorted(competency_engine.module_to_competencies.keys()))