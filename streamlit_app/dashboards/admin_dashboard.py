
"""
streamlit_app/dashboards/admin_dashboard.py
Admin Dashboard for BasicsB4AI with DEBUG MODE toggle

Provides functionalities:
- User management (create, update role, delete, impersonate)
- Dataset maintenance
- Research operations (instrument listing, human ratings import, dataset export, metrics)
- System operations (backup, clone databases, clear cache)
- Diagnostics (system metrics, audit log)
"""

import streamlit as st
import sqlite3

from core.admin import (
    user_service,
    data_service,
    research_service,
    system_service,
    diagnostics_service
)

# ---------------------------------------------------------
# DEBUG MODE TOGGLE
# ---------------------------------------------------------
if "DEBUG_MODE" not in st.session_state:
    st.session_state.DEBUG_MODE = False

st.sidebar.checkbox("DEBUG MODE", value=st.session_state.DEBUG_MODE, key="DEBUG_MODE")

DEBUG = st.session_state.DEBUG_MODE

# ---------------------------------------------------------
# MAIN DASHBOARD FUNCTION
# ---------------------------------------------------------
def show_admin_dashboard(username: str):
    """Renders the Admin Dashboard for the given admin username."""

    st.title("⚙️ Admin Dashboard")
    st.caption(f"Administrator: {username}")
    st.divider()

    # Tabs
    tabs = st.tabs([
        "👥 User Management",
        "🗄 Data Management",
        "🔬 Research Operations",
        "🖥 System Operations",
        "📊 Diagnostics"
    ])

    # ---------------------------------------------------------
# USER MANAGEMENT
# ---------------------------------------------------------
    with tabs[0]:
        st.subheader("User Governance")

        # Fetch users
        try:
            users = user_service.get_all_users()
            st.dataframe(users, width="stretch")
        except Exception as e:
            st.error(f"Error fetching users: {e}")
            st.stop()

        st.divider()
        st.subheader("Create User")
        new_user = st.text_input("Username")
        new_role = st.selectbox("Role", ["student", "teacher", "admin"])

        # -----------------------------
        # Cohort dropdown
        # -----------------------------
        cohort_list = user_service.get_all_cohorts()  # fetch from cohorts table
        new_cohort = st.selectbox(
            "Assign Cohort",
            ["None"] + cohort_list
        )
        cohort_to_pass = None if new_cohort == "None" else new_cohort

        if st.button("Create User"):
            if new_user:
                try:
                    user_service.create_user(
                        admin_user=username,      # current admin performing the action
                        username=new_user,
                        role=new_role,
                        password="defaultpass",   # adjust if you want a password input field
                        cohort_id=cohort_to_pass
                    )
                    st.success("User created")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating user: {e}")

        st.divider()
        st.subheader("Change Role & Cohort")
        target_user = st.text_input("Username to modify")
        role_change = st.selectbox("New Role", ["student", "teacher", "admin"], key="role_change")

        # Cohort dropdown for reassignment
        cohort_list = user_service.get_all_cohorts()  # fetch from cohorts table
        selected_cohort = st.selectbox(
            "Reassign Cohort (optional)",
            ["None"] + cohort_list,
            key="cohort_change"
        )
        cohort_to_pass = None if selected_cohort == "None" else selected_cohort

        if st.button("Update Role & Cohort"):
            if target_user:
                try:
                    # Update role
                    user_service.change_role(admin_user=username, username=target_user, new_role=role_change)

                    # Update cohort (None will clear cohort assignment)
                    user_service.update_user_cohort(admin_user=username, username=target_user, new_cohort_id=cohort_to_pass)

                    st.success("Role and cohort updated")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating role/cohort: {e}")

        st.divider()
        st.subheader("Delete User")
        delete_user = st.text_input("Username to delete")
        if st.button("Delete User"):
            if delete_user:
                try:
                    user_service.delete_user(username, delete_user)
                    st.warning("User deleted")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting user: {e}")

        st.divider()
        st.subheader("Impersonation")
        impersonate_user = st.text_input("Username to impersonate")
        if st.button("Impersonate"):
            if impersonate_user:
                st.session_state.username = impersonate_user
                st.rerun()

        st.divider()
        st.subheader("Cohort Management")
        new_cohort_input = st.text_input("New Cohort ID")
        if st.button("Create Cohort"):
            if new_cohort_input:
                try:
                    conn = user_service.get_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR IGNORE INTO cohorts (cohort_id) VALUES (?)",
                        (new_cohort_input.strip(),)
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"Cohort '{new_cohort_input}' created.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating cohort: {e}")

    # ---------------------------------------------------------
    # DATA MANAGEMENT
    # ---------------------------------------------------------
    with tabs[1]:
        st.subheader("Dataset Maintenance")

        student_reset = st.text_input("Student username")
        if st.button("Reset Student Data"):
            if student_reset:
                try:
                    data_service.reset_student_data(username, student_reset)
                    st.success("Student data cleared")
                except Exception as e:
                    st.error(f"Error resetting student data: {e}")

        st.divider()
        instrument_reset = st.text_input("Instrument ID")
        if st.button("Reset Instrument Data"):
            if instrument_reset:
                try:
                    data_service.reset_instrument(username, instrument_reset)
                    st.success("Instrument data cleared")
                except Exception as e:
                    st.error(f"Error resetting instrument data: {e}")

        st.divider()
        confirm = st.text_input("Type RESET to clear all study data")
        if st.button("Reset Entire Study"):
            if confirm == "RESET":
                try:
                    data_service.reset_study(username)
                    st.warning("All research data cleared")
                except Exception as e:
                    st.error(f"Error resetting study: {e}")

    # ---------------------------------------------------------
    # RESEARCH OPERATIONS
    # ---------------------------------------------------------
    with tabs[2]:
        st.subheader("Loaded Instruments")
        try:
            instruments = research_service.get_loaded_instruments()
            st.dataframe({"Instrument": instruments}, width="stretch")
        except Exception as e:
            st.error(f"Error loading instruments: {e}")

        st.divider()
        st.subheader("Import Human Ratings")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                rows = research_service.import_human_ratings(username, uploaded_file)
                st.success(f"{rows} human ratings imported")
            except Exception as e:
                st.error(f"Error importing human ratings: {e}")

        st.divider()
        st.subheader("Export Research Dataset")
        if st.button("Export Research Dataset"):
            try:
                path = research_service.export_research_dataset(username)
                st.success(f"Dataset exported → {path}")
            except Exception as e:
                st.error(f"Error exporting research dataset: {e}")

        st.divider()
        st.subheader("Dataset Metrics")
        try:
            metrics = diagnostics_service.get_research_metrics()
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Responses", metrics.get("total_responses", 0))
            col2.metric("Completions", metrics.get("total_completions", 0))
            col3.metric("Survey Scores", metrics.get("total_survey_scores", 0))
            col4.metric("Assessment Scores", metrics.get("total_assessment_scores", 0))
            col5.metric("LLM Ratings", metrics.get("total_llm_ratings", 0))
            col6.metric("Human Ratings", metrics.get("total_human_ratings", 0))

            if DEBUG:
                st.write("DEBUG - Metrics dict:", metrics)

        except Exception as e:
            st.error(f"Error fetching research metrics: {e}")

    # ---------------------------------------------------------
    # SYSTEM OPERATIONS
    # ---------------------------------------------------------
    with tabs[3]:
        st.subheader("Database Maintenance")
        if st.button("Backup Databases"):
            try:
                backups = system_service.backup_databases(username)
                st.success("Backup created")
                st.write(backups if DEBUG else backups.keys())
            except Exception as e:
                st.error(f"Error backing up databases: {e}")

        st.divider()
        if st.button("Clone Databases"):
            try:
                clones = system_service.clone_databases(username)
                st.success("Clone created")
                st.write(clones if DEBUG else clones.keys())
            except Exception as e:
                st.error(f"Error cloning databases: {e}")

        st.divider()
        if st.button("Clear Cache"):
            try:
                system_service.clear_cache(username)
                st.success("Cache cleared")
            except Exception as e:
                st.error(f"Error clearing cache: {e}")

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------
    with tabs[4]:
        st.subheader("System Metrics")
        try:
            diag = diagnostics_service.get_full_diagnostics()
            col1, col2, col3 = st.columns(3)
            col1.metric("Users", diag.get("total_users", 0))
            col2.metric("Students", diag.get("total_students", 0))
            col3.metric("Teachers", diag.get("total_teachers", 0))

            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Responses", diag.get("total_responses", 0))
            col2.metric("Completions", diag.get("total_completions", 0))
            col3.metric("Survey Scores", diag.get("total_survey_scores", 0))
            col4.metric("Assessment Scores", diag.get("total_assessment_scores", 0))

            if DEBUG:
                st.write("DEBUG - Full diagnostics:", diag)

        except Exception as e:
            st.error(f"Error fetching diagnostics: {e}")

        st.divider()
        st.subheader("Audit Log")
        try:
            logs = diagnostics_service.get_audit_log()
            st.dataframe(logs, width="stretch")
        except Exception as e:
            st.error(f"Error fetching audit log: {e}")