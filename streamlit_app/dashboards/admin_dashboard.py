
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
                    st.experimental_rerun()
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
                    st.experimental_rerun()
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
                    st.experimental_rerun()
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
        st.subheader("Scoped Instrument Reset")
        st.caption(
            "Delete responses for a specific instrument, optionally scoped to "
            "one cohort. Always preview before confirming. "
            "This cannot be undone."
        )

        import sqlite3 as _sq_rst
        from pathlib import Path as _P_rst

        # Find responses.db
        _db_rst = next(
            (p / "responses.db" for p in _P_rst(__file__).resolve().parents
             if (p / "responses.db").exists()), None
        )

        if _db_rst:
            # Get available instrument names and cohorts for selectors
            # Instruments from responses.db
            _conn_rst = _sq_rst.connect(_db_rst)
            _instruments = [
                r[0] for r in _conn_rst.execute(
                    "SELECT DISTINCT instrument_name FROM responses "
                    "ORDER BY instrument_name"
                ).fetchall()
            ]
            _conn_rst.close()

            # Cohorts from users.db (responses table has no cohort_id column)
            _db_users = next(
                (p / "users.db" for p in _P_rst(__file__).resolve().parents
                 if (p / "users.db").exists()), None
            )
            _cohorts = []
            if _db_users:
                _conn_u = _sq_rst.connect(_db_users)
                _cohorts = [
                    r[0] for r in _conn_u.execute(
                        "SELECT DISTINCT cohort_id FROM users "
                        "WHERE cohort_id IS NOT NULL ORDER BY cohort_id"
                    ).fetchall()
                ]
                _conn_u.close()

            col_i, col_c = st.columns(2)
            with col_i:
                _sel_instrument = st.selectbox(
                    "Instrument to reset",
                    options=["(select)"] + _instruments,
                    key="rst_instrument",
                )
            with col_c:
                _sel_cohort = st.selectbox(
                    "Restrict to cohort (optional)",
                    options=["All cohorts"] + _cohorts,
                    key="rst_cohort",
                )

            if _sel_instrument != "(select)":
                # Build user_id list for the selected cohort
                _cohort_users = None
                if _sel_cohort != "All cohorts" and _db_users:
                    _conn_u2 = _sq_rst.connect(_db_users)
                    _cohort_users = [
                        r[0] for r in _conn_u2.execute(
                            "SELECT username FROM users WHERE cohort_id=?",
                            (_sel_cohort,)
                        ).fetchall()
                    ]
                    _conn_u2.close()

                # Preview count
                _conn_rst2 = _sq_rst.connect(_db_rst)
                if _cohort_users is not None:
                    _placeholders = ",".join("?" * len(_cohort_users))
                    _preview_n = _conn_rst2.execute(
                        f"SELECT COUNT(*) FROM responses "
                        f"WHERE instrument_name=? AND user_id IN ({_placeholders})",
                        [_sel_instrument] + _cohort_users
                    ).fetchone()[0]
                    _preview_users_n = _conn_rst2.execute(
                        f"SELECT COUNT(DISTINCT user_id) FROM responses "
                        f"WHERE instrument_name=? AND user_id IN ({_placeholders})",
                        [_sel_instrument] + _cohort_users
                    ).fetchone()[0]
                    _sample = _conn_rst2.execute(
                        f"SELECT user_id, instrument_name, question_id, response_value "
                        f"FROM responses WHERE instrument_name=? "
                        f"AND user_id IN ({_placeholders}) LIMIT 5",
                        [_sel_instrument] + _cohort_users
                    ).fetchall()
                else:
                    _preview_n = _conn_rst2.execute(
                        "SELECT COUNT(*) FROM responses WHERE instrument_name=?",
                        (_sel_instrument,)
                    ).fetchone()[0]
                    _preview_users_n = _conn_rst2.execute(
                        "SELECT COUNT(DISTINCT user_id) FROM responses "
                        "WHERE instrument_name=?",
                        (_sel_instrument,)
                    ).fetchone()[0]
                    _sample = _conn_rst2.execute(
                        "SELECT user_id, instrument_name, question_id, response_value "
                        "FROM responses WHERE instrument_name=? LIMIT 5",
                        (_sel_instrument,)
                    ).fetchall()
                _conn_rst2.close()

                cohort_label = (
                    f" in cohort **{_sel_cohort}**"
                    if _sel_cohort != "All cohorts" else ""
                )
                st.info(
                    f"This will delete **{_preview_n} rows** across "
                    f"**{_preview_users_n} student(s)**{cohort_label}."
                )

                with st.expander("Preview rows to be deleted (first 5)",
                                 expanded=True):
                    import pandas as _pd_rst
                    st.dataframe(
                        _pd_rst.DataFrame(
                            _sample,
                            columns=["user_id","instrument_name",
                                     "question_id","response_value"]
                        ),
                        hide_index=True, width="stretch"
                    )
                    if _cohort_users is not None:
                        st.caption(
                            f"Cohort '{_sel_cohort}' contains "
                            f"{len(_cohort_users)} student(s): "
                            f"{', '.join(_cohort_users)}"
                        )

                _confirm_del = st.text_input(
                    f"Type DELETE to confirm removing {_preview_n} rows",
                    key="rst_confirm",
                )
                if st.button("⚠️ Delete Selected Responses",
                             key="rst_go", type="primary"):
                    if _confirm_del == "DELETE":
                        if _preview_n == 0:
                            st.warning("No rows match — nothing to delete.")
                        else:
                            try:
                                _conn_del = _sq_rst.connect(_db_rst)
                                if _cohort_users is not None:
                                    _ph2 = ",".join("?" * len(_cohort_users))
                                    _conn_del.execute(
                                        f"DELETE FROM responses "
                                        f"WHERE instrument_name=? "
                                        f"AND user_id IN ({_ph2})",
                                        [_sel_instrument] + _cohort_users
                                    )
                                else:
                                    _conn_del.execute(
                                        "DELETE FROM responses "
                                        "WHERE instrument_name=?",
                                        (_sel_instrument,)
                                    )
                                _conn_del.commit()
                                _conn_del.close()
                                st.success(
                                    f"Deleted {_preview_n} rows for "
                                    f"'{_sel_instrument}'"
                                    f"{cohort_label.replace('**','')}"
                                    f". Refresh the page to confirm."
                                )
                            except Exception as _e_del:
                                st.error(f"Delete failed: {_e_del}")
                    else:
                        st.warning("Type DELETE exactly to confirm.")
        else:
            st.error("responses.db not found.")

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