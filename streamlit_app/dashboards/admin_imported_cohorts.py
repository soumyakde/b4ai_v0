"""
streamlit_app/dashboards/admin_imported_cohorts.py

Admin-only panel (Data Management tab): import cohorts collected outside the platform (e.g. UB workshops
run on Qualtrics) so they can be analysed in the Teacher Dashboard next to the platform cohorts.
All data logic lives in core.analytics.datasets.imported_cohorts (no Streamlit there).
"""
import pandas as pd
import streamlit as st

from core.db_utils import DB_PATH
from core.analytics.datasets import imported_cohorts as ic


def _known_user_ids_and_cohorts():
    from auth.user_manager import get_user_cohort_map
    cmap = get_user_cohort_map()
    users = set(cmap.keys())
    try:
        import sqlite3
        with sqlite3.connect(str(DB_PATH)) as c:
            users |= {r[0] for r in c.execute("SELECT DISTINCT user_id FROM responses")}
    except Exception:
        pass
    return users, {v for v in cmap.values() if v}


def _known_instruments():
    try:
        from core.analytics.datasets.canonical_loader import _build_instruments_dict
        return set(_build_instruments_dict().keys())
    except Exception:
        return None


def _log(admin_user: str, action_name: str, details: str):
    try:
        from core.admin.audit_logger import AdminAction, log_admin_action
        log_admin_action(admin_user, AdminAction[action_name], details)
    except Exception:
        pass


def render_imported_cohorts_panel(admin_user: str) -> None:
    st.subheader("📥 Imported cohorts (outside-platform data)")
    st.caption(
        "Add cohorts collected outside the platform (e.g. UB workshops on Qualtrics) so the Teacher Dashboard can analyse them "
        "next to the platform cohorts. Imported data is stored separately from live student responses, appears in the Teacher "
        "Dashboard only when 'Include imported cohorts' is switched on, and is included in database backups. "
        "Use anonymous student IDs only (no names or initials), and only for data you are approved to analyse."
    )
    if st.session_state.get("imp_flash"):
        st.success(st.session_state.pop("imp_flash"))

    cur = ic.list_imports(DB_PATH)
    if cur.empty:
        st.info("No cohorts imported yet.")
    else:
        st.dataframe(cur, width="stretch", hide_index=True)
        with st.expander("Remove an imported cohort"):
            target = st.selectbox("Cohort to remove", cur["cohort_id"].tolist(), key="imp_del_target")
            confirm = st.text_input("Type the cohort name exactly to confirm removal", key="imp_del_confirm")
            if st.button("Remove this imported cohort", key="imp_del_btn"):
                if confirm.strip() != target:
                    st.error("The name typed does not match; nothing was removed.")
                else:
                    n = ic.delete_imported_cohort(DB_PATH, target)
                    _log(admin_user, "DELETE_IMPORTED_COHORT", f"{target}: {n} rows")
                    st.session_state["imp_flash"] = f"Removed {target} ({n} rows). Live student data was not touched."
                    st.cache_data.clear()
                    st.rerun()

    st.markdown("**Import or replace cohorts**")
    st.caption(
        "Files: (1) responses CSV in the dashboard's canonical format - columns user_id, module_id, instrument_key, question_id, "
        "response_value, item_score, construct, grade, submitted_at, completed_at, cohort_id; "
        "(2) optional demographics CSV - user_id, grade, gender, first_language_english. "
        "Importing a cohort name that already exists replaces that imported cohort."
    )
    f_resp = st.file_uploader("Responses CSV", type=["csv"], key="imp_resp")
    f_demo = st.file_uploader("Demographics CSV (optional)", type=["csv"], key="imp_demo")
    if f_resp is None:
        return
    try:
        canon = pd.read_csv(f_resp, dtype=str, keep_default_na=False)
        demo = pd.read_csv(f_demo, dtype=str, keep_default_na=False) if f_demo is not None else None
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return
    users, cohorts = _known_user_ids_and_cohorts()
    existing = set(cur["cohort_id"]) if not cur.empty else set()
    if "cohort_id" in canon.columns:
        replacing = set(canon["cohort_id"]) & existing
    else:
        replacing = set()
    res = ic.validate_import(canon, demo, users, cohorts, _known_instruments(), replacing)
    for e in res["errors"]:
        st.error(e)
    for w in res["warnings"]:
        st.warning(w)
    if not res["summary"].empty:
        st.markdown("**Preview (nothing saved yet)**")
        st.dataframe(res["summary"], width="stretch", hide_index=True)
        if replacing:
            st.info("These cohorts already exist and will be replaced: " + ", ".join(sorted(replacing)))
    if res["errors"]:
        st.stop()
    label = st.text_input("Batch label (for your records)", value=f_resp.name, key="imp_label")
    if st.button("Import into the dashboard", type="primary", key="imp_go"):
        out = ic.save_import(DB_PATH, canon, demo, label.strip() or f_resp.name)
        _log(admin_user, "IMPORT_COHORT", f"{label}: {out['cohorts']} cohort(s), {out['students']} students, {out['rows']} rows")
        st.session_state["imp_flash"] = (f"Imported {out['students']} students / {out['rows']} item responses in "
                                         f"{out['cohorts']} cohort(s). Switch on 'Include imported cohorts' in the Teacher Dashboard sidebar.")
        st.cache_data.clear()
        st.rerun()
