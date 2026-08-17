"""
streamlit_app/dashboards/admin_dashboard.py
Admin Dashboard for Basics4AI

Changes v3:
  - All use_container_width replaced with width='stretch'
"""

import os
import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

from core.admin import (
    user_service,
    data_service,
    research_service,
    system_service,
    diagnostics_service,
)
from auth.user_manager import (
    is_super_admin,
    get_pending_users,
    approve_user,
    reject_user,
    bulk_approve,
)
from core.analytics.llm_analysis import (
    PROVIDER_PRICING,   # kept for cost display in transcript store if needed
)

# Transcript store — graceful fallback if LLM modules not installed
try:
    from core.analytics.llm.transcript_store import (
        get_persistent_transcripts,
        get_transcript_count,
        delete_transcript,
        delete_all_transcripts_for_participant,
        load_for_analysis,
    )
    _TRANSCRIPT_STORE_AVAILABLE = True
except Exception as _ts_err:
    _TRANSCRIPT_STORE_AVAILABLE = False
    _TRANSCRIPT_STORE_ERR = str(_ts_err)

# ---------------------------------------------------------
# COHORT SELECTOR — shared by both transcript upload sections
# ---------------------------------------------------------

_COHORT_NONE = "— none —"
_COHORT_CREATE_NEW = "+ Create new cohort…"


def _cohort_selector(key_prefix: str, label: str = "Cohort") -> str | None:
    """
    Render a cohort-picker selectbox with an inline "create new cohort"
    escape hatch. Returns the resolved cohort_id, or None if "— none —".

    The "create new" path calls user_service.add_cohort() — the same
    function the existing Cohort Management section (User Management tab)
    uses — so a cohort created here is immediately usable everywhere else
    in the app too, not a parallel/duplicate mechanism.
    """
    # A st.success() called right before st.rerun() never actually reaches
    # the screen -- the rerun happens before the browser paints it. Stash
    # the message in session_state instead and show it on the NEXT render,
    # right before the widget it's about (fixed 2026-08-08).
    _flash_key = f"{key_prefix}_flash"
    if st.session_state.get(_flash_key):
        st.success(st.session_state.pop(_flash_key))

    # Streamlit forbids writing to st.session_state[key] for a widget's OWN
    # key in the same run the widget already rendered -- raises
    # StreamlitAPIException ("cannot be modified after the widget ... is
    # instantiated"), even if a st.rerun() follows right after. So the
    # "pre-select the newly created cohort" write can't happen inside the
    # button handler below (that's after st.selectbox() already ran this
    # pass) -- it has to be applied here, BEFORE st.selectbox() is called,
    # using a separate flag set on the previous run (fixed 2026-08-09,
    # crash found during the user's E.1 re-test).
    _pending_key = f"{key_prefix}_pending_select"
    if st.session_state.get(_pending_key):
        st.session_state[f"{key_prefix}_choice"] = st.session_state.pop(_pending_key)

    cohorts = user_service.get_all_cohorts()
    options = [_COHORT_NONE] + cohorts + [_COHORT_CREATE_NEW]
    choice = st.selectbox(label, options=options, key=f"{key_prefix}_choice")

    if choice == _COHORT_CREATE_NEW:
        new_id = st.text_input("New cohort ID", key=f"{key_prefix}_new_id")
        if st.button("Create cohort", key=f"{key_prefix}_new_btn"):
            new_id = new_id.strip()
            if new_id:
                user_service.add_cohort(new_id)
                st.session_state[_flash_key] = f"Cohort '{new_id}' created and selected below."
                # Deferred pre-select -- applied at the top of this function
                # on the next render, not here (see comment above).
                st.session_state[_pending_key] = new_id
                st.rerun()
            else:
                st.warning("Enter a cohort ID first.")
        return None  # not yet selected — admin must pick it after creating

    if choice == _COHORT_NONE:
        return None
    return choice


# ---------------------------------------------------------
# MAIN DASHBOARD FUNCTION
# ---------------------------------------------------------
def show_admin_dashboard(username: str):
    """Renders the Admin Dashboard."""

    # DEBUG MODE TOGGLE -- must live inside this function, not at module
    # level. A Streamlit server shares one Python process (and one
    # sys.modules cache) across every connected session; a widget placed at
    # true module level in an *imported* file only ever executes once, the
    # first time any session triggers the import after a deploy/restart --
    # every other session (or even the same admin's next rerun) simply
    # never sees it. Found live: the checkbox appeared "sometimes, seemingly
    # tied to a restart" rather than reliably on every page load.
    if "DEBUG_MODE" not in st.session_state:
        st.session_state.DEBUG_MODE = False
    st.sidebar.checkbox("DEBUG MODE", value=st.session_state.DEBUG_MODE, key="DEBUG_MODE")
    DEBUG = st.session_state.DEBUG_MODE

    st.title("⚙️ Admin Dashboard")
    st.caption(f"Administrator: {username}")
    st.divider()

    _is_sa = is_super_admin(username)
    tab_labels = [
        "👥 User Management",
        "🗄 Data Management",
        "🔬 Research Operations",
        "🖥 System Operations",
        "📊 Diagnostics",
    ]
    if _is_sa:
        tab_labels.insert(0, "🔐 Pending Approvals")

    tabs = st.tabs(tab_labels)
    tab_offset = 1 if _is_sa else 0

    # ---------------------------------------------------------
    # Added the lines below on 3-29-26 for Rather than making the admin dashboard write the setting 
    #(which would require a restart to take effect anyway since load_dotenv() only runs at boot), 
    # add a read-only status indicator in admin_dashboard.py. This keeps the admin informed without 
    # over-engineering a runtime toggle.
    # ---------------------------------------------------------
    from utils.quiz_mode import get_quiz_mode
    st.info(f"📋 Active Quiz Mode: **{get_quiz_mode().upper()}**  — change via `QUIZ_MODE` in `.env`, then restart.")
    #
    # ---------------------------------------------------------
    # PENDING APPROVALS (super admin only)
    # ---------------------------------------------------------
    if _is_sa:
        with tabs[0]:
            st.subheader("Pending Registration Approvals")
            # A st.success()/st.warning() called right before st.rerun() never
            # reaches the screen -- the rerun happens before the browser
            # paints it. Same fix already used for cohort_mgmt_flash /
            # admin_obs_upload_flash elsewhere in this file.
            if st.session_state.get("pending_approvals_flash"):
                st.success(st.session_state.pop("pending_approvals_flash"))
            if st.session_state.get("pending_approvals_flash_warn"):
                st.warning(st.session_state.pop("pending_approvals_flash_warn"))
            pending = get_pending_users()
            if not pending:
                st.success("✅ No pending registrations.")
            else:
                st.warning(f"{len(pending)} user(s) awaiting approval.")
                df = pd.DataFrame(pending)
                st.dataframe(df, width="stretch")

                st.divider()
                st.markdown("**Approve or reject individual users**")
                for p in pending:
                    u = p["username"]
                    col_u, col_a, col_r = st.columns([3, 1, 1])
                    col_u.write(f"**{u}** ({p['role']})")
                    if col_a.button("Approve", key=f"approve_{u}"):
                        try:
                            approve_user(username, u)
                            st.session_state["pending_approvals_flash"] = f"{u} approved."
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                    if col_r.button("Reject", key=f"reject_{u}"):
                        try:
                            reject_user(username, u)
                            st.session_state["pending_approvals_flash_warn"] = f"{u} rejected."
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

                st.divider()
                if st.button("✅ Approve ALL pending"):
                    try:
                        n = bulk_approve(username, [p["username"] for p in pending])
                        st.session_state["pending_approvals_flash"] = f"{n} user(s) approved."
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    # ---------------------------------------------------------
    # USER MANAGEMENT
    # ---------------------------------------------------------
    with tabs[tab_offset + 0]:
        st.subheader("User Governance")
        try:
            users = user_service.get_all_users()
            _status_options = ["All"] + sorted(users["status"].dropna().unique().tolist())
            _status_filter = st.selectbox(
                "Filter by status", options=_status_options, key="user_gov_status_filter",
            )
            _users_display = users if _status_filter == "All" else users[users["status"] == _status_filter]
            st.dataframe(_users_display, width="stretch")
        except Exception as e:
            st.error(f"Error fetching users: {e}")
            st.stop()

        st.divider()
        st.subheader("Create User")
        new_user    = st.text_input("Username")
        new_role    = st.selectbox("Role", ["student", "teacher", "admin"])
        cohort_list = user_service.get_all_cohorts()
        new_cohort  = st.selectbox("Assign Cohort", ["None"] + cohort_list)
        cohort_to_pass = None if new_cohort == "None" else new_cohort
        new_status  = "approved"  # admin panel creation is always pre-approved

        # Show generated password persistently until admin acknowledges
        if st.session_state.get("_new_user_pw"):
            st.success(
                f"✅ User **{st.session_state['_new_user_name']}** created "
                f"(status: {st.session_state['_new_user_status']})"
            )
            st.info(
                f"🔑 Generated password: `{st.session_state['_new_user_pw']}`  \n"
                "Share this with the user — it will **not** be shown again."
            )
            if st.button("✅ I've noted the password", key="pw_ack"):
                del st.session_state["_new_user_pw"]
                del st.session_state["_new_user_name"]
                del st.session_state["_new_user_status"]
                st.rerun()

        if st.button("Create User"):
            if new_user:
                try:
                    generated_pw = user_service.create_user(
                        admin_user=username,
                        username=new_user,
                        role=new_role,
                        cohort_id=cohort_to_pass,
                        status=new_status,
                    )
                    st.session_state["_new_user_pw"]     = generated_pw
                    st.session_state["_new_user_name"]   = new_user
                    st.session_state["_new_user_status"] = new_status
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating user: {e}")

        st.divider()
        st.subheader("Change Role & Cohort")
        target_user     = st.text_input("Username to modify")
        role_change     = st.selectbox("New Role", ["student", "teacher", "admin"], key="role_change")
        cohort_list2    = user_service.get_all_cohorts()
        selected_cohort = st.selectbox("Reassign Cohort (optional)", ["None"] + cohort_list2, key="cohort_change")
        cohort_to_pass2 = None if selected_cohort == "None" else selected_cohort

        if st.button("Update Role & Cohort"):
            if target_user:
                try:
                    user_service.change_role(admin_user=username, username=target_user, new_role=role_change)
                    user_service.update_user_cohort(admin_user=username, username=target_user, new_cohort_id=cohort_to_pass2)
                    st.success("Role and cohort updated")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.subheader("Delete User")
        st.caption(
            "⚠️ **Hard delete** — removes the account AND all research data "
            "(responses, completions, scores) for this user. "
            "This cannot be undone."
        )
        # Same swallowed-flash-message bug as Pending Approvals (M.14) --
        # st.success() called right before st.rerun() never reaches the
        # screen. Same session-state flash fix.
        if st.session_state.get("del_user_flash"):
            st.success(st.session_state.pop("del_user_flash"))

        # The "Username to delete"/"confirm" boxes kept showing the just-
        # deleted username afterward, since a text_input's value persists
        # under its key across reruns unless the key itself changes. Same
        # generation-counter technique already used elsewhere in this file
        # to force a fresh file_uploader after a batch completes -- bumping
        # "del_user_gen" gives both boxes a brand-new key next run, so
        # there's no stale value left to redisplay.
        st.session_state.setdefault("del_user_gen", 0)
        _du_gen = st.session_state["del_user_gen"]
        delete_user_input = st.text_input(
            "Username to delete", key=f"del_user_input_{_du_gen}"
        )

        # Live data preview before deletion
        if delete_user_input and delete_user_input.strip():
            _du_prev = delete_user_input.strip()
            try:
                from core.admin.data_service import count_student_data_footprint as _du_footprint
                _fp = _du_footprint(_du_prev)
                _fp_msg = (
                    f"User **{_du_prev}**: "
                    f"{_fp['responses']} response rows · "
                    f"{_fp['completions']} completion records · "
                    f"{_fp['survey_scores']} survey score(s) · "
                    f"{_fp['assessment_scores']} assessment score(s)"
                )
                if "transcripts" in _fp:
                    _fp_msg += f" · {_fp['transcripts']} transcript(s)"
                _fp_msg += " will also be deleted."
                st.info(_fp_msg)
            except Exception:
                pass

        _del_confirm = st.text_input(
            "Type the username again to confirm deletion",
            key=f"del_user_confirm_{_du_gen}",
        )
        if st.button("🗑 Delete User + All Data", key="del_user_btn", type="primary"):
            _du = delete_user_input.strip() if delete_user_input else ""
            if not _du:
                st.warning("Enter a username.")
            elif _del_confirm.strip() != _du:
                st.error("Confirmation username does not match. Nothing was deleted.")
            else:
                try:
                    from core.admin.data_service import reset_student_data
                    reset_student_data(username, _du)
                    user_service.delete_user(username, _du)
                    if _TRANSCRIPT_STORE_AVAILABLE:
                        try:
                            delete_all_transcripts_for_participant(_du)
                        except Exception:
                            pass
                    st.session_state["del_user_flash"] = (
                        f"✅ User **{_du}** and all associated research data deleted."
                    )
                    st.session_state["del_user_gen"] += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete error: {e}")

        st.divider()
        st.subheader("🧹 Bulk Cleanup")
        st.caption(
            "Bulk-delete withdrawn/duplicate **student** registrations — either "
            "rejected registrations, or student accounts with zero saved data "
            "(e.g. abandoned duplicate sign-ups in community settings where "
            "students register multiple times without waiting for approval). "
            "Restricted to student accounts only — teacher/admin accounts are "
            "never eligible here, use \"Delete User\" above for those. Same "
            "hard delete as above, applied to multiple accounts: review the "
            "list, export it, then confirm."
        )

        from core.admin.data_service import count_student_data_footprint as _bc_footprint

        _bc_source = st.radio(
            "Candidate source",
            options=["Rejected students", "Students with zero saved data"],
            horizontal=True,
            key="bulk_cleanup_source",
        )

        try:
            _bc_all_users = user_service.get_all_users()
            _bc_students = _bc_all_users[_bc_all_users["role"] == "student"]
        except Exception as e:
            _bc_students = None
            st.error(f"Could not load users: {e}")

        _bc_candidates = []
        if _bc_students is not None:
            if _bc_source == "Rejected students":
                _bc_candidates = sorted(
                    _bc_students[_bc_students["status"] == "rejected"]["username"].tolist()
                )
            else:
                with st.spinner("Scanning student accounts for zero saved data…"):
                    for _u in _bc_students["username"].tolist():
                        if sum(_bc_footprint(_u).values()) == 0:
                            _bc_candidates.append(_u)
                _bc_candidates = sorted(_bc_candidates)

        if not _bc_candidates:
            st.info("No candidates found for this source.")
        else:
            st.write(f"**{len(_bc_candidates)} candidate(s) found.**")

            _bc_selected = [
                _cand for _cand in _bc_candidates
                if st.checkbox(_cand, key=f"bulk_cleanup_cb_{_bc_source}_{_cand}")
            ]

            if _bc_selected:
                st.write(f"**{len(_bc_selected)} selected.**")

                import pandas as _bc_pd
                _bc_preview_rows = []
                for _u in _bc_selected:
                    _fp = _bc_footprint(_u)
                    _bc_row = {
                        "Username": _u,
                        "Responses": _fp["responses"],
                        "Completions": _fp["completions"],
                        "Survey Scores": _fp["survey_scores"],
                        "Assessment Scores": _fp["assessment_scores"],
                    }
                    if "transcripts" in _fp:
                        _bc_row["Transcripts"] = _fp["transcripts"]
                    _bc_preview_rows.append(_bc_row)
                _bc_preview_df = _bc_pd.DataFrame(_bc_preview_rows)
                st.dataframe(_bc_preview_df, hide_index=True, width="stretch")

                import io as _bc_io
                _bc_csv_buf = _bc_io.StringIO()
                _bc_preview_df.to_csv(_bc_csv_buf, index=False)
                _bc_export_clicked = st.download_button(
                    "📥 Export selected list to CSV (required before deletion unlocks)",
                    data=_bc_csv_buf.getvalue(),
                    file_name=f"bulk_cleanup_candidates_{_bc_source.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"bulk_cleanup_export_{_bc_source}",
                )

                if _bc_export_clicked:
                    st.session_state["bulk_cleanup_exported"] = True

                if not st.session_state.get("bulk_cleanup_exported", False):
                    st.warning("Export the list above before the delete button unlocks.")
                else:
                    _bc_confirm_phrase = st.text_input(
                        "Type DELETE to confirm bulk deletion",
                        key="bulk_cleanup_confirm_phrase",
                    )
                    _bc_armed = _bc_confirm_phrase.strip() == "DELETE"
                    if st.button(
                        f"🗑 Delete {len(_bc_selected)} Selected Student(s) + All Data",
                        key="bulk_cleanup_delete_btn",
                        type="primary",
                        disabled=not _bc_armed,
                    ):
                        from core.admin.data_service import reset_student_data as _bc_reset
                        _bc_deleted, _bc_failed = [], []
                        for _u in _bc_selected:
                            try:
                                _bc_reset(username, _u)
                                user_service.delete_user(username, _u)
                                if _TRANSCRIPT_STORE_AVAILABLE:
                                    try:
                                        delete_all_transcripts_for_participant(_u)
                                    except Exception:
                                        pass
                                _bc_deleted.append(_u)
                            except Exception as _bc_err:
                                _bc_failed.append(f"{_u} ({_bc_err})")
                        st.session_state["bulk_cleanup_exported"] = False
                        if _bc_deleted:
                            st.success(
                                f"✅ Deleted {len(_bc_deleted)} student(s): "
                                f"{', '.join(_bc_deleted)}"
                            )
                        if _bc_failed:
                            st.error(
                                f"❌ Failed to delete {len(_bc_failed)} student(s): "
                                f"{'; '.join(_bc_failed)}"
                            )
                        st.rerun()

        st.divider()
        st.subheader("Reset Password")
        st.caption(
            "Generates a new random password for the selected user. "
            "Show it to the user once — it is not stored anywhere after this page refreshes."
        )

        # Build student/teacher user list for selector
        try:
            _all_users_df = user_service.get_all_users()
            _reset_candidates = _all_users_df[
                _all_users_df["role"].isin(["student", "teacher", "admin"])
            ]["username"].tolist()
        except Exception:
            _reset_candidates = []

        col_reset_u, col_reset_btn = st.columns([3, 1])
        with col_reset_u:
            if _reset_candidates:
                _reset_target = st.selectbox(
                    "Select user",
                    options=["(select)"] + sorted(_reset_candidates),
                    key="pw_reset_target",
                )
            else:
                _reset_target = st.text_input(
                    "Username to reset password for",
                    key="pw_reset_target_txt",
                )
        with col_reset_btn:
            st.markdown("<br>", unsafe_allow_html=True)  # vertical align
            _do_reset = st.button(
                "🔑 Reset Password",
                key="pw_reset_btn",
                type="primary",
            )

        if _do_reset:
            _tgt = _reset_target if isinstance(_reset_target, str) else ""
            _tgt = _tgt.strip()
            if not _tgt or _tgt == "(select)":
                st.warning("Select a user first.")
            else:
                try:
                    from auth.user_manager import reset_password as _reset_pw
                    _new_pw = _reset_pw(username, _tgt)
                    if _new_pw:
                        st.success(f"✅ Password reset for **{_tgt}**.")
                        st.markdown("**🔑 Temporary password — copy it now:**")
                        # text_input with value= makes it easy to select-all and copy
                        # without any surrounding markdown or invisible characters
                        st.text_input(
                            "New password (click field, Ctrl+A, Ctrl+C)",
                            value=_new_pw,
                            key="pw_reset_display",
                            help="This is the exact password. Select all and copy.",
                        )
                        st.warning(
                            "This password is shown **once only**. "
                            "Navigate away or refresh and it is gone."
                        )
                        # Verify the hash round-trip to catch any DB mismatch early
                        try:
                            from auth.user_manager import authenticate_user as _auth_verify
                            _verified = _auth_verify(_tgt, _new_pw)
                            if _verified:
                                st.caption("✅ Verified: new password authenticates correctly.")
                            else:
                                st.error(
                                    "⚠️ Hash verification failed — the stored password "
                                    "did not match. Try resetting again or check DB path."
                                )
                        except Exception as _ve:
                            st.caption(f"Verification skipped: {_ve}")
                        # Audit log
                        try:
                            from core.admin.audit_logger import (
                                log_admin_action, AdminAction
                            )
                            log_admin_action(
                                username,
                                AdminAction.PASSWORD_RESET,
                                f"target_user={_tgt}",
                            )
                        except Exception:
                            pass
                    else:
                        st.error(
                            "Reset failed. Only admins and super-admins can reset passwords."
                        )
                except ImportError:
                    st.error("user_manager.reset_password not available.")
                except Exception as _pw_err:
                    st.error(f"Reset error: {_pw_err}")

        st.divider()
        st.subheader("🔒 Login Lockout Management")
        st.caption(
            "Participants (especially younger ones) can lose several minutes of "
            "precious in-person session time after a few mistyped passwords. "
            "Use these controls to clear a specific lockout on the spot, or "
            "pause lockout enforcement entirely for a time-boxed session."
        )

        from auth import login_security as _login_sec

        _lockout_col1, _lockout_col2 = st.columns(2)
        with _lockout_col1:
            _pilot_mode_on = st.checkbox(
                "⏸️ Pause lockout for this session (pilot mode)",
                value=not _login_sec.is_lockout_enabled(),
                key="pilot_lockout_toggle",
                help=(
                    "While checked, no one can be locked out for failed login "
                    "attempts, platform-wide. Resets to OFF (lockout enabled — "
                    "the safe default) automatically on every app restart, so "
                    "it never stays paused unintentionally."
                ),
            )
            _desired_enabled = not _pilot_mode_on
            if _desired_enabled != _login_sec.is_lockout_enabled():
                _login_sec.set_lockout_enabled(_desired_enabled)
                try:
                    from core.admin.audit_logger import log_admin_action, AdminAction
                    log_admin_action(
                        username,
                        AdminAction.TOGGLE_LOCKOUT_MODE,
                        f"lockout_enabled={_desired_enabled}",
                    )
                except Exception:
                    pass
                st.rerun()
            st.caption(
                "🔓 Lockout is currently **paused platform-wide**." if _pilot_mode_on
                else "🔒 Lockout is currently **active** (normal, safe default)."
            )

        with _lockout_col2:
            _clear_target = st.text_input(
                "Username to clear lockout for", key="clear_lockout_target"
            )
            if _clear_target.strip():
                _is_locked = _login_sec.is_locked_out(_clear_target.strip())
                st.caption(
                    f"🔒 **{_clear_target.strip()}** is currently locked out."
                    if _is_locked else
                    f"🔓 **{_clear_target.strip()}** is not currently locked out."
                )
            if st.button("Clear Lockout for This User", key="clear_lockout_btn"):
                if not _clear_target.strip():
                    st.warning("Enter a username first.")
                else:
                    _tgt2 = _clear_target.strip()
                    _n_cleared = _login_sec.clear_lockout(_tgt2)
                    if _n_cleared:
                        st.success(
                            f"✅ Cleared {_n_cleared} failed-attempt record(s) for "
                            f"**{_tgt2}** — they can log in immediately."
                        )
                    else:
                        st.info(f"**{_tgt2}** had no active lockout to clear.")
                    try:
                        from core.admin.audit_logger import log_admin_action, AdminAction
                        log_admin_action(
                            username,
                            AdminAction.CLEAR_LOGIN_LOCKOUT,
                            f"target_user={_tgt2}, records_cleared={_n_cleared}",
                        )
                    except Exception:
                        pass

        st.divider()
        st.subheader("Impersonation")
        impersonate_user = st.text_input("Username to impersonate")
        if st.button("Impersonate"):
            if impersonate_user:
                try:
                    from core.admin.audit_logger import log_admin_action, AdminAction
                    log_admin_action(
                        username,
                        AdminAction.IMPERSONATE_USER,
                        f"target_user={impersonate_user}",
                    )
                except Exception:
                    pass
                st.session_state.username = impersonate_user
                st.rerun()

        st.divider()
        st.subheader("Cohort Management")

        # Same flash-message fix as _cohort_selector() -- a st.success()
        # called right before st.rerun() never reaches the screen, since
        # the rerun happens before the browser paints it (fixed 2026-08-09).
        if st.session_state.get("cohort_mgmt_flash"):
            st.success(st.session_state.pop("cohort_mgmt_flash"))

        new_cohort_input = st.text_input("New Cohort ID")
        if st.button("Create Cohort"):
            if new_cohort_input:
                try:
                    user_service.add_cohort(new_cohort_input)
                    st.session_state["cohort_mgmt_flash"] = f"Cohort '{new_cohort_input}' created."
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("**Rename a cohort**")
        st.caption(
            "Updates the registry entry and every real reference to it "
            "(users, transcripts) in place — no delete+recreate, so "
            "nothing gets orphaned along the way."
        )
        _rename_cohorts = user_service.get_all_cohorts()
        if _rename_cohorts:
            # Fresh widget keys after a successful rename -- the old
            # cohort_id disappears from _rename_cohorts on the next render,
            # and a selectbox/text_input whose session_state value is no
            # longer in its options list is the same stale-widget-key bug
            # class already fixed elsewhere in this file (e.g. the
            # bulk-cohort-selector generation counter). Bumping the key
            # generation forces brand-new, blank widgets instead.
            st.session_state.setdefault("cohort_mgmt_rename_gen", 0)
            _rn_gen = st.session_state["cohort_mgmt_rename_gen"]
            _rename_from = st.selectbox(
                "Cohort to rename", options=["— select —"] + _rename_cohorts,
                key=f"cohort_mgmt_rename_from_{_rn_gen}",
            )
            if _rename_from != "— select —":
                _rename_to = st.text_input(
                    f"New ID for '{_rename_from}'", key=f"cohort_mgmt_rename_to_{_rn_gen}",
                )
                if st.button(f"✏️ Rename '{_rename_from}'", key="cohort_mgmt_rename_btn"):
                    _rn_result = user_service.rename_cohort(_rename_from, _rename_to)
                    if _rn_result["renamed"]:
                        st.session_state["cohort_mgmt_flash"] = (
                            f"Cohort '{_rename_from}' renamed to "
                            f"'{_rename_to.strip()}' — {_rn_result['users']} user(s), "
                            f"{_rn_result['transcripts']} transcript(s) updated."
                        )
                        st.session_state["cohort_mgmt_rename_gen"] += 1
                        st.rerun()
                    else:
                        st.warning(_rn_result["reason"])
        else:
            st.caption("No cohorts registered yet.")

        st.markdown("**Delete a cohort**")
        st.caption(
            "Only allowed when nothing currently references the cohort — "
            "reassign or clear any users/transcripts tagged with it first, "
            "so deleting it never silently orphans their data."
        )
        _existing_cohorts = user_service.get_all_cohorts()
        if _existing_cohorts:
            _del_cohort_choice = st.selectbox(
                "Cohort to delete", options=["— select —"] + _existing_cohorts,
                key="cohort_mgmt_delete_choice",
            )
            if _del_cohort_choice != "— select —":
                _usage = user_service.count_cohort_usage(_del_cohort_choice)
                if _usage["users"] or _usage["transcripts"]:
                    st.warning(
                        f"⚠️ Can't delete '{_del_cohort_choice}' — still in use: "
                        f"{_usage['users']} user(s), {_usage['transcripts']} transcript(s). "
                        "Reassign/clear those first."
                    )
                else:
                    st.caption(f"'{_del_cohort_choice}' is unused — safe to delete.")
                    if st.button(f"🗑 Delete '{_del_cohort_choice}'", key="cohort_mgmt_delete_btn"):
                        _result = user_service.delete_cohort(_del_cohort_choice)
                        if _result["deleted"]:
                            st.session_state["cohort_mgmt_flash"] = f"Cohort '{_del_cohort_choice}' deleted."
                            st.rerun()
                        else:
                            # Defensive only -- the button above is already
                            # gated on zero usage, so this means the cohort
                            # was used by something between the check above
                            # and this click.
                            st.warning(
                                f"'{_del_cohort_choice}' became in-use just now — "
                                f"{_result['users']} user(s), {_result['transcripts']} transcript(s)."
                            )
        else:
            st.caption("No cohorts registered yet.")

    # ---------------------------------------------------------
    # DATA MANAGEMENT
    # ---------------------------------------------------------
    with tabs[tab_offset + 1]:
        st.subheader("Dataset Maintenance")
        student_reset = st.text_input("Student username")
        if st.button("Reset Student Data"):
            if student_reset:
                try:
                    data_service.reset_student_data(username, student_reset)
                    st.success("Student data cleared")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.subheader("🎯 Reset One Instrument for One Student")
        st.caption(
            "For when a participant jumped ahead and answered a module's "
            "MCQ/survey before it was covered in session — undoes just that "
            "one instrument, leaving all their other legitimate data intact. "
            "Use the broader resets below only if you actually need to wipe "
            "more than one instrument."
        )

        from modules.registry.discover import discover_all_module_numbers

        try:
            _ti_users_df = user_service.get_all_users()
            _ti_candidates = sorted(
                _ti_users_df[_ti_users_df["role"] == "student"]["username"].tolist()
            )
        except Exception:
            _ti_candidates = []

        _TI_TYPE_SUFFIX = {
            "Content MCQ Assessment": "content_mcq_assessment",
            "SCCCES Survey":          "b4ai_sccces_survey",
            "SIMS Survey":            "b4ai_sims_survey",
            "Module Reflection":      "module_reflections",
        }

        # Non-module-scoped instruments (pre/post-course) — full instrument_name,
        # not composed with a module number. Same keys used elsewhere in
        # _ASSESSMENT_LABELS (teacher_dashboard.py).
        _TI_NONMODULE_INSTRUMENTS = {
            "Pre — AI Misconceptions":              "precourse_pre_ai_misconceptions_assessment",
            "Post — AI Misconceptions":             "postcourse_post_ai_misconceptions_assessment",
            "Pre — AI Conceptual Inventory":        "precourse_pre_aici_assessment",
            "Post — AI Conceptual Inventory":       "postcourse_post_aici_assessment",
        }

        _ti_col1, _ti_col2 = st.columns(2)
        with _ti_col1:
            _ti_student = st.selectbox(
                "Student", options=["(select)"] + _ti_candidates,
                key="targeted_reset_student",
            )
        with _ti_col2:
            _ti_type = st.selectbox(
                "Instrument",
                options=list(_TI_TYPE_SUFFIX.keys()) + list(_TI_NONMODULE_INSTRUMENTS.keys()),
                key="targeted_reset_type",
            )

        _ti_is_module_scoped = _ti_type in _TI_TYPE_SUFFIX

        if _ti_is_module_scoped:
            _ti_module_n = st.selectbox(
                "Module", options=discover_all_module_numbers(),
                key="targeted_reset_module",
            )
            _ti_instrument_name = f"module{_ti_module_n}_{_TI_TYPE_SUFFIX[_ti_type]}"
        else:
            st.caption("This assessment isn't tied to a specific module — no Module selector needed.")
            _ti_instrument_name = _TI_NONMODULE_INSTRUMENTS[_ti_type]

        if _ti_student and _ti_student != "(select)":
            try:
                _ti_counts = data_service.count_user_instrument_rows(
                    _ti_student, _ti_instrument_name
                )
                _ti_total = sum(_ti_counts.values())
                if _ti_total == 0:
                    st.info(
                        f"**{_ti_student}** has no data for `{_ti_instrument_name}` "
                        "— nothing to reset."
                    )
                else:
                    st.warning(
                        f"Will delete for **{_ti_student}** / `{_ti_instrument_name}`: "
                        f"{_ti_counts['responses']} response(s), "
                        f"{_ti_counts['completions']} completion flag(s), "
                        f"{_ti_counts['survey_scores']} survey score(s), "
                        f"{_ti_counts['assessment_scores']} assessment score(s)."
                    )
            except Exception as _ti_err:
                st.error(f"Error checking data: {_ti_err}")

            if st.button("🎯 Reset This Instrument for This Student", key="targeted_reset_btn"):
                try:
                    data_service.reset_user_instrument(
                        username, _ti_student, _ti_instrument_name
                    )
                    st.success(
                        f"✅ Reset `{_ti_instrument_name}` for **{_ti_student}** "
                        "— they can retake it."
                    )
                except Exception as _ti_err2:
                    st.error(f"Reset error: {_ti_err2}")

        st.divider()
        instrument_reset = st.text_input("Instrument ID")
        if st.button("Reset Instrument Data"):
            if instrument_reset:
                try:
                    data_service.reset_instrument(username, instrument_reset)
                    st.success("Instrument data cleared")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        confirm = st.text_input("Type RESET to clear all study data")
        if st.button("Reset Entire Study"):
            if confirm == "RESET":
                try:
                    data_service.reset_study(username)
                    st.warning("All research data cleared")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # RESEARCH OPERATIONS
    # ---------------------------------------------------------
    with tabs[tab_offset + 2]:

        # ── Loaded Instruments ────────────────────────────────────────────────
        st.subheader("Loaded Instruments")
        try:
            instruments = research_service.get_loaded_instruments()
            st.dataframe({"Instrument": instruments}, width="stretch")
        except Exception as e:
            st.error(f"Error: {e}")

        st.divider()

        # ── Import Human Ratings ──────────────────────────────────────────────
        st.subheader("Import Human Ratings")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                rows = research_service.import_human_ratings(username, uploaded_file)
                st.success(f"{rows} human ratings imported")
            except Exception as e:
                st.error(f"Error: {e}")

        st.divider()

        # ── Export ────────────────────────────────────────────────────────────
        st.subheader("Export Research Dataset")
        if st.button("Export Research Dataset"):
            try:
                path = research_service.export_research_dataset(username)
                st.success(f"Dataset exported → {path}")
            except Exception as e:
                st.error(f"Error: {e}")

        st.divider()

        st.divider()

        # ── Interview Transcript Store ────────────────────────────────────────
        st.subheader("📂 Interview Transcript Store")
        st.caption(
            "Manage the persistent interview transcript store used by the "
            "Teacher Dashboard for ITA and DTA analysis. "
            "Transcripts are matched to student reflection data by "
            "**participant ID** — the ID here must exactly match the student "
            "username in the responses database for the two sources to align."
        )

        if not _TRANSCRIPT_STORE_AVAILABLE:
            st.warning(
                f"Transcript store unavailable: {_TRANSCRIPT_STORE_ERR}. "
                "Ensure core/analytics/llm/transcript_store.py is installed."
            )
        else:
            # ── Current store contents ────────────────────────────────────────
            try:
                _raw = get_persistent_transcripts()
                # transcript_store may return a DataFrame or a list
                if hasattr(_raw, "to_dict"):
                    # DataFrame — convert to list of dicts
                    _transcripts = _raw.to_dict("records") if not _raw.empty else []
                elif isinstance(_raw, list):
                    _transcripts = _raw
                else:
                    _transcripts = []
                _count = len(_transcripts)
            except Exception as _te:
                _transcripts = []
                _count       = 0
                st.error(f"Could not load transcript store: {_te}")

            col_cnt, col_clr = st.columns([3, 1])
            col_cnt.metric("Transcripts in store", _count)

            if _count > 0:
                # Build display table
                _rows = []
                for t in _transcripts:
                    _rows.append({
                        "Participant ID": t.get("participant_id", "—"),
                        "Source type":    t.get("source_type", "—"),
                        "Cohort":         t.get("cohort_id") or "—",
                        "Characters":     t.get("char_count", "—"),
                        "Uploaded by":    t.get("uploaded_by", "—"),
                        "Uploaded at":    t.get("uploaded_at", "—"),
                        "Transcript ID":  t.get("transcript_id",
                                          t.get("id", "—")),
                    })
                _df = pd.DataFrame(_rows)
                st.dataframe(_df, hide_index=True, width="stretch")

                st.divider()

                # ── Delete individual transcript ──────────────────────────────
                st.markdown("**Delete a transcript**")
                st.caption(
                    "Use this when you need to re-upload a transcript with a "
                    "corrected participant ID to match the reflection data."
                )
                # Build label → (participant_id, source_type) map
                _del_opts = {
                    f"{t.get('participant_id', t.get('id', '—'))}  "
                    f"[{t.get('source_type', 'interview')}]"
                    f"{'  · ' + t['cohort_id'] if t.get('cohort_id') else ''}": (
                        str(t.get("participant_id", t.get("id", ""))),
                        str(t.get("source_type", "interview")),
                    )
                    for t in _transcripts
                }
                _del_label = st.selectbox(
                    "Select transcript to delete",
                    options=list(_del_opts.keys()),
                    key="admin_del_transcript_id",
                )
                if st.button("🗑 Delete selected transcript",
                             key="admin_del_transcript_btn"):
                    try:
                        _del_pid, _del_src = _del_opts[_del_label]
                        delete_transcript(_del_pid, _del_src)
                        st.success(f"Transcript for '{_del_pid}' deleted.")
                        st.rerun()
                    except Exception as _de:
                        st.error(f"Delete failed: {_de}")

                st.divider()

                # ── Delete ALL transcripts ────────────────────────────────────
                st.markdown("**Clear entire transcript store**")
                st.caption(
                    "Removes all transcripts. Use before a fresh bulk upload "
                    "when participant IDs have changed across the cohort."
                )
                _confirm_clear = st.text_input(
                    "Type DELETE ALL to confirm",
                    key="admin_clear_transcripts_confirm",
                )
                if st.button("🗑 Clear all transcripts",
                             key="admin_clear_all_btn",
                             type="secondary"):
                    if _confirm_clear == "DELETE ALL":
                        _deleted = 0
                        _errors  = []
                        for t in _transcripts:
                            _pid = str(t.get("participant_id", t.get("id", "")))
                            _src = str(t.get("source_type", "interview"))
                            try:
                                delete_transcript(_pid, _src)
                                _deleted += 1
                            except Exception as _ce:
                                _errors.append(f"{_pid}: {_ce}")
                        if _errors:
                            st.warning(
                                f"Deleted {_deleted}, "
                                f"{len(_errors)} error(s): "
                                + "; ".join(_errors)
                            )
                        else:
                            st.success(
                                f"All {_deleted} transcript(s) cleared."
                            )
                        st.rerun()
                    else:
                        st.warning("Type DELETE ALL (exactly) to confirm.")
            else:
                st.info("No transcripts in store. Upload below.")

            st.divider()

            # ── Upload new transcripts with participant ID mapping ────────────
            st.markdown("**Upload interview transcripts**")
            st.caption(
                "Upload one or more transcript files. "
                "Each file is assigned a **participant ID** that must match "
                "the student's username in the responses database exactly — "
                "this is how reflections and interviews are linked in analysis. "
                "Accepted formats: `.txt`, `.vtt` (WebVTT), `.pdf`."
            )

            # Generation counter: bumping this gives the uploader a fresh
            # widget key, which is the standard Streamlit trick to actually
            # clear a file_uploader (there's no other way to reset one).
            # Used both after a successful save and by the explicit
            # "Clear form" button below (fixed 2026-08-08).
            if "admin_int_upload_gen" not in st.session_state:
                st.session_state["admin_int_upload_gen"] = 0
            _int_gen = st.session_state["admin_int_upload_gen"]

            _upload_files = st.file_uploader(
                "Select transcript files",
                type=["txt", "vtt", "pdf"],
                accept_multiple_files=True,
                key=f"admin_transcript_upload_{_int_gen}",
                help="Upload multiple files at once. "
                     "Participant IDs are set below.",
            )

            if _upload_files:
                st.markdown(
                    "**Cohort for this batch** — applies to every file below by "
                    "default; each file can still be set to a different cohort "
                    "individually. Use this for pre-pilot interviews recorded "
                    "before the platform existed, or any transcript you want "
                    "grouped by cohort for LLM Analysis comparisons."
                )
                _bulk_cohort = _cohort_selector("admin_int_bulk_cohort", label="Cohort (applies to all files below)")

                st.markdown(
                    "**Map each file to a participant ID** "
                    "(must match the student username exactly):"
                )

                # Fetch existing usernames as suggestions
                try:
                    _users = user_service.get_all_users()
                    if hasattr(_users, "to_dict"):
                        _usernames = list(
                            _users["username"].dropna().tolist()
                        )
                    elif isinstance(_users, list):
                        _usernames = [
                            u.get("username", "") for u in _users
                        ]
                    else:
                        _usernames = []
                except Exception:
                    _usernames = []

                # Import _infer_pid for smart default participant ID
                try:
                    from core.analytics.llm.transcript_store import _infer_pid as _ts_infer_pid
                except Exception:
                    _ts_infer_pid = lambda fn: Path(fn).stem

                _cohort_options = [_COHORT_NONE] + user_service.get_all_cohorts()
                _bulk_index = _cohort_options.index(_bulk_cohort) if _bulk_cohort in _cohort_options else 0

                # Per-file selectboxes use a persistent widget key, so their
                # `index=` default only takes effect the first time each key
                # is created -- changing the bulk cohort afterward silently
                # has no effect on files already rendered. Tracking a single
                # "last seen bulk value" and popping stale keys (the
                # 2026-08-08 fix) isn't fully reliable either -- if the same
                # filename is re-used across batches, its old key can carry
                # a stale value from an even earlier bulk setting that
                # happens to coincidentally match "last seen" again, so the
                # change goes undetected. Folding a generation counter into
                # the key sidesteps the whole class of bug: every bulk
                # change gives every per-file widget a brand new key, so
                # there is no persisted value left to go stale (fixed
                # 2026-08-09, confirmed still failing after the first fix).
                if "admin_int_bulk_cohort_gen" not in st.session_state:
                    st.session_state["admin_int_bulk_cohort_gen"] = 0
                if st.session_state.get("admin_int_bulk_cohort_last_seen") != _bulk_cohort:
                    st.session_state["admin_int_bulk_cohort_gen"] += 1
                    st.session_state["admin_int_bulk_cohort_last_seen"] = _bulk_cohort
                _bulk_gen = st.session_state["admin_int_bulk_cohort_gen"]

                _mappings = {}
                for _uf in _upload_files:
                    _default_id = _ts_infer_pid(_uf.name)
                    _col_pid, _col_cohort = st.columns([2, 1])
                    _pid = _col_pid.text_input(
                        f"Participant ID for **{_uf.name}**",
                        value=_default_id,
                        key=f"admin_pid_{_uf.name}",
                        help=(
                            "Type or paste the exact student username. "
                            "E.g. if the student's login is 'student01', "
                            "enter 'student01'."
                        ),
                    )
                    _file_cohort_choice = _col_cohort.selectbox(
                        "Cohort",
                        options=_cohort_options,
                        index=_bulk_index,
                        key=f"admin_cohort_{_uf.name}_{_bulk_gen}",
                        help="Defaults to the batch cohort above — change to override for just this file.",
                    )
                    _file_cohort = None if _file_cohort_choice == _COHORT_NONE else _file_cohort_choice
                    _mappings[_uf.name] = (_uf, _pid.strip(), _file_cohort)

                _int_clear_col, _int_upload_col = st.columns([1, 2])
                if _int_clear_col.button(
                    "🔄 Clear form", key="admin_int_clear_batch",
                    help="Discard this batch's file selections and start fresh, "
                         "without uploading anything.",
                ):
                    st.session_state["admin_int_upload_gen"] += 1
                    # Also force fresh per-file cohort widgets for the next
                    # batch, in case a re-uploaded file shares a filename
                    # with one from this batch (fixed 2026-08-09).
                    st.session_state["admin_int_bulk_cohort_gen"] = \
                        st.session_state.get("admin_int_bulk_cohort_gen", 0) + 1
                    st.rerun()

                if _int_upload_col.button(
                    "📤 Upload transcripts to store",
                    key="admin_upload_submit",
                    type="primary",
                    disabled=not _mappings,
                ):
                    # Validate participant IDs before writing anything
                    _unknown = [
                        _pid for (_fobj, _pid, _coh) in _mappings.values()
                        if _usernames and _pid and _pid not in _usernames
                    ]
                    if _unknown:
                        st.warning(
                            "⚠️ The following participant IDs are not registered "
                            "usernames — double-check spelling before proceeding "
                            "(this is expected/fine for pre-pilot participants "
                            "who never registered on the platform):\n"
                            + "\n".join(f"• `{p}`" for p in _unknown)
                        )

                    _saved   = 0
                    _skipped = []
                    for _fname, (_fobj, _pid, _coh) in _mappings.items():
                        if not _pid:
                            _skipped.append(f"{_fname} (no participant ID set)")
                            continue
                        try:
                            _raw = _fobj.read()
                            try:
                                _text = _raw.decode("utf-8", errors="replace")
                            except Exception:
                                _text = _raw.decode("latin-1", errors="replace")

                            from core.analytics.llm.transcript_store \
                                import save_transcript
                            save_transcript(
                                participant_id=_pid,
                                content=_text,
                                source_type="interview",
                                filename=_fname,
                                uploaded_by=username,
                                cohort_id=_coh,
                            )
                            _saved += 1
                        except Exception as _ue:
                            _skipped.append(f"{_fname}: {_ue}")

                    if _saved:
                        # Flash message pattern (see _cohort_selector) --
                        # st.rerun() below would otherwise wipe this before
                        # it's ever painted.
                        st.session_state["admin_int_upload_flash"] = (
                            f"✅ {_saved} transcript(s) saved to store."
                        )
                        # Fresh uploader for the next batch -- this is the
                        # "Done" / "clear after upload" facility.
                        st.session_state["admin_int_upload_gen"] += 1
                        st.session_state["admin_int_bulk_cohort_gen"] = \
                            st.session_state.get("admin_int_bulk_cohort_gen", 0) + 1
                    if _skipped:
                        st.session_state["admin_int_upload_flash_warn"] = (
                            f"Skipped {len(_skipped)}: " + "; ".join(_skipped)
                        )
                    st.rerun()

            if st.session_state.get("admin_int_upload_flash"):
                st.success(st.session_state.pop("admin_int_upload_flash"))
            if st.session_state.get("admin_int_upload_flash_warn"):
                st.warning(st.session_state.pop("admin_int_upload_flash_warn"))

        st.divider()

        # ── Observer/Instructor Transcript Store ────────────────────────────
        st.subheader("📂 Observer/Instructor Transcript Store")
        st.caption(
            "Manage the persistent observer/instructor transcript store used by the "
            "Teacher Dashboard for ITA and DTA analysis — session notes or recordings "
            "from an observer or instructor, kept separate from interview transcripts. "
            "Transcripts are matched to student reflection data by "
            "**participant ID** — the ID here must exactly match the student "
            "username in the responses database for the two sources to align."
        )

        if not _TRANSCRIPT_STORE_AVAILABLE:
            st.warning(
                f"Transcript store unavailable: {_TRANSCRIPT_STORE_ERR}. "
                "Ensure core/analytics/llm/transcript_store.py is installed."
            )
        else:
            # ── Current store contents (observer type only) ────────────────────
            try:
                _raw_obs = get_persistent_transcripts(source_type="observer")
                if hasattr(_raw_obs, "to_dict"):
                    _obs_transcripts = _raw_obs.to_dict("records") if not _raw_obs.empty else []
                elif isinstance(_raw_obs, list):
                    _obs_transcripts = _raw_obs
                else:
                    _obs_transcripts = []
                _obs_count = len(_obs_transcripts)
            except Exception as _te:
                _obs_transcripts = []
                _obs_count       = 0
                st.error(f"Could not load transcript store: {_te}")

            col_obs_cnt, col_obs_clr = st.columns([3, 1])
            col_obs_cnt.metric("Observer/instructor transcripts in store", _obs_count)

            if _obs_count > 0:
                _obs_rows = []
                for t in _obs_transcripts:
                    _obs_rows.append({
                        "Participant ID": t.get("participant_id", "—"),
                        "Source type":    t.get("source_type", "—"),
                        "Cohort":         t.get("cohort_id") or "—",
                        "Characters":     t.get("char_count", "—"),
                        "Uploaded by":    t.get("uploaded_by", "—"),
                        "Uploaded at":    t.get("uploaded_at", "—"),
                        "Transcript ID":  t.get("transcript_id",
                                          t.get("id", "—")),
                    })
                _obs_df = pd.DataFrame(_obs_rows)
                st.dataframe(_obs_df, hide_index=True, width="stretch")

                st.divider()

                # ── Delete individual observer transcript ───────────────────
                st.markdown("**Delete a transcript**")
                st.caption(
                    "Use this when you need to re-upload a transcript with a "
                    "corrected participant ID to match the reflection data."
                )
                _obs_del_opts = {
                    f"{t.get('participant_id', t.get('id', '—'))}  "
                    f"[{t.get('source_type', 'observer')}]"
                    f"{'  · ' + t['cohort_id'] if t.get('cohort_id') else ''}": (
                        str(t.get("participant_id", t.get("id", ""))),
                        str(t.get("source_type", "observer")),
                    )
                    for t in _obs_transcripts
                }
                _obs_del_label = st.selectbox(
                    "Select transcript to delete",
                    options=list(_obs_del_opts.keys()),
                    key="admin_obs_del_transcript_id",
                )
                if st.button("🗑 Delete selected transcript",
                             key="admin_obs_del_transcript_btn"):
                    try:
                        _obs_del_pid, _obs_del_src = _obs_del_opts[_obs_del_label]
                        delete_transcript(_obs_del_pid, _obs_del_src)
                        st.success(f"Transcript for '{_obs_del_pid}' deleted.")
                        st.rerun()
                    except Exception as _de:
                        st.error(f"Delete failed: {_de}")

                st.divider()

                # ── Delete ALL observer transcripts ──────────────────────────
                st.markdown("**Clear entire observer/instructor transcript store**")
                st.caption(
                    "Removes all observer/instructor transcripts only — interview "
                    "transcripts and reflections are untouched. "
                    "Use before a fresh bulk upload when participant IDs have "
                    "changed across the cohort."
                )
                _obs_confirm_clear = st.text_input(
                    "Type DELETE ALL to confirm",
                    key="admin_obs_clear_transcripts_confirm",
                )
                if st.button("🗑 Clear all observer/instructor transcripts",
                             key="admin_obs_clear_all_btn",
                             type="secondary"):
                    if _obs_confirm_clear == "DELETE ALL":
                        _obs_deleted = 0
                        _obs_errors  = []
                        for t in _obs_transcripts:
                            _pid = str(t.get("participant_id", t.get("id", "")))
                            _src = str(t.get("source_type", "observer"))
                            try:
                                delete_transcript(_pid, _src)
                                _obs_deleted += 1
                            except Exception as _ce:
                                _obs_errors.append(f"{_pid}: {_ce}")
                        if _obs_errors:
                            st.warning(
                                f"Deleted {_obs_deleted}, "
                                f"{len(_obs_errors)} error(s): "
                                + "; ".join(_obs_errors)
                            )
                        else:
                            st.success(
                                f"All {_obs_deleted} transcript(s) cleared."
                            )
                        st.rerun()
                    else:
                        st.warning("Type DELETE ALL (exactly) to confirm.")
            else:
                st.info("No observer/instructor transcripts in store. Upload below.")

            st.divider()

            # ── Upload new observer/instructor transcripts ──────────────────
            st.markdown("**Upload observer/instructor transcripts**")
            st.caption(
                "Upload one or more transcript files. "
                "Each file is assigned a **participant ID** that must match "
                "the student's username in the responses database exactly — "
                "this is how reflections and observer/instructor notes are linked in analysis. "
                "Accepted formats: `.txt`, `.vtt` (WebVTT), `.pdf`."
            )

            if "admin_obs_upload_gen" not in st.session_state:
                st.session_state["admin_obs_upload_gen"] = 0
            _obs_gen = st.session_state["admin_obs_upload_gen"]

            _obs_upload_files = st.file_uploader(
                "Select transcript files",
                type=["txt", "vtt", "pdf"],
                accept_multiple_files=True,
                key=f"admin_obs_transcript_upload_{_obs_gen}",
                help="Upload multiple files at once. "
                     "Participant IDs are set below.",
            )

            if _obs_upload_files:
                st.markdown(
                    "**Cohort for this batch** — applies to every file below by "
                    "default; each file can still be set to a different cohort "
                    "individually."
                )
                _obs_bulk_cohort = _cohort_selector("admin_obs_bulk_cohort", label="Cohort (applies to all files below)")

                st.markdown(
                    "**Map each file to a participant ID** "
                    "(must match the student username exactly):"
                )

                try:
                    _obs_users = user_service.get_all_users()
                    if hasattr(_obs_users, "to_dict"):
                        _obs_usernames = list(
                            _obs_users["username"].dropna().tolist()
                        )
                    elif isinstance(_obs_users, list):
                        _obs_usernames = [
                            u.get("username", "") for u in _obs_users
                        ]
                    else:
                        _obs_usernames = []
                except Exception:
                    _obs_usernames = []

                try:
                    from core.analytics.llm.transcript_store import _infer_pid as _obs_infer_pid
                except Exception:
                    _obs_infer_pid = lambda fn: Path(fn).stem

                _obs_cohort_options = [_COHORT_NONE] + user_service.get_all_cohorts()
                _obs_bulk_index = _obs_cohort_options.index(_obs_bulk_cohort) if _obs_bulk_cohort in _obs_cohort_options else 0

                # See the matching comment in the Interview Transcript Store
                # section above -- a generation-counter key sidesteps the
                # stale-key class of bug entirely (fixed 2026-08-09).
                if "admin_obs_bulk_cohort_gen" not in st.session_state:
                    st.session_state["admin_obs_bulk_cohort_gen"] = 0
                if st.session_state.get("admin_obs_bulk_cohort_last_seen") != _obs_bulk_cohort:
                    st.session_state["admin_obs_bulk_cohort_gen"] += 1
                    st.session_state["admin_obs_bulk_cohort_last_seen"] = _obs_bulk_cohort
                _obs_bulk_gen = st.session_state["admin_obs_bulk_cohort_gen"]

                _obs_mappings = {}
                for _uf in _obs_upload_files:
                    _obs_default_id = _obs_infer_pid(_uf.name)
                    _obs_col_pid, _obs_col_cohort = st.columns([2, 1])
                    _obs_pid = _obs_col_pid.text_input(
                        f"Participant ID for **{_uf.name}**",
                        value=_obs_default_id,
                        key=f"admin_obs_pid_{_uf.name}",
                        help=(
                            "Type or paste the exact student username. "
                            "E.g. if the student's login is 'student01', "
                            "enter 'student01'."
                        ),
                    )
                    _obs_file_cohort_choice = _obs_col_cohort.selectbox(
                        "Cohort",
                        options=_obs_cohort_options,
                        index=_obs_bulk_index,
                        key=f"admin_obs_cohort_{_uf.name}_{_obs_bulk_gen}",
                        help="Defaults to the batch cohort above — change to override for just this file.",
                    )
                    _obs_file_cohort = None if _obs_file_cohort_choice == _COHORT_NONE else _obs_file_cohort_choice
                    _obs_mappings[_uf.name] = (_uf, _obs_pid.strip(), _obs_file_cohort)

                _obs_clear_col, _obs_upload_col = st.columns([1, 2])
                if _obs_clear_col.button(
                    "🔄 Clear form", key="admin_obs_clear_batch",
                    help="Discard this batch's file selections and start fresh, "
                         "without uploading anything.",
                ):
                    st.session_state["admin_obs_upload_gen"] += 1
                    st.session_state["admin_obs_bulk_cohort_gen"] = \
                        st.session_state.get("admin_obs_bulk_cohort_gen", 0) + 1
                    st.rerun()

                if _obs_upload_col.button(
                    "📤 Upload transcripts to store",
                    key="admin_obs_upload_submit",
                    type="primary",
                    disabled=not _obs_mappings,
                ):
                    _obs_unknown = [
                        _pid for (_fobj, _pid, _coh) in _obs_mappings.values()
                        if _obs_usernames and _pid and _pid not in _obs_usernames
                    ]
                    if _obs_unknown:
                        st.warning(
                            "⚠️ The following participant IDs are not registered "
                            "usernames — double-check spelling before proceeding "
                            "(this is expected/fine for pre-pilot participants "
                            "who never registered on the platform):\n"
                            + "\n".join(f"• `{p}`" for p in _obs_unknown)
                        )

                    _obs_saved   = 0
                    _obs_skipped = []
                    for _fname, (_fobj, _pid, _coh) in _obs_mappings.items():
                        if not _pid:
                            _obs_skipped.append(f"{_fname} (no participant ID set)")
                            continue
                        try:
                            _raw = _fobj.read()
                            try:
                                _text = _raw.decode("utf-8", errors="replace")
                            except Exception:
                                _text = _raw.decode("latin-1", errors="replace")

                            from core.analytics.llm.transcript_store \
                                import save_transcript
                            save_transcript(
                                participant_id=_pid,
                                content=_text,
                                source_type="observer",
                                filename=_fname,
                                uploaded_by=username,
                                cohort_id=_coh,
                            )
                            _obs_saved += 1
                        except Exception as _ue:
                            _obs_skipped.append(f"{_fname}: {_ue}")

                    if _obs_saved:
                        st.session_state["admin_obs_upload_flash"] = (
                            f"✅ {_obs_saved} transcript(s) saved to store."
                        )
                        st.session_state["admin_obs_upload_gen"] += 1
                        st.session_state["admin_obs_bulk_cohort_gen"] = \
                            st.session_state.get("admin_obs_bulk_cohort_gen", 0) + 1
                    if _obs_skipped:
                        st.session_state["admin_obs_upload_flash_warn"] = (
                            f"Skipped {len(_obs_skipped)}: "
                            + "; ".join(_obs_skipped)
                        )
                    st.rerun()

            if st.session_state.get("admin_obs_upload_flash"):
                st.success(st.session_state.pop("admin_obs_upload_flash"))
            if st.session_state.get("admin_obs_upload_flash_warn"):
                st.warning(st.session_state.pop("admin_obs_upload_flash_warn"))

        # ── Dataset Metrics ───────────────────────────────────────────────────
        if DEBUG:
            # --- Begin DEBUG for ensuring correct database is being read to read survey scores
            from core.db_utils import get_connection

            conn = get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM survey_scores")
                st.write("DEBUG — survey_scores rows:", cursor.fetchone()[0])
            except Exception as e:
                st.write("DEBUG ERROR:", e)
            conn.close()
            # ---END of DEBUG


        st.subheader("Dataset Metrics")
        try:
            metrics = diagnostics_service.get_research_metrics()
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Responses",         metrics.get("total_responses", 0))
            col2.metric("Completions",       metrics.get("total_completions", 0))
            col3.metric("Survey Scores",     metrics.get("total_survey_scores", 0))
            col4.metric("Assessment Scores", metrics.get("total_assessment_scores", 0))
            col5.metric("LLM Ratings",       metrics.get("total_llm_ratings", 0))
            col6.metric("Human Ratings",     metrics.get("total_human_ratings", 0))
            #if DEBUG:
            #    st.write("DEBUG — Metrics:", metrics)
        except Exception as e:
            st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # SYSTEM OPERATIONS
    # ---------------------------------------------------------
    with tabs[tab_offset + 3]:
        st.subheader("Database Maintenance")
        if st.button("Backup Databases"):
            try:
                backups = system_service.backup_databases(username)
                st.success("Backup created")
                st.write(backups if DEBUG else list(backups.keys()))
            except Exception as e:
                st.error(f"Error: {e}")

        st.divider()
        st.subheader("⬇️ Download Databases Now")
        st.caption(
            "Creates a fresh backup, then packages both databases into a single "
            "ZIP for you to download directly to your own computer. This is the "
            "only backup path that leaves Railway entirely — same-volume backups "
            "protect against admin mistakes, but not against a platform-level "
            "problem with the volume itself. Keep this off-platform copy somewhere "
            "safe (e.g. your own drive), especially before or after important "
            "pilot sessions."
        )
        if st.button("📦 Prepare Download", key="prep_download_btn"):
            try:
                import io as _io
                import zipfile as _zipfile

                _fresh = system_service.backup_databases(username)
                _zip_buf = _io.BytesIO()
                with _zipfile.ZipFile(_zip_buf, "w", _zipfile.ZIP_DEFLATED) as _zf:
                    for _label, _path in _fresh.items():
                        _zf.write(_path, arcname=_path.name)
                st.session_state["_db_download_zip"] = _zip_buf.getvalue()
                st.session_state["_db_download_name"] = (
                    f"b4ai_databases_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip"
                )
                st.success("Ready — click below to download.")
            except Exception as e:
                st.error(f"Error preparing download: {e}")

        if st.session_state.get("_db_download_zip"):
            st.download_button(
                label="⬇️ Download ZIP (responses.db + users.db)",
                data=st.session_state["_db_download_zip"],
                file_name=st.session_state["_db_download_name"],
                mime="application/zip",
                key="db_zip_download_btn",
            )

        st.divider()
        if st.button("Clone Databases"):
            try:
                clones = system_service.clone_databases(username)
                st.success("Clone created")
                st.write(clones if DEBUG else list(clones.keys()))
            except Exception as e:
                st.error(f"Error: {e}")

        st.divider()
        if st.button("Clear Cache"):
            try:
                system_service.clear_cache(username)
                st.success("Cache cleared")
            except Exception as e:
                st.error(f"Error: {e}")

        # ── Restore Databases ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Restore Databases")
        st.caption(
            "Select a backup set to restore. Both databases (users.db and "
            "responses.db) will be replaced. Row counts are verified after "
            "restore and logged to restore_log."
        )
        try:
            _backups = system_service.list_backups()
        except Exception as _be:
            _backups = []
            st.error(f"Could not list backups: {_be}")

        if not _backups:
            st.info("No complete backup sets found in the backups/ directory.")
        else:
            _backup_labels = {
                f"{b['timestamp']} — users: {b['users'].name}, "
                f"responses: {b['responses'].name}": b["timestamp"]
                for b in _backups
            }
            _selected_label = st.selectbox(
                "Select backup set to restore",
                options=list(_backup_labels.keys()),
                key="restore_backup_select",
            )
            _selected_ts = _backup_labels[_selected_label]

            _restore_confirm = st.text_input(
                "Type RESTORE to confirm — this overwrites live databases",
                key="restore_confirm_input",
            )
            if st.button("🔄 Restore Selected Backup", key="restore_btn",
                         type="secondary"):
                if _restore_confirm != "RESTORE":
                    st.warning("Type RESTORE (exactly) to confirm.")
                else:
                    try:
                        _result = system_service.restore_databases(
                            admin_user=username,
                            backup_timestamp=_selected_ts,
                        )
                        if _result["verified"]:
                            st.success(_result["message"])
                        else:
                            st.warning(_result["message"])
                        st.markdown(
                            f"**Row counts:** users {_result['users_pre']}→"
                            f"{_result['users_post']}  |  "
                            f"responses {_result['responses_pre']}→"
                            f"{_result['responses_post']}"
                        )
                    except FileNotFoundError as _fnf:
                        st.error(str(_fnf))
                    except Exception as _re:
                        st.error(f"Restore failed: {_re}")

        # ── Auto-backup Status ────────────────────────────────────────────────
        st.divider()
        st.subheader("Auto-Backup Status")
        try:
            _status = system_service.get_system_status()
            _interval = int(os.environ.get("BACKUP_INTERVAL_HOURS", "24"))
            st.info(
                f"🕐 Auto-backup interval: every **{_interval} hour(s)** "
                f"(set `BACKUP_INTERVAL_HOURS` in `.env` to change, then restart).  \n"
                f"Backup files in store: **{_status.get('backup_files', 0)}**"
            )
        except Exception as _se:
            st.error(f"Could not read system status: {_se}")
        # ─────────────────────────────────────────────────────────────────────

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------
    with tabs[tab_offset + 4]:
        st.subheader("System Metrics")
        try:
            diag = diagnostics_service.get_full_diagnostics()
            col1, col2, col3 = st.columns(3)
            col1.metric("Users",    diag.get("total_users", 0))
            col2.metric("Students", diag.get("total_students", 0))
            col3.metric("Teachers", diag.get("total_teachers", 0))
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Responses",         diag.get("total_responses", 0))
            col2.metric("Completions",       diag.get("total_completions", 0))
            col3.metric("Survey Scores",     diag.get("total_survey_scores", 0))
            col4.metric("Assessment Scores", diag.get("total_assessment_scores", 0))
            if DEBUG:
                st.write("DEBUG — Full diagnostics:", diag)
        except Exception as e:
            st.error(f"Error: {e}")

        st.divider()
        st.subheader("Audit Log")
        try:
            logs = diagnostics_service.get_audit_log()
            st.dataframe(logs, width="stretch")

            # ── CSV Export ────────────────────────────────────────────────────
            from core.admin.audit_logger import get_all_logs_df
            import io as _io
            _log_df = get_all_logs_df(limit=5000)
            if not _log_df.empty:
                _csv_buf = _io.StringIO()
                _log_df.to_csv(_csv_buf, index=False)
                st.download_button(
                    label="⬇️ Download Audit Log (CSV)",
                    data=_csv_buf.getvalue().encode("utf-8"),
                    file_name=f"b4ai_audit_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="audit_log_csv_download",
                )
            # ─────────────────────────────────────────────────────────────────

        except Exception as e:
            st.error(f"Error: {e}")
