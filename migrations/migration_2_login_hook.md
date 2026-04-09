"""
migrations/migration_2_login_hook.md
=====================================
Migration 2 — Write RID to session state at login

This is a documentation file showing the EXACT change to make in
streamlit_app/app.py. It is a 4-line addition, nothing removed.

Location
--------
File:    streamlit_app/app.py
Block:   show_login() function, login success branch
Around:  lines 285-287 (the st.session_state block after status checks)

Change
------
Find this exact block (lines 285-287):

    st.session_state.logged_in = True
    st.session_state.username  = uname
    st.session_state.mode      = "dashboard"

Replace with:

    st.session_state.logged_in = True
    st.session_state.username  = uname
    st.session_state.mode      = "dashboard"
    # Migration 2 — resolve RID at login, cache in session for write paths
    try:
        from auth.user_manager import get_user_rid
        st.session_state.rid = get_user_rid(uname)
    except Exception:
        st.session_state.rid = None  # graceful fallback during transition

That is the ONLY change to app.py.

Companion change: add get_user_rid() to auth/user_manager.py
--------------------------------------------------------------
Add this function after get_user_status() (around line 238):

def get_user_rid(username: str) -> str | None:
    \"\"\"
    Return the RID for a given username, or None if not yet assigned.
    Called once at login — result is cached in st.session_state.rid
    for the duration of the session.
    \"\"\"
    with get_connection() as conn:
        result = conn.execute(
            "SELECT rid FROM users WHERE username = ?", (username,)
        ).fetchone()
    return result[0] if result else None

Verification
------------
After deploying, log in as any student and confirm in the Streamlit
session state inspector (or add a temporary st.write) that:
    st.session_state.rid  →  non-None 16-char hex string
    st.session_state.username  →  still present (not removed)

Both must coexist during the transition period.

What this does NOT change
--------------------------
- No submission logic changes in this migration
- No research table writes change in this migration
- The student experience is identical
- RID is written to session but not yet used by any write path
  (that is Migration 3)

Migration 3 hook point
-----------------------
After Migration 2 is verified, Migration 3 changes the survey/module
submission engine to write st.session_state.rid alongside user_id.
The centralized entry point is db_utils.mark_instrument_complete()
and the survey response writer — both already use get_connection().
"""
