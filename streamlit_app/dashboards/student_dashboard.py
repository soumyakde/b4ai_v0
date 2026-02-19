"""
Student Dashboard
Derived-state architecture.
No JSON.
No stored completion flags.
Everything computed from SQLite + registry.
"""

import streamlit as st

from modules.registry.module_registry import module_registry
from core.progress_engine import (
    is_module_complete,
    is_module_unlocked,
)


# ----------------------------------------------------------
# DASHBOARD UI
# ----------------------------------------------------------

def show_student_dashboard(username: str):

    st.title("🎓 Student Dashboard")
    st.markdown(f"Welcome **{username}**")

    st.divider()

    modules = module_registry.get_ordered_modules()

    for module in modules:

        meta = module["meta"]
        module_id = meta["module_id"]
        title = meta["title"]
        description = meta.get("description", "")

        complete = is_module_complete(username, module)
        unlocked = is_module_unlocked(username, module_id)

        # ----------------------------------------
        # Status Display
        # ----------------------------------------

        if complete:
            status_icon = "✅"
            status_text = "Completed"
        elif unlocked:
            status_icon = "🔓"
            status_text = "Available"
        else:
            status_icon = "🔒"
            status_text = "Locked"

        st.markdown(f"### {status_icon} {title} ({status_text})")

        if description:
            st.caption(description)

        # ----------------------------------------
        # Navigation
        # ----------------------------------------

        if unlocked and not complete:
            if st.button(f"Open {title}", key=module_id):
                st.session_state.view = module_id
                st.rerun()

        st.divider()