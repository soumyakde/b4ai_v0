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
            # Green (colorblind-safe: also use check pattern + text)
            badge_color = "#009E73"   # teal-green — safe for deuteranopia
            badge_bg    = "#E6F7F1"
            status_icon = "✅"
            status_text = "Completed"
            btn_label   = None
        elif unlocked:
            # Blue — universally distinguishable
            badge_color = "#0077BB"
            badge_bg    = "#E6F3FB"
            status_icon = "🔓"
            status_text = "Available"
            btn_label   = f"▶ Open {title}"
        else:
            # Gray — locked
            badge_color = "#888888"
            badge_bg    = "#F0F0F0"
            status_icon = "🔒"
            status_text = "Locked"
            btn_label   = None

        st.markdown(
            f"<div style='"
            f"background:{badge_bg};"
            f"border-left:5px solid {badge_color};"
            f"border-radius:6px;"
            f"padding:0.6rem 1rem 0.4rem 1rem;"
            f"margin-bottom:0.5rem;'>"
            f"<span style='font-size:1.15rem;font-weight:700;"
            f"color:{badge_color};'>{status_icon} {title}</span>"
            f"<span style='font-size:0.85rem;color:{badge_color};"
            f"margin-left:0.6rem;font-weight:600;'>[{status_text}]</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if description:
            st.caption(description)

        if unlocked and not complete:
            if st.button(btn_label, key=module_id,
                         type="primary" if unlocked else "secondary"):
                st.session_state.view = module_id
                st.rerun()

        st.divider()