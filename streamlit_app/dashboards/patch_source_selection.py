"""
Patch: Multi-source selection for ITA and DTA.
Applies to the deployed teacher_dashboard.py (the full 2792-line version).
Run: python patch_source_selection.py <path_to_teacher_dashboard.py>
"""
import sys, re

path = sys.argv[1] if len(sys.argv) > 1 else "teacher_dashboard.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

original_len = len(content)
patches_applied = []

# -----------------------------------------------------------------------
# Helper function: add _load_combined_transcripts before _execute_dta_run
# -----------------------------------------------------------------------
HELPER_FN = '''
def _load_combined_transcripts(
    sources: list,
    per_run_files=None,
) -> list:
    """
    Load and merge transcripts from one or more sources.

    sources : list of str, each one of:
        "responses"   — module reflection notes from responses.db
        "persistent"  — uploaded interview transcripts (persistent store)
        "per_run"     — files uploaded fresh for this run

    When a participant appears in more than one source their texts are
    concatenated so the analysis sees the full picture.
    """
    import sqlite3 as _sq, re as _re
    from pathlib import Path as _Pth
    from collections import defaultdict

    combined: dict = defaultdict(lambda: {"content": [], "source_types": []})

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
                "WHERE instrument_name LIKE \'%module_reflections%\' "
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
                transcripts = load_for_analysis(
                    source="persistent", source_type="interview",
                )
                for t in transcripts:
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
                try:
                    text = raw.decode("utf-8", errors="replace")
                except Exception:
                    text = raw.decode("latin-1", errors="replace")
                pid = _re.sub(r"\\.(vtt|txt|pdf)$", "", f_obj.name, flags=_re.I).strip()
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

'''

# Insert helper before _execute_dta_run
anchor_dta = "def _execute_dta_run("
if anchor_dta in content and "_load_combined_transcripts" not in content:
    content = content.replace(anchor_dta, HELPER_FN + anchor_dta, 1)
    patches_applied.append("helper _load_combined_transcripts inserted")

# -----------------------------------------------------------------------
# Patch 1: DTA run panel — replace single selectbox with multi-checkboxes
# -----------------------------------------------------------------------
OLD_DTA_SOURCE = '''    st.markdown("**Data source:**")
    dta_source_display = st.selectbox(
        "Source",
        options=[
            "Module reflection notes (from responses DB)",
            "Persistent store (uploaded interviews)",
            "Upload now (this run only)",
        ],
        key="dta_source",
    )
    _DTA_SRC_MAP = {
        "Module reflection notes (from responses DB)": "responses",
        "Persistent store (uploaded interviews)":      "persistent",
        "Upload now (this run only)":                  "per_run",
    }
    dta_source_key = _DTA_SRC_MAP[dta_source_display]

    dta_per_run_files = None
    if dta_source_key == "per_run":
        dta_per_run_files = st.file_uploader(
            "Upload .txt or .pdf files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="dta_upload",
        )'''

NEW_DTA_SOURCE = '''    st.markdown("**Data sources** (select one or more):")
    dta_col1, dta_col2, dta_col3 = st.columns(3)
    dta_use_reflections = dta_col1.checkbox(
        "Module reflections (DB)",
        value=True,
        key="dta_src_reflections",
        help="End-of-module reflection notes already in responses.db",
    )
    dta_use_persistent = dta_col2.checkbox(
        "Interview transcripts (store)",
        value=False,
        key="dta_src_persistent",
        help="Semi-structured interview transcripts uploaded via Admin dashboard",
    )
    dta_use_per_run = dta_col3.checkbox(
        "Upload now (this run only)",
        value=False,
        key="dta_src_per_run",
        help="Upload transcript files fresh for this run",
    )

    dta_selected_sources = []
    if dta_use_reflections: dta_selected_sources.append("responses")
    if dta_use_persistent:  dta_selected_sources.append("persistent")
    if dta_use_per_run:     dta_selected_sources.append("per_run")

    dta_per_run_files = None
    if dta_use_per_run:
        dta_per_run_files = st.file_uploader(
            "Upload transcript files (.vtt, .txt, .pdf)",
            type=["vtt", "txt", "pdf"],
            accept_multiple_files=True,
            key="dta_upload",
        )

    if not dta_selected_sources:
        st.warning("Select at least one data source.")'''

if OLD_DTA_SOURCE in content:
    content = content.replace(OLD_DTA_SOURCE, NEW_DTA_SOURCE, 1)
    patches_applied.append("DTA source multi-select")

# -----------------------------------------------------------------------
# Patch 2: DTA run button call — pass sources list
# -----------------------------------------------------------------------
OLD_DTA_CALL = '''    if run_dta:
        _execute_dta_run(
            username=username,
            canonical_df=canonical_df,
            source_key=dta_source_key,
            per_run_files=dta_per_run_files,
            models=dta_models,
            temperature=dta_temp,
            construct_groups=dta_groups,
            show_stream=st.session_state.get("dta_show_stream", False),
        )'''

NEW_DTA_CALL = '''    if run_dta:
        if not dta_selected_sources:
            st.warning("Select at least one data source before running.")
        else:
            _execute_dta_run(
                username=username,
                canonical_df=canonical_df,
                sources=dta_selected_sources,
                per_run_files=dta_per_run_files,
                models=dta_models,
                temperature=dta_temp,
                construct_groups=dta_groups,
                show_stream=st.session_state.get("dta_show_stream", False),
            )'''

if OLD_DTA_CALL in content:
    content = content.replace(OLD_DTA_CALL, NEW_DTA_CALL, 1)
    patches_applied.append("DTA run button call updated")

# -----------------------------------------------------------------------
# Patch 3: _execute_dta_run signature — replace source_key with sources
# -----------------------------------------------------------------------
OLD_EXEC_SIG = '''def _execute_dta_run(
    username, canonical_df, source_key, per_run_files,
    models, temperature, construct_groups,
    show_stream: bool = False,
) -> None:
    """Load transcripts and run DTA phases 2-5 with streaming transparency."""'''

NEW_EXEC_SIG = '''def _execute_dta_run(
    username, canonical_df, sources, per_run_files,
    models, temperature, construct_groups,
    show_stream: bool = False,
) -> None:
    """Load transcripts from one or more sources and run DTA phases 2-5."""
    if isinstance(sources, str):
        sources = [sources]  # backward compat'''

if OLD_EXEC_SIG in content:
    content = content.replace(OLD_EXEC_SIG, NEW_EXEC_SIG, 1)
    patches_applied.append("_execute_dta_run signature updated")

# -----------------------------------------------------------------------
# Patch 4: Replace the data-loading block inside _execute_dta_run
#          (the big if source_key == "responses" block)
# -----------------------------------------------------------------------
OLD_LOAD_BLOCK = '''    with st.spinner("Loading data..."):
        try:
            transcripts = load_for_analysis(
                source=source_key,
                source_type="interview",
                per_run_files=per_run_files,
            )
        except Exception as e:
            st.error(f"Could not load data: {e}"); return

    if source_key == "responses":
        from collections import defaultdict
        import re as _re2, sqlite3 as _sq
        from pathlib import Path as _Path2
        grouped = defaultdict(list)
        db2 = next(
            (p / "responses.db" for p in _Path2(__file__).resolve().parents
             if (p / "responses.db").exists()), None
        )
        if db2:
            conn2 = _sq.connect(db2)
            rows2 = conn2.execute(
                "SELECT user_id, response_value FROM responses "
                "WHERE instrument_name LIKE \'%module_reflections%\'"
            ).fetchall()
            conn2.close()
            for uid, rval in rows2:
                if rval and str(rval).strip():
                    grouped[uid].append(str(rval))
        transcripts = [
            {"participant_id": pid, "source_type": "reflection",
             "content": " ".join(texts)}
            for pid, texts in grouped.items()
            if any(t.strip() for t in texts)
        ]

    if not transcripts:
        st.warning("No transcript data found."); return'''

NEW_LOAD_BLOCK = '''    src_label = " + ".join(sources)
    with st.spinner(f"Loading data from: {src_label}..."):
        try:
            transcripts = _load_combined_transcripts(
                sources=sources,
                per_run_files=per_run_files,
            )
        except Exception as e:
            st.error(f"Could not load data: {e}"); return

    if not transcripts:
        st.warning(
            f"No transcript data found in: {src_label}. "
            "Check that the selected sources contain data."
        ); return'''

if OLD_LOAD_BLOCK in content:
    content = content.replace(OLD_LOAD_BLOCK, NEW_LOAD_BLOCK, 1)
    patches_applied.append("_execute_dta_run load block replaced")

# -----------------------------------------------------------------------
# Patch 5: Update source_type passed to _dta_create_run
# -----------------------------------------------------------------------
OLD_CREATE = '''        run_id = _dta_create_run(
            model=model, temperature=temperature,
            source_type=source_key,
            construct_groups=construct_groups,
            created_by=username,
        )'''

NEW_CREATE = '''        run_id = _dta_create_run(
            model=model, temperature=temperature,
            source_type="+".join(sources),
            construct_groups=construct_groups,
            created_by=username,
        )'''

if OLD_CREATE in content:
    content = content.replace(OLD_CREATE, NEW_CREATE, 1)
    patches_applied.append("_dta_create_run source_type updated")

# -----------------------------------------------------------------------
# Patch 6: ITA Step 5 — replace single selectbox with multi-checkboxes
# -----------------------------------------------------------------------
OLD_ITA_SOURCE = '''        # Source selector — editable here so user never has to go back
        _src_display = st.selectbox(
            "Data source",
            options=[
                "Module reflection notes (from responses DB)",
                "Persistent store (interviews uploaded by admin)",
                "Upload now (this run only)",
            ],
            index=0,
            key="llm_step5_source",
        )
        _SRC_MAP = {
            "Persistent store (interviews uploaded by admin)": "persistent",
            "Upload now (this run only)": "per_run",
            "Module reflection notes (from responses DB)": "responses",
        }
        source_key = _SRC_MAP[_src_display]'''

NEW_ITA_SOURCE = '''        # Multi-source selector — user can combine reflections + interviews
        st.markdown("**Data sources** (select one or more):")
        _ita_s_col1, _ita_s_col2, _ita_s_col3 = st.columns(3)
        _ita_use_ref = _ita_s_col1.checkbox(
            "Module reflections (DB)", value=True, key="llm_s5_src_ref",
            help="End-of-module reflection notes from responses.db",
        )
        _ita_use_int = _ita_s_col2.checkbox(
            "Interview transcripts", value=False, key="llm_s5_src_int",
            help="Uploaded semi-structured interview transcripts",
        )
        _ita_use_upl = _ita_s_col3.checkbox(
            "Upload now", value=False, key="llm_s5_src_upl",
            help="Upload transcript files fresh for this run",
        )
        ita_sources = []
        if _ita_use_ref: ita_sources.append("responses")
        if _ita_use_int: ita_sources.append("persistent")
        if _ita_use_upl: ita_sources.append("per_run")
        source_key = "+".join(ita_sources) if ita_sources else "responses"'''

if OLD_ITA_SOURCE in content:
    content = content.replace(OLD_ITA_SOURCE, NEW_ITA_SOURCE, 1)
    patches_applied.append("ITA Step 5 source multi-select")

# -----------------------------------------------------------------------
# Patch 7: ITA _run_ita_pipeline — update loading to use combined loader
# -----------------------------------------------------------------------
OLD_ITA_LOAD = '''    # source may already be a resolved key or a display string
    _VALID = {"persistent", "per_run", "responses"}
    if source in _VALID:
        source_key = source
    else:
        source_key = {
            "Persistent store (interviews uploaded by admin)": "persistent",
            "Persistent store": "persistent",
            "Upload now (this run only)": "per_run",
            "Upload now": "per_run",
            "Module reflection notes (from responses DB)": "responses",
            "Reflection notes (DB)": "responses",
        }.get(source, "responses" if "reflect" in str(source).lower() else "persistent")

    per_run_files = (st.session_state.get("llm_g_upload") or
                     st.session_state.get("llm_e_files"))

    with st.spinner(f"Loading transcripts from **{source_key}**..."):
        try:
            transcripts = load_for_analysis(
                source=source_key,
                source_type="interview",
                per_run_files=per_run_files if source_key == "per_run" else None,
            )
        except Exception as e:
            st.error(f"Could not load transcripts: {e}"); return

    if not transcripts:
        _hints = {
            "persistent": "No interviews in persistent store — ask admin to upload transcripts first.",
            "per_run":    "No files received — return to Step 1 and upload files before running.",
            "responses":  "No reflection notes found — check that responses.db contains rows with instrument_name LIKE \'%module_reflections%\'.",
        }
        st.warning(f"No transcript data found. {_hints.get(source_key, \'\')} (source_key={source_key})")
        return

    # For reflection notes: group per-participant responses into one text block
    if source_key == "responses":
        from collections import defaultdict
        grouped = defaultdict(list)
        for t in transcripts:
            grouped[t["participant_id"]].append(t["content"])
        transcripts = [
            {"participant_id": pid, "source_type": "reflection",
             "content": " ".join(texts)}
            for pid, texts in grouped.items()
            if any(t.strip() for t in texts)
        ]'''

NEW_ITA_LOAD = '''    # Resolve source string to list of source keys
    _VALID_KEYS = {"persistent", "per_run", "responses"}
    if isinstance(source, list):
        source_keys = source
    elif "+" in str(source):
        source_keys = [s for s in source.split("+") if s in _VALID_KEYS]
    elif source in _VALID_KEYS:
        source_keys = [source]
    else:
        # Legacy display string fallback
        _legacy = {
            "Persistent store (interviews uploaded by admin)": "persistent",
            "Persistent store": "persistent",
            "Upload now (this run only)": "per_run",
            "Upload now": "per_run",
            "Module reflection notes (from responses DB)": "responses",
            "Reflection notes (DB)": "responses",
        }
        source_keys = [_legacy.get(source, "responses")]

    source_key = "+".join(source_keys)
    per_run_files = (st.session_state.get("llm_g_upload") or
                     st.session_state.get("llm_e_files") or
                     st.session_state.get("llm_s5_upload"))

    src_label = " + ".join(source_keys)
    with st.spinner(f"Loading transcripts from: {src_label}..."):
        try:
            transcripts = _load_combined_transcripts(
                sources=source_keys,
                per_run_files=per_run_files if "per_run" in source_keys else None,
            )
        except Exception as e:
            st.error(f"Could not load transcripts: {e}"); return

    if not transcripts:
        st.warning(
            f"No transcript data found in: {src_label}. "
            "Check that the selected sources contain data."
        )
        return'''

if OLD_ITA_LOAD in content:
    content = content.replace(OLD_ITA_LOAD, NEW_ITA_LOAD, 1)
    patches_applied.append("ITA pipeline load block updated")

# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Original: {original_len} chars → New: {len(content)} chars")
print(f"Patches applied ({len(patches_applied)}):")
for p in patches_applied:
    print(f"  ✅ {p}")
not_applied = 7 - len(patches_applied)
if not_applied:
    print(f"  ⚠️  {not_applied} patch(es) not applied — anchors may differ in deployed file")
