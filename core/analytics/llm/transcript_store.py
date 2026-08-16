# core/analytics/llm/transcript_store.py
"""
Transcript Store
================
Manages interview transcripts for ITA and DTA analysis.

Two storage modes:
    Persistent — uploaded to responses.db, reused across runs
    Per-run    — uploaded fresh each run

DB schema (added to responses.db):
    transcripts(id, participant_id, source_type, content, uploaded_by, uploaded_at, cohort_id)

cohort_id (added 2026-08-08) tags a transcript with a cohort directly,
independent of any registered user record — needed for pre-pilot interview
transcripts recorded before the platform existed, whose participants have
no row in `users` at all and therefore no way to derive a cohort via the
normal users.cohort_id join. Nullable; existing/older rows without one are
simply untagged (no filter applies to them, they're not excluded either).
"""

from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import sqlite3
import re

import pandas as pd

# -----------------------------------------------------------------------
# DB path
# -----------------------------------------------------------------------
def _find_db() -> Path:
    """Return the real responses.db path.

    Bug fixed 2026-08-04: this used to only search parent directories for a
    file literally named "responses.db", which never found the real
    persistent-volume path on Railway (SQLITE_PATH=/app/data/responses.db,
    not a parent of this file's location). It was silently resolving to a
    file in the container's ephemeral filesystem instead, so anything
    written through this module (interview / observer transcripts) never
    actually persisted across a redeploy. SQLITE_PATH is now checked first,
    matching how every other part of the app resolves this same file
    (see core/db_utils.py, teacher_dashboard.py:_get_responses_db_path()).
    """
    import os
    env_path = os.getenv("SQLITE_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "responses.db"
        if candidate.exists():
            return candidate
    return here.parents[min(3, len(here.parents)-1)] / "responses.db"

_DB_PATH = _find_db()


def _get_ts_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or _DB_PATH)


# -----------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------

def init_transcript_table(db_path: Optional[Path] = None) -> None:
    """Create transcripts table if not present. Safe to call multiple times.

    Also idempotently migrates in the cohort_id column (added 2026-08-08)
    for tables created before it existed — ALTER TABLE ADD COLUMN, guarded
    by a check so re-running this never errors on an already-migrated table.
    """
    conn = _get_ts_conn(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id TEXT NOT NULL,
            source_type    TEXT NOT NULL,
            content        TEXT NOT NULL,
            uploaded_by    TEXT NOT NULL,
            uploaded_at    TEXT NOT NULL,
            cohort_id      TEXT
        )
    """)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(transcripts)").fetchall()}
    if "cohort_id" not in existing_cols:
        conn.execute("ALTER TABLE transcripts ADD COLUMN cohort_id TEXT")
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------
# Text extraction
# -----------------------------------------------------------------------

def _parse_vtt(text: str) -> str:
    """Strip WebVTT header, timestamps, and cue ids — return spoken text only."""
    lines = text.splitlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line or line == "WEBVTT":
            continue
        if re.match(r"\d{2}:\d{2}:\d{2}\.\d+ --> \d{2}:\d{2}:\d{2}\.\d+", line):
            continue
        if line.startswith("NOTE") or line.isdigit():
            continue
        # Remove speaker labels like "Speaker 1:" at start of line
        line = re.sub(r"^[A-Za-z0-9 ]+:\s*", "", line)
        if line:
            out.append(line)
    return " ".join(out)


def _extract_text(file_obj, filename: str) -> str:
    """Extract plain text from a Streamlit UploadedFile."""
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    raw = file_obj.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)

    if filename.lower().endswith(".vtt"):
        return _parse_vtt(text)

    if filename.lower().endswith(".pdf"):
        try:
            import pdfplumber, io as _io
            with pdfplumber.open(_io.BytesIO(raw if isinstance(raw, bytes)
                                             else raw.encode())) as pdf:
                return " ".join(p.extract_text() or "" for p in pdf.pages).strip()
        except ImportError:
            pass  # fall through to plain text
        try:
            import pypdf, io as _io2
            reader = pypdf.PdfReader(_io2.BytesIO(raw if isinstance(raw, bytes)
                                                  else raw.encode()))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as e:
            raise ValueError(f"Could not read PDF '{filename}': {e}")

    return text.strip()


def _infer_pid(filename: str) -> str:
    """
    Derive participant ID from filename stem.

    Strips known programme/content suffixes iteratively until none remain,
    so compound names like 'aa_ub_b4ai_transcript' resolve to 'aa_ub'
    rather than stopping at 'aa_ub_b4ai' after a single pass.

    Known suffixes stripped (case-insensitive):
        _transcript, _b4ai, _basics4ai, _interview, _recording, _reflection
    """
    stem = Path(filename).stem
    _SUFFIXES = re.compile(
        r"_(transcript|b4ai|basics4ai|interview|recording|reflection)$",
        flags=re.I,
    )
    while True:
        new_stem = _SUFFIXES.sub("", stem)
        if new_stem == stem:
            break
        stem = new_stem
    return stem.strip("_").replace(" ", "_")


# -----------------------------------------------------------------------
# Persistent store — upload
# -----------------------------------------------------------------------

def upload_transcripts_persistent(
    files: List,
    source_type: str,
    uploaded_by: str,
    db_path: Optional[Path] = None,
    cohort_id: Optional[str] = None,
) -> Dict:
    """
    Upload transcript files to the persistent DB store.

    Parameters
    ----------
    files        : list of Streamlit UploadedFile objects
    source_type  : "interview" or "reflection"
    uploaded_by  : username of uploader
    db_path      : optional path to responses.db
    cohort_id    : optional cohort tag applied to every file in this batch

    Returns
    -------
    dict: {"uploaded": int, "skipped": int, "errors": dict{fname: msg}}
    """
    init_transcript_table(db_path)
    conn = _get_ts_conn(db_path)
    now  = datetime.utcnow().isoformat()

    uploaded = 0
    skipped  = 0
    errors   = {}

    for f_obj in files:
        fname = getattr(f_obj, "name", str(f_obj))
        try:
            text = _extract_text(f_obj, fname)
            if not text:
                errors[fname] = "No text content extracted."
                continue

            pid = _infer_pid(fname)

            existing = conn.execute(
                "SELECT id FROM transcripts WHERE participant_id=? AND source_type=?",
                (pid, source_type)
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE transcripts
                       SET content=?, uploaded_by=?, uploaded_at=?, cohort_id=?
                       WHERE participant_id=? AND source_type=?""",
                    (text, uploaded_by, now, cohort_id, pid, source_type)
                )
                skipped += 1
            else:
                conn.execute(
                    """INSERT INTO transcripts
                       (participant_id, source_type, content, uploaded_by, uploaded_at, cohort_id)
                       VALUES (?,?,?,?,?,?)""",
                    (pid, source_type, text, uploaded_by, now, cohort_id)
                )
                uploaded += 1

        except Exception as e:
            errors[fname] = str(e)

    conn.commit()
    conn.close()
    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}


# -----------------------------------------------------------------------
# Persistent store — read/delete
# -----------------------------------------------------------------------

def get_persistent_transcripts(
    source_type: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return all stored transcripts as a DataFrame.

    Includes char_count (length of content) without loading full text into
    memory — safe to call for large transcript stores.
    """
    init_transcript_table(db_path)
    conn = _get_ts_conn(db_path)
    if source_type:
        rows = conn.execute(
            "SELECT id, participant_id, source_type, uploaded_by, uploaded_at, "
            "length(content) as char_count, cohort_id "
            "FROM transcripts WHERE source_type=? ORDER BY participant_id",
            (source_type,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, participant_id, source_type, uploaded_by, uploaded_at, "
            "length(content) as char_count, cohort_id "
            "FROM transcripts ORDER BY source_type, participant_id"
        ).fetchall()
    conn.close()
    cols = ["id", "participant_id", "source_type",
            "uploaded_by", "uploaded_at", "char_count", "cohort_id"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def get_transcript_count(
    source_type: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Count stored transcripts, optionally filtered by source_type."""
    init_transcript_table(db_path)
    conn = _get_ts_conn(db_path)
    if source_type:
        n = conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE source_type=?",
            (source_type,)
        ).fetchone()[0]
    else:
        n = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    conn.close()
    return n


def delete_transcript(
    participant_id: str,
    source_type: str,
    db_path: Optional[Path] = None,
) -> None:
    """Delete a stored transcript by participant_id + source_type."""
    init_transcript_table(db_path)
    conn = _get_ts_conn(db_path)
    conn.execute(
        "DELETE FROM transcripts WHERE participant_id=? AND source_type=?",
        (participant_id, source_type)
    )
    conn.commit()
    conn.close()


def count_transcripts_for_participant(
    participant_id: str,
    db_path: Optional[Path] = None,
) -> int:
    """Count stored transcripts (any source_type) for one participant."""
    init_transcript_table(db_path)
    conn = _get_ts_conn(db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM transcripts WHERE participant_id=?",
        (participant_id,)
    ).fetchone()[0]
    conn.close()
    return n


def delete_all_transcripts_for_participant(
    participant_id: str,
    db_path: Optional[Path] = None,
) -> int:
    """
    Delete every stored transcript for a participant, across all
    source_types (interview, observer, etc.). Used when a user account
    is deleted, so transcripts don't become orphaned data tied to a
    username that no longer exists. Returns the number of rows deleted.
    """
    init_transcript_table(db_path)
    conn = _get_ts_conn(db_path)
    cursor = conn.execute(
        "DELETE FROM transcripts WHERE participant_id=?",
        (participant_id,)
    )
    n = cursor.rowcount
    conn.commit()
    conn.close()
    return n


# -----------------------------------------------------------------------
# Single-transcript save — called by admin_dashboard explicit ID mapping
# -----------------------------------------------------------------------

def save_transcript(
    participant_id: str,
    content: str,
    source_type: str = "interview",
    filename: str = "",
    uploaded_by: str = "admin",
    db_path: Optional[Path] = None,
    cohort_id: Optional[str] = None,
) -> None:
    """
    Save or update a single transcript with an explicitly supplied participant ID.

    Called by the admin dashboard transcript upload panel, where the admin
    maps each file to a participant ID manually.  Inserts a new row, or
    updates the existing one if the (participant_id, source_type) pair
    already exists.

    Parameters
    ----------
    participant_id : str  — must match the student's username in responses.db
                      (or, for pre-pilot transcripts with no registered user,
                      any consistent identifier — cohort_id is what makes
                      those usable for cohort-scoped analysis regardless)
    content        : str  — full plain-text transcript content
    source_type    : str  — 'interview' (default) | 'reflection' | 'observer'
    filename       : str  — original filename, stored for audit purposes
    uploaded_by    : str  — username of the admin who uploaded the file
    db_path        : Path | None — optional override for responses.db location
    cohort_id      : str | None — cohort tag, independent of any user record
    """
    init_transcript_table(db_path)
    conn = _get_ts_conn(db_path)
    now  = datetime.utcnow().isoformat()

    existing = conn.execute(
        "SELECT id FROM transcripts WHERE participant_id=? AND source_type=?",
        (participant_id, source_type)
    ).fetchone()

    if existing:
        conn.execute(
            """UPDATE transcripts
               SET content=?, uploaded_by=?, uploaded_at=?, cohort_id=?
               WHERE participant_id=? AND source_type=?""",
            (content, uploaded_by, now, cohort_id, participant_id, source_type)
        )
    else:
        conn.execute(
            """INSERT INTO transcripts
               (participant_id, source_type, content, uploaded_by, uploaded_at, cohort_id)
               VALUES (?,?,?,?,?,?)""",
            (participant_id, source_type, content, uploaded_by, now, cohort_id)
        )

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------
# Per-run (in-memory) upload — returns list of transcript dicts
# -----------------------------------------------------------------------

def upload_transcripts_per_run(
    files: List,
    source_type: str = "interview",
) -> List[Dict]:
    """
    Parse uploaded files and return as a list of transcript dicts.
    Nothing is written to the DB.

    Returns list of {"participant_id": str, "source_type": str, "content": str}
    """
    results = []
    for f_obj in files:
        fname = getattr(f_obj, "name", str(f_obj))
        try:
            text = _extract_text(f_obj, fname)
            if text:
                results.append({
                    "participant_id": _infer_pid(fname),
                    "source_type":    source_type,
                    "content":        text,
                })
        except Exception:
            pass
    return results


# -----------------------------------------------------------------------
# Unified loader — called by ITA and DTA pipelines
# -----------------------------------------------------------------------

def load_for_analysis(
    source: str,
    source_type: str = "interview",
    per_run_files: Optional[List] = None,
    module_id: Optional[str] = None,
    db_path: Optional[Path] = None,
    cohort_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load transcripts from a named source.

    Parameters
    ----------
    source       : "persistent" | "per_run" | "responses"
    source_type  : "interview" | "reflection" | "observer"
    per_run_files: list of Streamlit file objects (required when source="per_run")
    module_id    : optional filter for "responses" source
    db_path      : optional path to responses.db
    cohort_ids   : optional list — when provided, only "persistent"-source
                   transcripts tagged with one of these cohort_id values are
                   returned. Untagged transcripts (cohort_id IS NULL) are
                   excluded when a filter is active, since they can't be
                   attributed to any of the selected cohorts.

    Returns
    -------
    List of {"participant_id": str, "source_type": str, "content": str, "cohort_id": str|None}
    """
    if source == "persistent":
        init_transcript_table(db_path)
        conn = _get_ts_conn(db_path)
        query = "SELECT participant_id, source_type, content, cohort_id FROM transcripts WHERE 1=1"
        params: list = []
        if source_type:
            query += " AND source_type=?"
            params.append(source_type)
        if cohort_ids:
            placeholders = ",".join("?" for _ in cohort_ids)
            query += f" AND cohort_id IN ({placeholders})"
            params.extend(cohort_ids)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [{"participant_id": r[0], "source_type": r[1], "content": r[2], "cohort_id": r[3]}
                for r in rows]

    elif source == "per_run":
        if not per_run_files:
            return []
        return upload_transcripts_per_run(per_run_files, source_type)

    elif source == "responses":
        # Load reflection notes from the responses table
        db = db_path or _find_db()
        if not db.exists():
            return []
        conn = sqlite3.connect(db)
        query = ("SELECT user_id, instrument_name, response_value FROM responses "
                 "WHERE instrument_name LIKE '%module_reflections%' "
                 "AND response_value IS NOT NULL")
        params = []
        if module_id:
            query += " AND instrument_name LIKE ?"
            params.append(f"%{module_id}%")
        rows = conn.execute(query, params).fetchall()
        conn.close()
        results = []
        for uid, iname, rval in rows:
            if rval and str(rval).strip():
                m = re.match(r"module_?(\d+)", iname, re.I)
                results.append({
                    "participant_id": uid,
                    "source_type":    "reflection",
                    "module_id":      f"module_{m.group(1)}" if m else iname,
                    "content":        str(rval),
                })
        return results

    return []
