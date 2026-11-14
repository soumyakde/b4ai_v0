"""
core/analytics/datasets/imported_cohorts.py

Admin-imported cohorts (e.g. UB workshops, community-centre groups) that were collected outside the
platform (Qualtrics) but share the dashboard's canonical long format.

Storage: two extra tables INSIDE responses.db (so the existing backup / restore / download-databases
tools cover them automatically, and nothing else in the database is touched):

    imported_canonical     the 11 canonical columns + import_batch
    imported_demographics  user_id, grade, grade_level, gender, first_language_english, cohort_id, import_batch

The live `responses` table is never read or written here. Imported data only enters the dashboard when
load_canonical_data(include_imported=True) is called (the Teacher Dashboard sidebar switch).
No Streamlit imports: pure data handling, testable.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

CANON_COLS = ["user_id", "module_id", "instrument_key", "question_id", "response_value",
              "item_score", "construct", "grade", "submitted_at", "completed_at", "cohort_id"]
REQUIRED_COLS = ["user_id", "module_id", "instrument_key", "question_id", "response_value",
                 "item_score", "construct", "cohort_id"]
DEMO_COLS = ["user_id", "grade", "grade_level", "gender", "first_language_english"]

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{1,39}$")
_GRADES = {"Fourth (4th) grade": 4, "Fifth (5th) grade": 5, "Sixth (6th) grade": 6,
           "Seventh (7th) grade": 7, "Eighth (8th) grade": 8, "Adult": "Adult"}


def _connect(db_path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def ensure_tables(db_path) -> None:
    with _connect(db_path) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS imported_canonical (
            user_id TEXT, module_id TEXT, instrument_key TEXT, question_id TEXT, response_value TEXT,
            item_score REAL, construct TEXT, grade TEXT, submitted_at TEXT, completed_at TEXT,
            cohort_id TEXT, import_batch TEXT, imported_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS imported_demographics (
            user_id TEXT, grade TEXT, grade_level TEXT, gender TEXT, first_language_english TEXT,
            cohort_id TEXT, import_batch TEXT, imported_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_imp_canon_cohort ON imported_canonical(cohort_id)")


def _bool_or_none(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def normalise_demographics(demo: Optional[pd.DataFrame], canon: pd.DataFrame) -> pd.DataFrame:
    """One row per student in `canon`, columns DEMO_COLS + cohort_id. Students missing from `demo` get empty fields."""
    users = canon[["user_id", "cohort_id"]].drop_duplicates("user_id")
    if demo is None or demo.empty:
        d = pd.DataFrame(columns=DEMO_COLS)
    else:
        d = demo.copy()
        for c in DEMO_COLS:
            if c not in d.columns:
                d[c] = None
        d = d[DEMO_COLS].drop_duplicates("user_id")
    d["user_id"] = d["user_id"].astype(str).str.strip()
    out = users.merge(d, on="user_id", how="left")
    out["first_language_english"] = out["first_language_english"].map(_bool_or_none)
    out["gender"] = out["gender"].where(out["gender"].isin(["Male", "Female"]), None)
    out["grade"] = out["grade"].where(out["grade"].isin(_GRADES.keys()), None)
    out["grade_level"] = out["grade"].map(_GRADES)
    return out[DEMO_COLS + ["cohort_id"]]


def validate_import(canon: pd.DataFrame, demo: Optional[pd.DataFrame], known_user_ids: set,
                    registered_cohorts: set, known_instruments: Optional[set] = None,
                    replacing_cohorts: Optional[set] = None) -> Dict[str, object]:
    """Return {'errors': [...], 'warnings': [...], 'summary': DataFrame}. Any error blocks the import."""
    errors: List[str] = []
    warns: List[str] = []
    replacing = set(replacing_cohorts or ())
    missing = [c for c in REQUIRED_COLS if c not in canon.columns]
    if missing:
        return {"errors": [f"Missing required columns: {', '.join(missing)}"], "warnings": [], "summary": pd.DataFrame()}
    if canon.empty:
        return {"errors": ["The file has no rows."], "warnings": [], "summary": pd.DataFrame()}

    c = canon.copy()
    for col in ("user_id", "module_id", "instrument_key", "question_id", "construct", "cohort_id"):
        c[col] = c[col].astype(str).str.strip()
        if (c[col].isin(["", "nan", "None"])).any():
            errors.append(f"Column '{col}' has empty values.")
    bad_ids = sorted({u for u in c["user_id"] if not _ID_RE.match(u)})
    if bad_ids:
        errors.append("Student IDs must be 2-40 letters/digits/-/_ (no spaces, no names). Offending: "
                      + ", ".join(bad_ids[:8]) + (" ..." if len(bad_ids) > 8 else ""))
    clash = sorted(set(c["user_id"]) & {str(u).strip() for u in known_user_ids})
    if clash:
        errors.append("These IDs already belong to registered platform users: " + ", ".join(clash[:8]))
    # an imported ID may only belong to one cohort
    multi = c.groupby("user_id")["cohort_id"].nunique()
    if (multi > 1).any():
        errors.append("A student ID appears in more than one cohort: " + ", ".join(multi[multi > 1].index[:8]))
    cc = set(c["cohort_id"]) & {str(x) for x in registered_cohorts}
    if cc:
        errors.append("Cohort name(s) already used by registered platform cohorts: " + ", ".join(sorted(cc)))
    scores = pd.to_numeric(c["item_score"], errors="coerce")
    if scores.isna().any():
        warns.append(f"{int(scores.isna().sum())} rows have no numeric item_score (kept; ignored by scoring).")
    dup = c.duplicated(["user_id", "instrument_key", "question_id"]).sum()
    if dup:
        errors.append(f"{int(dup)} duplicate rows (same student, instrument and question).")
    if known_instruments is not None:
        unk = sorted(set(c["instrument_key"]) - set(known_instruments))
        if unk:
            errors.append("Instrument keys not known to this dashboard (the survey would not match): " + ", ".join(unk[:6]))
    # item_score must be consistent with response_value within an instrument/question
    inc = c.assign(_s=scores).groupby(["instrument_key", "question_id", "response_value"])["_s"].nunique()
    if (inc > 1).any():
        errors.append("The same answer text has different item_scores for the same question.")

    d = normalise_demographics(demo, c)
    nod = d[d[["grade", "gender", "first_language_english"]].isna().all(axis=1)]
    if len(nod):
        warns.append(f"{len(nod)} student(s) have no demographics ({', '.join(nod['user_id'][:6])}"
                     f"{' ...' if len(nod) > 6 else ''}): they appear in all-student views but drop out when a Grade/Gender/Language filter is set.")
    partial = d[d[["grade", "gender", "first_language_english"]].isna().any(axis=1) & ~d.index.isin(nod.index)]
    if len(partial):
        warns.append(f"{len(partial)} student(s) have some demographics missing.")
    summary = (c.groupby("cohort_id").agg(students=("user_id", "nunique"), rows=("user_id", "size"),
                                          modules=("module_id", "nunique")).reset_index())
    return {"errors": errors, "warnings": warns, "summary": summary}


def save_import(db_path, canon: pd.DataFrame, demo: Optional[pd.DataFrame], batch_label: str,
                replace_cohorts: bool = True) -> Dict[str, int]:
    """Write the import. With replace_cohorts, existing imported rows for the same cohort_id(s) are removed first (re-import)."""
    ensure_tables(db_path)
    c = canon.copy()
    for col in CANON_COLS:
        if col not in c.columns:
            c[col] = ""
    c = c[CANON_COLS].copy()
    c["item_score"] = pd.to_numeric(c["item_score"], errors="coerce")
    c["grade"] = c["grade"].astype(object).where(c["grade"].notna(), None)
    d = normalise_demographics(demo, c)
    c["import_batch"] = batch_label
    d["import_batch"] = batch_label
    d["first_language_english"] = d["first_language_english"].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(bool(v)))
    d["grade_level"] = d["grade_level"].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    # fill grade in canon from demographics (matches the dashboard, which carries grade on every row)
    gmap = d.set_index("user_id")["grade"]
    c["grade"] = c["user_id"].map(gmap)
    cohorts = sorted(set(c["cohort_id"]))
    with _connect(db_path) as conn:
        if replace_cohorts:
            for co in cohorts:
                conn.execute("DELETE FROM imported_canonical WHERE cohort_id=?", (co,))
                conn.execute("DELETE FROM imported_demographics WHERE cohort_id=?", (co,))
        c.to_sql("imported_canonical", conn, if_exists="append", index=False)
        d[["user_id", "grade", "grade_level", "gender", "first_language_english", "cohort_id", "import_batch"]].to_sql(
            "imported_demographics", conn, if_exists="append", index=False)
    return {"rows": int(len(c)), "students": int(c["user_id"].nunique()), "cohorts": len(cohorts)}


def list_imports(db_path) -> pd.DataFrame:
    """One row per imported cohort: cohort_id, students, rows, batch, imported_at. Empty frame if nothing imported."""
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["cohort_id", "students", "rows", "import_batch", "imported_at"])
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT cohort_id, COUNT(DISTINCT user_id) AS students, COUNT(*) AS rows, "
            "MAX(import_batch) AS import_batch, MAX(imported_at) AS imported_at "
            "FROM imported_canonical GROUP BY cohort_id ORDER BY cohort_id", conn)


def delete_imported_cohort(db_path, cohort_id: str) -> int:
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM imported_canonical WHERE cohort_id=?", (cohort_id,)).fetchone()[0]
        conn.execute("DELETE FROM imported_canonical WHERE cohort_id=?", (cohort_id,))
        conn.execute("DELETE FROM imported_demographics WHERE cohort_id=?", (cohort_id,))
    return int(n)


def load_imported(db_path, cohorts: Optional[List[str]] = None
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """(canonical_df, demographics_df, cohort_map) in the same shapes the dashboard loader returns."""
    empty_c = pd.DataFrame(columns=CANON_COLS)
    empty_d = pd.DataFrame(columns=DEMO_COLS)
    if not Path(db_path).exists():
        return empty_c, empty_d, {}
    ensure_tables(db_path)
    with _connect(db_path) as conn:
        c = pd.read_sql_query("SELECT " + ",".join(CANON_COLS) + " FROM imported_canonical", conn)
        d = pd.read_sql_query("SELECT user_id, grade, grade_level, gender, first_language_english, cohort_id FROM imported_demographics", conn)
    if cohorts is not None:
        c = c[c["cohort_id"].isin(cohorts)]
        d = d[d["cohort_id"].isin(cohorts)]
    if c.empty:
        return empty_c, empty_d, {}
    d["first_language_english"] = d["first_language_english"].map(
        lambda v: True if v == "True" else (False if v == "False" else None)).astype(object)
    d["grade_level"] = d["grade_level"].map(lambda v: None if v is None else (int(v) if str(v).isdigit() else v))
    cmap = dict(zip(d["user_id"], d["cohort_id"]))
    # students present in canon but (defensively) missing from demographics
    for u, co in zip(c["user_id"], c["cohort_id"]):
        cmap.setdefault(u, co)
    return c.reset_index(drop=True), d[DEMO_COLS].reset_index(drop=True), cmap
