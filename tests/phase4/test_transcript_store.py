"""
Phase 4 — Transcript Store Unit Tests
test_transcript_store.py

Run from project root:
    python tests/phase4/test_transcript_store.py
"""

import sys, os, importlib.util, sqlite3, tempfile, io
from pathlib import Path
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR,"..","..", "core","analytics","llm","transcript_store.py"),
    os.path.join(_THIS_DIR, "transcript_store.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES
              if os.path.exists(os.path.normpath(p))), None)
if not _path:
    print("ERROR: transcript_store.py not found"); sys.exit(1)

_spec = importlib.util.spec_from_file_location("transcript_store", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

init_transcript_table          = _mod.init_transcript_table
upload_transcripts_persistent  = _mod.upload_transcripts_persistent
get_persistent_transcripts     = _mod.get_persistent_transcripts
delete_transcript              = _mod.delete_transcript
get_transcript_count           = _mod.get_transcript_count
load_transcripts_per_run       = _mod.load_transcripts_per_run
load_for_analysis              = _mod.load_for_analysis
_infer_participant_id          = _mod._infer_participant_id
_extract_text_from_file        = _mod._extract_text_from_file

PASS = FAIL = 0
def check(label, got, expected):
    global PASS, FAIL
    ok = (got == expected)
    if ok: print(f"  ✅ PASS  {label}"); PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  got={got!r}  expected={expected!r}"); FAIL += 1

def check_true(label, cond, detail=""):
    global PASS, FAIL
    if cond: print(f"  ✅ PASS  {label}"); PASS += 1
    else: print(f"  ❌ FAIL  {label}  {detail}"); FAIL += 1

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def make_temp_db() -> Path:
    """Create a temporary SQLite DB for testing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    # Add responses table for reflection tests
    conn = sqlite3.connect(tmp.name)
    conn.execute("""CREATE TABLE responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        instrument_name TEXT NOT NULL,
        question_id TEXT NOT NULL,
        response_value TEXT,
        submitted_at TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()
    return Path(tmp.name)

class FakeFile:
    """Minimal file-like object simulating Streamlit UploadedFile."""
    def __init__(self, name: str, content: str):
        self.name    = name
        self._data   = content.encode("utf-8")
        self._pos    = 0
    def read(self):
        return self._data
    def seek(self, pos):
        self._pos = pos

# -----------------------------------------------------------------------
# Group 1 — _infer_participant_id
# -----------------------------------------------------------------------
print("\n[Group 1] _infer_participant_id")

check("ant_ub_interview.txt",  _infer_participant_id("ant_ub_interview.txt"),  "ant_ub")
check("gl_ub.txt",             _infer_participant_id("gl_ub.txt"),             "gl_ub")
check("participant_01.txt",    _infer_participant_id("participant_01.txt"),    "participant_01")
check("single.txt",            _infer_participant_id("single.txt"),            "single")
check("aa_ub_session2.txt",    _infer_participant_id("aa_ub_session2.txt"),    "aa_ub")

# -----------------------------------------------------------------------
# Group 2 — _extract_text_from_file
# -----------------------------------------------------------------------
print("\n[Group 2] _extract_text_from_file — .txt")

f = FakeFile("ant_ub.txt", "This is the interview content.")
text = _extract_text_from_file(f, "ant_ub.txt")
check("txt content extracted",   text, "This is the interview content.")

f2 = FakeFile("test.csv", "col1,col2\nval1,val2")
text2 = _extract_text_from_file(f2, "test.csv")
check_true("non-txt decoded",    len(text2) > 0)

# -----------------------------------------------------------------------
# Group 3 — init_transcript_table
# -----------------------------------------------------------------------
print("\n[Group 3] init_transcript_table")

db = make_temp_db()
init_transcript_table(db)
conn = sqlite3.connect(db)
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()]
conn.close()
check_true("transcripts table created", "transcripts" in tables)

# Safe to call twice
init_transcript_table(db)
check_true("idempotent second call", True)

# -----------------------------------------------------------------------
# Group 4 — upload_transcripts_persistent
# -----------------------------------------------------------------------
print("\n[Group 4] upload_transcripts_persistent")

db = make_temp_db()
files = [
    FakeFile("ant_ub_interview.txt", "Ant's interview content here."),
    FakeFile("gl_ub_interview.txt",  "GL interview text here."),
    FakeFile("am_ub_interview.txt",  "AM interview text here."),
]
result = upload_transcripts_persistent(files, "interview", "teacher1", db)

check("3 uploaded",       result["uploaded"], 3)
check("0 skipped",        result["skipped"],  0)
check("0 errors",         len(result["errors"]), 0)
check("count = 3",        get_transcript_count("interview", db), 3)

# -----------------------------------------------------------------------
# Group 5 — duplicate upload updates existing
# -----------------------------------------------------------------------
print("\n[Group 5] Duplicate upload — updates, does not duplicate")

files2 = [FakeFile("ant_ub_interview.txt", "Updated content for ant_ub.")]
result2 = upload_transcripts_persistent(files2, "interview", "teacher1", db)
check("0 uploaded (updated)", result2["uploaded"], 0)
check("1 skipped (updated)",  result2["skipped"],  1)
check("count still 3",        get_transcript_count("interview", db), 3)

# Verify content was updated
df = get_persistent_transcripts("interview", db)
ant_row = df[df["participant_id"] == "ant_ub"].iloc[0]
check("content updated", ant_row["content"], "Updated content for ant_ub.")

# -----------------------------------------------------------------------
# Group 6 — get_persistent_transcripts
# -----------------------------------------------------------------------
print("\n[Group 6] get_persistent_transcripts")

df = get_persistent_transcripts("interview", db)
check_true("returns DataFrame",   isinstance(df, pd.DataFrame))
check("3 rows",                   len(df), 3)
check_true("correct columns",
    set(df.columns) == {"id","participant_id","source_type",
                        "content","uploaded_by","uploaded_at"})
check_true("all source_type=interview",
    (df["source_type"] == "interview").all())

# Different source_type returns empty
df_ref = get_persistent_transcripts("reflection", db)
check("reflection returns empty", len(df_ref), 0)

# None returns all
db2 = make_temp_db()
upload_transcripts_persistent(
    [FakeFile("p1_interview.txt","int"), FakeFile("p1_reflection.txt","ref")],
    "interview", "admin", db2
)
upload_transcripts_persistent(
    [FakeFile("p2_reflection.txt","ref2")],
    "reflection", "admin", db2
)
df_all = get_persistent_transcripts(None, db2)
check("None returns all types", len(df_all), 3)

# -----------------------------------------------------------------------
# Group 7 — delete_transcript
# -----------------------------------------------------------------------
print("\n[Group 7] delete_transcript")

deleted = delete_transcript("gl_ub", "interview", db)
check("delete returns True",  deleted, True)
check("count now 2",          get_transcript_count("interview", db), 2)

not_found = delete_transcript("nonexistent", "interview", db)
check("delete nonexistent returns False", not_found, False)

# -----------------------------------------------------------------------
# Group 8 — load_transcripts_per_run
# -----------------------------------------------------------------------
print("\n[Group 8] load_transcripts_per_run")

files3 = [
    FakeFile("svs_ub_session1.txt", "Session 1 content for svs."),
    FakeFile("oz_ub.txt",           "Oz interview text."),
]
per_run = load_transcripts_per_run(files3)
check("2 records loaded",     len(per_run), 2)
check_true("dicts have required keys",
    all("participant_id" in r and "content" in r for r in per_run))
check("svs_ub participant_id", per_run[0]["participant_id"], "svs_ub")
check("source_type=per_run",   per_run[0]["source_type"],    "per_run")
check("oz content correct",    per_run[1]["content"],         "Oz interview text.")

# Empty file raises ValueError
empty = [FakeFile("empty.txt", "")]
try:
    load_transcripts_per_run(empty)
    print("  ❌ FAIL  No error on empty file"); FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError on empty file: {str(e)[:60]}"); PASS += 1

# -----------------------------------------------------------------------
# Group 9 — load_for_analysis unified interface
# -----------------------------------------------------------------------
print("\n[Group 9] load_for_analysis — unified interface")

# persistent mode
records = load_for_analysis("persistent", "interview", db_path=db)
check("persistent: 2 records", len(records), 2)
check_true("persistent: dicts have content",
    all("content" in r for r in records))

# per_run mode
files4 = [FakeFile("tw_ub.txt", "TW content")]
records_pr = load_for_analysis("per_run", per_run_files=files4)
check("per_run: 1 record",     len(records_pr), 1)
check("per_run: participant",  records_pr[0]["participant_id"], "tw_ub")

# responses mode — with reflection data in DB
db3 = make_temp_db()
conn3 = sqlite3.connect(db3)
conn3.executemany(
    "INSERT INTO responses (user_id, instrument_name, question_id, response_value, submitted_at) VALUES (?,?,?,?,?)",
    [
        ("ant_ub", "module1_module_reflections", "Q1", "I learned about AI today.", "2026-01-01"),
        ("gl_ub",  "module1_module_reflections", "Q1", "The maze activity was fun.", "2026-01-01"),
        ("ant_ub", "module2_module_reflections", "Q1", "Module 2 was interesting.",  "2026-01-01"),
    ]
)
conn3.commit(); conn3.close()

records_r = load_for_analysis("responses", db_path=db3)
check("responses: 3 records",      len(records_r), 3)
check_true("responses: source_type reflection",
    all(r["source_type"] == "reflection" for r in records_r))

# module filter
records_m1 = load_for_analysis("responses", module_id="module1", db_path=db3)
check("responses module1: 2 records", len(records_m1), 2)

# invalid source raises ValueError
try:
    load_for_analysis("invalid_source")
    print("  ❌ FAIL  No error on invalid source"); FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError on invalid source"); PASS += 1

# -----------------------------------------------------------------------
# Group 10 — invalid source_type raises ValueError
# -----------------------------------------------------------------------
print("\n[Group 10] Input validation")

try:
    upload_transcripts_persistent([], "invalid_type", "admin", db)
    print("  ❌ FAIL  No error on invalid source_type"); FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError on invalid source_type"); PASS += 1

try:
    load_for_analysis("per_run", per_run_files=None)
    print("  ❌ FAIL  No error on missing per_run_files"); FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError on missing per_run_files"); PASS += 1

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — transcript_store verified.")
else:
    print("  ❌ SOME TESTS FAILED — review above.")
print('='*60)
