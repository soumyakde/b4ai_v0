"""
Phase 4 — ITA Pipeline Unit Tests
test_ita_pipeline.py

Groups 1-7: No API calls — test chunking, DB ops, JSON parsing,
            run management.
Group 8:    Optional live API calls (requires --live flag + real keys).

Run:
    python tests/phase4/test_ita_pipeline.py
    python tests/phase4/test_ita_pipeline.py --live
"""

import sys, os, importlib.util, tempfile, sqlite3
from pathlib import Path
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR,"..","..", "core","analytics","llm","ita_pipeline.py"),
    os.path.join(_THIS_DIR, "ita_pipeline.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES
              if os.path.exists(os.path.normpath(p))), None)
if not _path:
    print("ERROR: ita_pipeline.py not found"); sys.exit(1)

# Mock sentence_transformers if not available (disk space constraint on CI)
try:
    import sentence_transformers
except ImportError:
    import types, hashlib, numpy as _np
    _mock = types.ModuleType("sentence_transformers")
    class _MockModel:
        def encode(self, texts, **kw):
            result = []
            for t in texts:
                h = hashlib.md5(t.encode()).digest()
                v = _np.frombuffer(h*4, dtype=_np.uint8)[:64].astype(_np.float32)
                v = v / (_np.linalg.norm(v) + 1e-10)
                result.append(v)
            return _np.array(result)
    _mock.SentenceTransformer = lambda n: _MockModel()
    sys.modules["sentence_transformers"] = _mock

_spec = importlib.util.spec_from_file_location("ita_pipeline", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_phase1         = _mod.run_phase1
run_phase2         = _mod.run_phase2
run_phase3         = _mod.run_phase3
run_phase4         = _mod.run_phase4
run_phase5         = _mod.run_phase5
run_phase6         = _mod.run_phase6
run_phase2_dedup   = _mod.run_phase2_dedup
create_run         = _mod.create_run
save_phase_result  = _mod.save_phase_result
load_phase_result  = _mod.load_phase_result
get_run            = _mod.get_run
list_runs          = _mod.list_runs
_parse_json_response = _mod._parse_json_response
_estimate_tokens   = _mod._estimate_tokens
SYSTEM_PROMPT      = _mod.SYSTEM_PROMPT
LIVE = "--live" in sys.argv

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

def make_db():
    t = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    t.close()
    return Path(t.name)

# -----------------------------------------------------------------------
# Group 1 — _estimate_tokens
# -----------------------------------------------------------------------
print("\n[Group 1] _estimate_tokens")

check("empty string",    _estimate_tokens(""), 1)
check("4 chars = 1 tok", _estimate_tokens("test"), 1)
check("100 chars",       _estimate_tokens("a" * 100), 25)
check("2500 chars",      _estimate_tokens("a" * 2500), 625)

# -----------------------------------------------------------------------
# Group 2 — run_phase1: basic chunking
# -----------------------------------------------------------------------
print("\n[Group 2] run_phase1 — basic chunking")

# Short transcript — fits in one chunk
short = [{"participant_id": "ant_ub",
          "content": "I learned that AI can recognize images. "
                     "The maze game was fun. I want to learn more."}]
chunks = run_phase1(short, chunk_size=2500)
check("1 short transcript → 1 chunk", len(chunks), 1)
check("chunk_index=0",                chunks[0]["chunk_index"], 0)
check("participant_id preserved",     chunks[0]["participant_id"], "ant_ub")
check_true("content non-empty",       len(chunks[0]["content"]) > 0)
check_true("token_estimate present",  "token_estimate" in chunks[0])

# Long transcript — must split
long_text = ("AI is interesting and I learned many things about it. " * 60)
long = [{"participant_id": "gl_ub", "content": long_text}]
chunks_long = run_phase1(long, chunk_size=200)
check_true("long transcript splits",  len(chunks_long) > 1)
check_true("all chunks under limit",
    all(c["token_estimate"] <= 250 for c in chunks_long))
check_true("chunk indices sequential",
    [c["chunk_index"] for c in chunks_long] ==
    list(range(len(chunks_long))))

# Multiple transcripts
multi = [
    {"participant_id": "u1", "content": "Short text one."},
    {"participant_id": "u2", "content": "Short text two."},
    {"participant_id": "u3", "content": "Short text three."},
]
chunks_multi = run_phase1(multi, chunk_size=2500)
check("3 transcripts → 3 chunks", len(chunks_multi), 3)
check_true("global indices unique",
    len({c["chunk_index"] for c in chunks_multi}) == 3)

# Empty input
check("empty transcripts", run_phase1([]), [])

# Transcript with empty content skipped
empty_content = [{"participant_id": "x", "content": ""},
                 {"participant_id": "y", "content": "Real content here."}]
chunks_ec = run_phase1(empty_content)
check("empty content skipped", len(chunks_ec), 1)

# -----------------------------------------------------------------------
# Group 3 — _parse_json_response
# -----------------------------------------------------------------------
print("\n[Group 3] _parse_json_response")

# Clean JSON
r1 = _parse_json_response('{"codes": [{"name": "test", "chunk_index": 0}]}')
check_true("clean JSON parsed",   r1 is not None)
check_true("codes key present",   "codes" in r1)

# JSON with markdown fences
r2 = _parse_json_response('```json\n{"themes": []}\n```')
check_true("fenced JSON parsed",  r2 is not None)
check_true("themes key present",  "themes" in r2)

# JSON with leading text
r3 = _parse_json_response('Here is the result:\n{"name": "AI learning"}')
check_true("leading text handled", r3 is not None)

# Invalid JSON
r4 = _parse_json_response("This is not JSON at all.")
check("invalid JSON → None",       r4, None)

# Empty string
r5 = _parse_json_response("")
check("empty string → None",       r5, None)

# Nested JSON
r6 = _parse_json_response('{"themes": [{"name": "t1", "code_indices": [0,1,2]}]}')
check_true("nested JSON parsed",   r6 is not None)
check("code_indices present",      r6["themes"][0]["code_indices"], [0,1,2])

# -----------------------------------------------------------------------
# Group 4 — DB: create_run, get_run, list_runs
# -----------------------------------------------------------------------
print("\n[Group 4] Run management — create, get, list")

db = make_db()
run_id = create_run("claude", 0.0, "interview", "teacher1", db_path=db)
check_true("run_id is UUID string", len(run_id) == 36 and "-" in run_id)

run = get_run(run_id, db_path=db)
check_true("get_run returns dict",  isinstance(run, dict))
check("run model=claude",           run["model"],       "claude")
check("run temperature=0.0",        run["temperature"], 0.0)
check("run status=created",         run["status"],      "created")
check("run phase_reached=0",        run["phase_reached"], 0)
check("run source_type",            run["source_type"], "interview")

# Create second run
run_id2 = create_run("gemini", 0.5, "reflection", "teacher1", db_path=db)
runs = list_runs("teacher1", db_path=db)
check("list_runs returns 2",        len(runs), 2)
check_true("sorted newest first",   runs[0]["created_at"] >= runs[1]["created_at"])

# get_run nonexistent
check("get nonexistent run",        get_run("nonexistent", db_path=db), None)

# -----------------------------------------------------------------------
# Group 5 — DB: save and load phase results
# -----------------------------------------------------------------------
print("\n[Group 5] save_phase_result / load_phase_result")

db2   = make_db()
rid   = create_run("gpt", 0.0, "interview", "admin", db_path=db2)

# Save phase 1
phase1_data = {"phase": 1, "chunks": [{"chunk_index": 0, "content": "test"}]}
save_phase_result(rid, 1, phase1_data, db_path=db2)

loaded = load_phase_result(rid, 1, db_path=db2)
check_true("loaded is dict",        isinstance(loaded, dict))
check("phase key preserved",        loaded["phase"], 1)
check_true("chunks preserved",      "chunks" in loaded)

# Phase reached updated
run2 = get_run(rid, db_path=db2)
check("phase_reached updated to 1", run2["phase_reached"], 1)

# Save phase 6 → status=complete
save_phase_result(rid, 6, {"phase": 6, "report_text": "Final report."}, db_path=db2)
run3 = get_run(rid, db_path=db2)
check("status=complete after phase 6", run3["status"], "complete")
check("phase_reached=6",               run3["phase_reached"], 6)

# Overwrite phase 1
save_phase_result(rid, 1, {"phase": 1, "chunks": [{"chunk_index": 0, "content": "updated"}]}, db_path=db2)
loaded2 = load_phase_result(rid, 1, db_path=db2)
check("overwrite works", loaded2["chunks"][0]["content"], "updated")

# Load nonexistent phase
check("load missing phase → None",  load_phase_result(rid, 99, db_path=db2), None)

# -----------------------------------------------------------------------
# Group 6 — run_phase2_dedup (structural, no API)
# -----------------------------------------------------------------------
print("\n[Group 6] run_phase2_dedup — structural")

sample_codes = [
    {"name": "AI learning",   "description": "How machines learn.", "quote": "q1", "chunk_index": 0},
    {"name": "Ethics AI",     "description": "Fairness issues.",    "quote": "q2", "chunk_index": 1},
    {"name": "Game activity", "description": "Maze engagement.",    "quote": "q3", "chunk_index": 2},
]
dedup = run_phase2_dedup(sample_codes, threshold=0.85)
check_true("returns dict",          isinstance(dedup, dict))
check_true("codes_dedup key",       "codes_dedup" in dedup)
check_true("n_before key",          "n_before" in dedup)
check_true("n_after key",           "n_after" in dedup)
check_true("n_removed key",         "n_removed" in dedup)
check("n_before=3",                 dedup["n_before"], 3)
check_true("n_after <= n_before",   dedup["n_after"] <= dedup["n_before"])
check_true("n_removed consistent",
    dedup["n_removed"] == dedup["n_before"] - dedup["n_after"])

# Empty codes
dedup_empty = run_phase2_dedup([], threshold=0.85)
check("empty codes: n_before=0",   dedup_empty["n_before"], 0)
check("empty codes: n_after=0",    dedup_empty["n_after"],  0)

# -----------------------------------------------------------------------
# Group 7 — SYSTEM_PROMPT content
# -----------------------------------------------------------------------
print("\n[Group 7] SYSTEM_PROMPT content")

check_true("contains Basics4AI",   "Basics4AI" in SYSTEM_PROMPT)
check_true("contains Braun",       "Braun" in SYSTEM_PROMPT)
check_true("contains inductive",   "inductive" in SYSTEM_PROMPT)
check_true("mentions age range",   "10" in SYSTEM_PROMPT and "14" in SYSTEM_PROMPT)
check_true("non-empty",            len(SYSTEM_PROMPT) > 200)

# -----------------------------------------------------------------------
# Group 8 — Live phase tests (requires --live + API keys + sentence-transformers)
# -----------------------------------------------------------------------
print(f"\n[Group 8] Live pipeline test {'(ENABLED)' if LIVE else '(SKIPPED — run with --live)'}")

if LIVE:
    # Minimal 3-transcript dataset
    test_transcripts = [
        {"participant_id": "p1", "content":
         "I found the AI activities really interesting. Learning about how "
         "AI recognizes images was exciting. The fairness discussion made me "
         "think about who makes these systems. I enjoyed working with my team."},
        {"participant_id": "p2", "content":
         "The maze game was fun and helped me understand how AI searches. "
         "I was surprised that AI can learn from examples. The robot activity "
         "showed me that AI needs lots of data. I want to know more about bias."},
        {"participant_id": "p3", "content":
         "AI literacy is important for everyone. The plugged activities were "
         "better for me than the unplugged ones. I liked when we discussed "
         "real AI examples like recommendation systems. Fairness in AI matters."},
    ]

    print("\n  Phase 1: Chunking...")
    chunks = run_phase1(test_transcripts, chunk_size=2500)
    check("phase1: 3 chunks", len(chunks), 3)
    print(f"  ℹ️  {len(chunks)} chunks produced")

    db3    = make_db()
    run_id = create_run("claude", 0.0, "interview", "test_user", db_path=db3)

    print("\n  Phase 2: Generating codes (Claude, T=0)...")
    p2 = run_phase2(chunks, "claude", 0.0, n_codes=3,
                    run_id=run_id, db_path=db3)
    check_true("phase2: no error",    not p2.get("errors"))
    check_true("phase2: codes list",  isinstance(p2.get("codes"), list))
    check_true("phase2: codes > 0",   len(p2.get("codes",[])) > 0)
    print(f"  ℹ️  {len(p2['codes'])} codes generated")
    for c in p2["codes"][:3]:
        print(f"  ℹ️  Code: '{c.get('name','')}' (chunk {c.get('chunk_index','')})")

    print("\n  Phase 2b: Deduplicating...")
    dedup = run_phase2_dedup(p2["codes"], threshold=0.80)
    print(f"  ℹ️  {dedup['n_before']} → {dedup['n_after']} codes after dedup")

    print("\n  Phase 3: Searching themes (T=0)...")
    p3 = run_phase3(dedup["codes_dedup"], "claude", 0.0, n_themes=3,
                    run_id=run_id, db_path=db3)
    check_true("phase3: no error",   p3.get("error") is None)
    check_true("phase3: themes list",isinstance(p3.get("themes"), list))
    print(f"  ℹ️  {len(p3.get('themes',[]))} themes identified")
    for t in p3.get("themes",[]):
        print(f"  ℹ️  Theme: '{t.get('name','')}'")

    print("\n  Phase 5: Defining themes...")
    p5 = run_phase5(p3["themes"], dedup["codes_dedup"], "claude", 0.0,
                    run_id=run_id, db_path=db3)
    check_true("phase5: themes_defined", len(p5.get("themes_defined",[])) > 0)
    print(f"  ℹ️  Defined themes:")
    for t in p5.get("themes_defined",[]):
        print(f"  ℹ️  '{t.get('name','')}': {str(t.get('summary',''))[:80]}...")

    print("\n  Phase 6: Writing report...")
    p6 = run_phase6(p5["themes_defined"], dedup["codes_dedup"], "claude", 0.0,
                    run_id=run_id, db_path=db3)
    check_true("phase6: report_text",  bool(p6.get("report_text")))
    check_true("phase6: no error",     p6.get("error") is None)
    print(f"  ℹ️  Report length: {len(p6.get('report_text',''))} chars")
    print(f"  ℹ️  Report preview: {p6.get('report_text','')[:300]}...")

    # Verify all phases saved in DB
    for ph in [2, 3, 5, 6]:
        loaded = load_phase_result(run_id, ph, db_path=db3)
        check_true(f"phase {ph} saved to DB", loaded is not None)

    final_run = get_run(run_id, db_path=db3)
    check("run status=complete", final_run["status"], "complete")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — ita_pipeline verified.")
else:
    print("  ❌ SOME TESTS FAILED — review above.")
print('='*60)
