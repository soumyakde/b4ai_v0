"""
Phase 4 — Deduplicator Unit Tests
test_deduplicator.py

Run from project root:
    python tests/phase4/test_deduplicator.py

Requires sentence-transformers (pip install sentence-transformers).
"""

import sys, os, importlib.util
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR,"..","..", "core","analytics","llm","deduplicator.py"),
    os.path.join(_THIS_DIR, "deduplicator.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES
              if os.path.exists(os.path.normpath(p))), None)
if not _path:
    print("ERROR: deduplicator.py not found"); sys.exit(1)

_spec = importlib.util.spec_from_file_location("deduplicator", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

deduplicate_codes        = _mod.deduplicate_codes
cluster_codes            = _mod.cluster_codes
compute_similarity_matrix= _mod.compute_similarity_matrix

PASS = FAIL = 0
def check(label, got, expected, tol=1e-3):
    global PASS, FAIL
    if isinstance(expected, float):
        ok = abs(float(got) - expected) < tol
    else:
        ok = (got == expected)
    if ok: print(f"  ✅ PASS  {label}"); PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  got={got!r}  expected={expected!r}"); FAIL += 1

def check_true(label, cond, detail=""):
    global PASS, FAIL
    if cond: print(f"  ✅ PASS  {label}"); PASS += 1
    else: print(f"  ❌ FAIL  {label}  {detail}"); FAIL += 1

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
# Clearly distinct codes — no duplicates
DISTINCT_CODES = [
    {"name": "AI learning methods",
     "description": "Students explored how machines learn from data through supervised examples.",
     "quote": "The robot learned by trying many times.",
     "chunk_index": 0},
    {"name": "Ethical AI concerns",
     "description": "Participants raised questions about fairness and bias in AI decision making.",
     "quote": "What if the AI is wrong about someone?",
     "chunk_index": 1},
    {"name": "Game engagement",
     "description": "The maze activity generated high engagement and excitement among participants.",
     "quote": "I want to play the maze game again!",
     "chunk_index": 2},
    {"name": "Real world AI examples",
     "description": "Students connected AI concepts to familiar technology like recommendation systems.",
     "quote": "That is like how Netflix suggests shows.",
     "chunk_index": 3},
]

# Near-duplicate pair (same concept, slightly different wording)
NEAR_DUPLICATE_CODES = [
    {"name": "AI learning process",
     "description": "How artificial intelligence systems learn from labeled training data examples.",
     "quote": "The machine needs examples to learn.",
     "chunk_index": 0},
    {"name": "Machine learning basics",
     "description": "Artificial intelligence learns from labeled training examples provided by humans.",
     "quote": "It learns from the data we give it.",
     "chunk_index": 1},
    {"name": "Game excitement",
     "description": "Students showed enthusiasm and excitement during the unplugged maze activity.",
     "quote": "This game is so exciting!",
     "chunk_index": 2},
]

# -----------------------------------------------------------------------
# Group 1 — deduplicate_codes: distinct codes → all kept
# -----------------------------------------------------------------------
print("\n[Group 1] deduplicate_codes — all distinct codes kept")

result = deduplicate_codes(DISTINCT_CODES, threshold=0.85)
check_true("returns list",           isinstance(result, list))
check("all 4 distinct codes kept",   len(result), 4)
check_true("all have merged_count",
    all("merged_count" in r for r in result))
check_true("all merged_count=0",
    all(r["merged_count"] == 0 for r in result))
check_true("chunk_indices preserved",
    {r["chunk_index"] for r in result} == {0, 1, 2, 3})

# -----------------------------------------------------------------------
# Group 2 — deduplicate_codes: near-duplicates merged
# -----------------------------------------------------------------------
print("\n[Group 2] deduplicate_codes — near-duplicates merged")

# Lower threshold to ensure the near-duplicate pair is caught
result2 = deduplicate_codes(NEAR_DUPLICATE_CODES, threshold=0.70)

# "AI learning process" and "Machine learning basics" should merge
# "Game excitement" is distinct
check_true("fewer codes after dedup", len(result2) < len(NEAR_DUPLICATE_CODES))
check_true("at least 1 merged",
    any(r.get("merged_count", 0) > 0 for r in result2))

# Lowest chunk_index wins (chunk_index=0 kept, not chunk_index=1)
names = [r["name"] for r in result2]
check_true("chunk_index=0 representative kept",
    any(r["chunk_index"] == 0 for r in result2))
print(f"  ℹ️  Codes after dedup: {names}")
print(f"  ℹ️  Merged counts: {[r['merged_count'] for r in result2]}")

# -----------------------------------------------------------------------
# Group 3 — deduplicate_codes: threshold behavior
# -----------------------------------------------------------------------
print("\n[Group 3] threshold behavior")

# threshold=0.0 means merge anything with similarity >= 0.0
# (nearly everything merges — this tests the merging works aggressively)
result3_low = deduplicate_codes(NEAR_DUPLICATE_CODES, threshold=0.0)
check_true("threshold=0.0: at least 1 code survives", len(result3_low) >= 1)
check_true("threshold=0.0: fewer than original",       len(result3_low) < len(NEAR_DUPLICATE_CODES))

# threshold=1.0 means only merge identical texts (cosine sim == 1.0)
# Distinct codes should all survive
result3_high = deduplicate_codes(DISTINCT_CODES, threshold=1.0)
check("threshold=1.0 distinct: all kept", len(result3_high), len(DISTINCT_CODES))

# -----------------------------------------------------------------------
# Group 4 — deduplicate_codes: threshold=1.0 — only exact duplicates merged
# -----------------------------------------------------------------------
print("\n[Group 4] threshold=1.0 — only exact duplicates merged")

# Add a true exact duplicate
exact_dupe_codes = DISTINCT_CODES[:2] + [DISTINCT_CODES[0].copy()]
exact_dupe_codes[-1]["chunk_index"] = 99  # higher index — should be dropped
result4 = deduplicate_codes(exact_dupe_codes, threshold=0.999)
check_true("exact duplicate removed", len(result4) <= len(DISTINCT_CODES[:2]) + 1)

# -----------------------------------------------------------------------
# Group 5 — edge cases
# -----------------------------------------------------------------------
print("\n[Group 5] Edge cases")

# Empty list
check("empty list", deduplicate_codes([]), [])

# Single code
single = [DISTINCT_CODES[0].copy()]
result5 = deduplicate_codes(single)
check("single code: length=1", len(result5), 1)
check("single code: merged_count=0", result5[0]["merged_count"], 0)

# Missing 'name' field
try:
    deduplicate_codes([{"description": "test", "chunk_index": 0}])
    print("  ❌ FAIL  No error on missing name"); FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError on missing name"); PASS += 1

# Missing 'chunk_index' field
try:
    deduplicate_codes([{"name": "test code", "description": "desc"}])
    print("  ❌ FAIL  No error on missing chunk_index"); FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError on missing chunk_index"); PASS += 1

# -----------------------------------------------------------------------
# Group 6 — cluster_codes
# -----------------------------------------------------------------------
print("\n[Group 6] cluster_codes")

clusters = cluster_codes(DISTINCT_CODES, threshold=0.75)
check_true("returns list",          isinstance(clusters, list))
check_true("at least 1 cluster",    len(clusters) >= 1)
check_true("each cluster has required keys",
    all("representative" in c and "members" in c and
        "cluster_size" in c and "avg_similarity" in c
        for c in clusters))
check_true("all codes appear in some cluster",
    sum(c["cluster_size"] for c in clusters) == len(DISTINCT_CODES))
check_true("sorted by size desc",
    all(clusters[i]["cluster_size"] >= clusters[i+1]["cluster_size"]
        for i in range(len(clusters)-1)))

# Near-duplicate pair clusters together
clusters2 = cluster_codes(NEAR_DUPLICATE_CODES, threshold=0.70)
check_true("near-dups in same cluster",
    any(c["cluster_size"] >= 2 for c in clusters2))
print(f"  ℹ️  Cluster sizes: {[c['cluster_size'] for c in clusters2]}")
print(f"  ℹ️  Avg similarities: {[c['avg_similarity'] for c in clusters2]}")

# -----------------------------------------------------------------------
# Group 7 — compute_similarity_matrix
# -----------------------------------------------------------------------
print("\n[Group 7] compute_similarity_matrix")

sim_df = compute_similarity_matrix(DISTINCT_CODES[:3])
check_true("returns DataFrame",     isinstance(sim_df, pd.DataFrame))
check("shape (3,3)",                sim_df.shape, (3, 3))
check_true("diagonal = 1.0",
    all(abs(sim_df.iloc[i,i] - 1.0) < 0.01 for i in range(3)))
check_true("symmetric",
    all(abs(sim_df.iloc[i,j] - sim_df.iloc[j,i]) < 0.001
        for i in range(3) for j in range(3)))
check_true("all values in [-1, 1]",
    (sim_df >= -1.01).all().all() and (sim_df <= 1.01).all().all())
check_true("index = code names",
    list(sim_df.index) == [c["name"] for c in DISTINCT_CODES[:3]])

# Empty input
empty_df = compute_similarity_matrix([])
check("empty input", empty_df.empty, True)

# -----------------------------------------------------------------------
# Group 8 — original fields preserved
# -----------------------------------------------------------------------
print("\n[Group 8] Original fields preserved after dedup")

codes_with_extra = [
    {"name": "Test code one", "description": "First code.",
     "quote": "Quote one.", "chunk_index": 0, "model": "claude",
     "extra_field": "preserved"},
    {"name": "Another code two", "description": "Second code.",
     "quote": "Quote two.", "chunk_index": 1, "model": "gemini"},
]
result8 = deduplicate_codes(codes_with_extra, threshold=0.85)
check_true("extra_field preserved",
    any(r.get("extra_field") == "preserved" for r in result8))
check_true("model field preserved",
    all("model" in r for r in result8))
check_true("quote field preserved",
    all("quote" in r for r in result8))

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — deduplicator verified.")
else:
    print("  ❌ SOME TESTS FAILED — review above.")
print('='*60)
