"""
Phase 4 — Theme Comparator Unit Tests
test_theme_comparator.py

Run from project root:
    python tests/phase4/test_theme_comparator.py
"""

import sys, os, importlib.util, types
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR,"..","..", "core","analytics","llm","theme_comparator.py"),
    os.path.join(_THIS_DIR, "theme_comparator.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES
              if os.path.exists(os.path.normpath(p))), None)
if not _path:
    print("ERROR: theme_comparator.py not found"); sys.exit(1)

# -----------------------------------------------------------------------
# Mock sentence_transformers if not installed
# (tests still verify structure; run on machine for real embeddings)
# -----------------------------------------------------------------------
try:
    import sentence_transformers
    _REAL_EMBEDDINGS = True
except ImportError:
    import hashlib
    mock_st = types.ModuleType("sentence_transformers")
    class _MockModel:
        def encode(self, texts, **kwargs):
            result = []
            for t in texts:
                h = hashlib.md5(t.encode()).digest()
                v = np.frombuffer(h*4, dtype=np.uint8)[:64].astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-10)
                result.append(v)
            return np.array(result)
    mock_st.SentenceTransformer = lambda n: _MockModel()
    sys.modules["sentence_transformers"] = mock_st
    _REAL_EMBEDDINGS = False

_spec = importlib.util.spec_from_file_location("theme_comparator", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compare_runs      = _mod.compare_runs
compare_all_runs  = _mod.compare_all_runs
get_best_match    = _mod.get_best_match
align_themes      = _mod.align_themes
_interpret_agreement = _mod._interpret_agreement
_jaccard          = _mod._jaccard

PASS = FAIL = 0
def check(label, got, expected, tol=1e-3):
    global PASS, FAIL
    if isinstance(expected, float):
        ok = abs(float(got)-expected) < tol
    else:
        ok = (got == expected)
    if ok: print(f"  ✅ PASS  {label}"); PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  got={got!r}  expected={expected!r}"); FAIL += 1

def check_true(label, cond, detail=""):
    global PASS, FAIL
    if cond: print(f"  ✅ PASS  {label}"); PASS += 1
    else: print(f"  ❌ FAIL  {label}  {detail}"); FAIL += 1

print(f"\n  Embeddings: {'real (sentence-transformers)' if _REAL_EMBEDDINGS else 'mock'}")

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
THEMES_A = [
    {"name": "AI learning mechanisms",
     "description": "How machines learn from labeled training data examples."},
    {"name": "Ethics and fairness",
     "description": "Bias and fairness concerns in AI decision making systems."},
    {"name": "Engagement and motivation",
     "description": "Student enthusiasm and intrinsic motivation during activities."},
]
# Themes B — deliberately similar to A (same model, higher temperature)
THEMES_B = [
    {"name": "Machine learning process",
     "description": "The process by which AI systems learn from training examples."},
    {"name": "AI bias and fairness",
     "description": "Concerns about discrimination and fairness in automated decisions."},
    {"name": "Student motivation",
     "description": "Intrinsic motivation and engagement levels during learning activities."},
]
# Themes C — deliberately different (different topic)
THEMES_C = [
    {"name": "Food preferences",
     "description": "What students like to eat for lunch and dinner."},
    {"name": "Sports activities",
     "description": "Physical education and after-school sports participation."},
    {"name": "Music interests",
     "description": "Student preferences for different musical genres."},
]

# -----------------------------------------------------------------------
# Group 1 — _jaccard (no embeddings needed)
# -----------------------------------------------------------------------
print("\n[Group 1] _jaccard similarity")

check("identical strings",  _jaccard("hello world", "hello world"), 1.0)
check("no overlap",         _jaccard("hello world", "foo bar"),     0.0)
check("partial overlap",    _jaccard("ai learning", "learning machine"),
      len({"ai","learning"} & {"learning","machine"}) /
      len({"ai","learning"} | {"learning","machine"}))
check("empty strings",      _jaccard("", ""), 0.0)

# -----------------------------------------------------------------------
# Group 2 — _interpret_agreement
# -----------------------------------------------------------------------
print("\n[Group 2] _interpret_agreement")

check("0.85 → strong",   _interpret_agreement(0.85), "strong")
check("0.80 → strong",   _interpret_agreement(0.80), "strong")
check("0.70 → moderate", _interpret_agreement(0.70), "moderate")
check("0.65 → moderate", _interpret_agreement(0.65), "moderate")
check("0.55 → weak",     _interpret_agreement(0.55), "weak")
check("0.50 → weak",     _interpret_agreement(0.50), "weak")
check("0.40 → poor",     _interpret_agreement(0.40), "poor")
check("0.0  → poor",     _interpret_agreement(0.0),  "poor")

# -----------------------------------------------------------------------
# Group 3 — compare_runs: structure and keys
# -----------------------------------------------------------------------
print("\n[Group 3] compare_runs — result structure")

r = compare_runs(THEMES_A, THEMES_B, "Model A T=0", "Model A T=0.5")

check("no error",           r["error"], None)
check("label_a",            r["label_a"], "Model A T=0")
check("label_b",            r["label_b"], "Model A T=0.5")
check("n_themes_a=3",       r["n_themes_a"], 3)
check("n_themes_b=3",       r["n_themes_b"], 3)
check_true("cosine_matrix is DataFrame",
    isinstance(r["cosine_matrix"], pd.DataFrame))
check("cosine_matrix shape", r["cosine_matrix"].shape, (3, 3))
check_true("jaccard_matrix is DataFrame",
    isinstance(r["jaccard_matrix"], pd.DataFrame))
check_true("agreement_matrix is DataFrame",
    isinstance(r["agreement_matrix"], pd.DataFrame))
check_true("best_matches is list",    isinstance(r["best_matches"], list))
check("best_matches length=3",        len(r["best_matches"]), 3)
check_true("overall_agreement is float",
    isinstance(r["overall_agreement"], float))
check_true("interpretation is str",
    isinstance(r["interpretation"], str))
check_true("interpretation valid",
    r["interpretation"] in ("strong","moderate","weak","poor"))

# Best matches have required keys
for bm in r["best_matches"]:
    check_true("best_match has theme_a",
        "theme_a" in bm and "best_match_b" in bm)
    check_true("best_match has scores",
        "cosine" in bm and "jaccard" in bm and "agreement" in bm)

# -----------------------------------------------------------------------
# Group 4 — compare_runs: similar vs different themes
# -----------------------------------------------------------------------
print("\n[Group 4] compare_runs — similar > different")

r_sim  = compare_runs(THEMES_A, THEMES_B, "A", "B (similar)")
r_diff = compare_runs(THEMES_A, THEMES_C, "A", "C (different)")

check_true("similar themes score higher than different",
    r_sim["overall_agreement"] > r_diff["overall_agreement"],
    f"sim={r_sim['overall_agreement']}, diff={r_diff['overall_agreement']}")

print(f"  ℹ️  A vs B (similar):   {r_sim['overall_agreement']:.4f} ({r_sim['interpretation']})")
print(f"  ℹ️  A vs C (different): {r_diff['overall_agreement']:.4f} ({r_diff['interpretation']})")

# -----------------------------------------------------------------------
# Group 5 — compare_runs: error on empty
# -----------------------------------------------------------------------
print("\n[Group 5] compare_runs — empty run handling")

r_empty = compare_runs([], THEMES_B)
check_true("error on empty run_a", r_empty["error"] is not None)

r_empty2 = compare_runs(THEMES_A, [])
check_true("error on empty run_b", r_empty2["error"] is not None)

# -----------------------------------------------------------------------
# Group 6 — compare_all_runs
# -----------------------------------------------------------------------
print("\n[Group 6] compare_all_runs")

runs = {
    "Claude T=0":   THEMES_A,
    "Claude T=0.5": THEMES_B,
    "GPT T=0":      THEMES_C,
}
all_r = compare_all_runs(runs)

check("no error",              all_r["error"], None)
check("labels count=3",        len(all_r["labels"]), 3)
check_true("summary_matrix DataFrame",
    isinstance(all_r["summary_matrix"], pd.DataFrame))
check("summary_matrix shape",  all_r["summary_matrix"].shape, (3, 3))
check_true("diagonal = 1.0",
    all(abs(all_r["summary_matrix"].iloc[i,i]-1.0)<0.01 for i in range(3)))
check_true("symmetric",
    all(
        abs(all_r["summary_matrix"].iloc[i,j] -
            all_r["summary_matrix"].iloc[j,i]) < 0.001
        for i in range(3) for j in range(3)
    ))

# Pairwise dict has 3 pairs (C(3,2)=3)
check("3 pairwise comparisons", len(all_r["pairwise"]), 3)

# < 2 runs → error
r_one = compare_all_runs({"only": THEMES_A})
check_true("error with < 2 runs", r_one["error"] is not None)

# -----------------------------------------------------------------------
# Group 7 — get_best_match
# -----------------------------------------------------------------------
print("\n[Group 7] get_best_match")

query = {"name": "AI bias concerns",
         "description": "Fairness and discrimination in AI systems."}
best, score = get_best_match(query, THEMES_B)

check_true("best match is dict",   isinstance(best, dict))
check_true("best match has name",  "name" in best)
check_true("score in [0,1]",       0 <= score <= 1)
# Should match "AI bias and fairness" in THEMES_B
check_true("matches ethics theme",
    "bias" in best["name"].lower() or "fair" in best["name"].lower(),
    f"got: {best['name']}")
print(f"  ℹ️  Query: '{query['name']}'")
print(f"  ℹ️  Best match: '{best['name']}' (score={score})")

# Empty candidates
best_e, score_e = get_best_match(query, [])
check("empty candidates: empty dict", best_e, {})
check("empty candidates: score=0",    score_e, 0.0)

# -----------------------------------------------------------------------
# Group 8 — align_themes
# -----------------------------------------------------------------------
print("\n[Group 8] align_themes")

aligned = align_themes(THEMES_A, THEMES_B, "Claude", "GPT")

check_true("returns DataFrame",   isinstance(aligned, pd.DataFrame))
check("rows = len(themes_a)",     len(aligned), len(THEMES_A))
check_true("Claude Theme column",
    "Claude Theme" in aligned.columns)
check_true("GPT Best Match column",
    "GPT Best Match" in aligned.columns)
check_true("Agreement Score column",
    "Agreement Score" in aligned.columns)
check_true("Interpretation column",
    "Interpretation" in aligned.columns)
check_true("all interpretations valid",
    aligned["Interpretation"].isin(
        ["strong","moderate","weak","poor"]
    ).all())
check_true("all scores in [0,1]",
    aligned["Agreement Score"].between(0, 1).all())

print(f"\n  Side-by-side alignment preview:")
for _, row in aligned.iterrows():
    print(f"  '{row['Claude Theme']}' ↔ '{row['GPT Best Match']}' "
          f"(score={row['Agreement Score']:.3f}, {row['Interpretation']})")

# Empty inputs
empty_aligned = align_themes([], THEMES_B)
check("empty themes_a: empty DataFrame", empty_aligned.empty, True)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — theme_comparator verified.")
else:
    print("  ❌ SOME TESTS FAILED — review above.")
print('='*60)
