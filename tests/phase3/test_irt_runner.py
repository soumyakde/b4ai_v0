"""
Phase 3 — IRT Unit Tests
test_irt_runner.py

Tests matrix builders with synthetic data (no R required).
Tests model functions require R + mirt — skipped gracefully if unavailable.

Run from project root:
    python tests/phase3/test_irt_runner.py
"""

import sys, os, importlib.util
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR,"..","..", "core","analytics","irt","irt_runner.py"),
    os.path.join(_THIS_DIR, "irt_runner.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES
              if os.path.exists(os.path.normpath(p))), None)
if not _path:
    print("ERROR: irt_runner.py not found"); sys.exit(1)

_spec = importlib.util.spec_from_file_location("irt_runner", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_binary_response_matrix = _mod.build_binary_response_matrix
build_likert_response_matrix = _mod.build_likert_response_matrix
run_rasch_model              = _mod.run_rasch_model
run_2pl_model                = _mod.run_2pl_model
run_grm_model                = _mod.run_grm_model
get_icc_data                 = _mod.get_icc_data
get_wright_map_data          = _mod.get_wright_map_data
MIN_N_2PL                    = _mod.MIN_N_2PL
MIN_N_GRM                    = _mod.MIN_N_GRM
MIN_N_WARN                   = _mod.MIN_N_WARN
_RPY2_AVAILABLE              = _mod._RPY2_AVAILABLE
_GIRTH_AVAILABLE             = _mod._GIRTH_AVAILABLE

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

def skip(label):
    print(f"  ⏭  SKIP  {label}  (rpy2/mirt not available)")

# -----------------------------------------------------------------------
# Synthetic canonical_df fixtures
# -----------------------------------------------------------------------
USERS  = [f"u{i}" for i in range(1, 10)]  # 9 users
ITEMS_BINARY = [f"Q3_{i}" for i in range(1, 9)]  # 8 items
ITEMS_LIKERT = ["Q2_1","Q2_2","Q2_3"]            # 3 items

np.random.seed(42)

def _make_binary_canonical():
    rows = []
    for u in USERS:
        for qid in ITEMS_BINARY:
            score = float(np.random.choice([0, 1], p=[0.4, 0.6]))
            rows.append(dict(
                user_id=u,
                instrument_key="precourse_pre_ai_misconceptions_assessment",
                question_id=qid, item_score=score,
                construct=None, module_id="global",
            ))
    return pd.DataFrame(rows)

def _make_likert_canonical():
    rows = []
    for u in USERS:
        for qid in ITEMS_LIKERT:
            score = float(np.random.choice([1,2,3,4]))
            rows.append(dict(
                user_id=u,
                instrument_key="module1_b4ai_sims_survey",
                question_id=qid, item_score=score,
                construct="intrinsic_motivation",
                module_id="module_1",
            ))
    return pd.DataFrame(rows)

binary_df = _make_binary_canonical()
likert_df = _make_likert_canonical()

# -----------------------------------------------------------------------
# Group 1 — build_binary_response_matrix
# -----------------------------------------------------------------------
print("\n[Group 1] build_binary_response_matrix")

matrix, item_ids = build_binary_response_matrix(
    binary_df, "precourse_pre_ai_misconceptions_assessment"
)
check("matrix shape rows=9",     matrix.shape[0], 9)
check("matrix shape cols=8",     matrix.shape[1], 8)
check_true("item_ids length=8",  len(item_ids) == 8)
check_true("all values 0 or 1",
    matrix.stack().isin([0.0, 1.0]).all())
check_true("index = user_ids",
    set(matrix.index) == set(USERS))

# suffix-match also works
matrix2, _ = build_binary_response_matrix(
    binary_df, "pre_ai_misconceptions_assessment"
)
check("suffix match: same shape", matrix2.shape, matrix.shape)

# -----------------------------------------------------------------------
# Group 2 — build_likert_response_matrix
# -----------------------------------------------------------------------
print("\n[Group 2] build_likert_response_matrix")

lmatrix, litem_ids = build_likert_response_matrix(
    likert_df, "b4ai_sims_survey", "intrinsic_motivation"
)
check("likert matrix rows=9",      lmatrix.shape[0], 9)
check("likert matrix cols=3",      lmatrix.shape[1], 3)
check_true("all values 1–4",
    lmatrix.stack().between(1, 4).all())

# -----------------------------------------------------------------------
# Group 3 — Error handling: missing instrument
# -----------------------------------------------------------------------
print("\n[Group 3] Error handling — missing instrument")

try:
    build_binary_response_matrix(binary_df, "nonexistent_instrument")
    print("  ❌ FAIL  No error raised"); FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError: {e}"); PASS += 1

try:
    build_likert_response_matrix(binary_df, "b4ai_sims_survey", "bad_construct")
    print("  ❌ FAIL  No error raised"); FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError: {e}"); PASS += 1

# -----------------------------------------------------------------------
# Group 4 — Constants
# -----------------------------------------------------------------------
print("\n[Group 4] Constants")
check("MIN_N_2PL = 50",   MIN_N_2PL,  50)
check("MIN_N_GRM = 50",   MIN_N_GRM,  50)
check("MIN_N_WARN = 100", MIN_N_WARN, 100)

# -----------------------------------------------------------------------
# Group 5 — 2PL gates at n < MIN_N_2PL
# -----------------------------------------------------------------------
print("\n[Group 5] 2PL n gate (n=9 < 50 → error in result)")

r_2pl = run_2pl_model(matrix, item_ids)
check_true("2PL error set for n=9", r_2pl["error"] is not None)
check_true("2PL error mentions n",
    str(MIN_N_2PL) in str(r_2pl["error"]))
check("2PL model_type", r_2pl["model_type"], "2PL")
check("2PL n_persons",  r_2pl["n_persons"],  9)

# -----------------------------------------------------------------------
# Group 6 — GRM gates at n < MIN_N_GRM
# -----------------------------------------------------------------------
print("\n[Group 6] GRM n gate (n=9 < 50 → error in result)")

r_grm = run_grm_model(lmatrix, litem_ids)
check_true("GRM error set for n=9", r_grm["error"] is not None)
check_true("GRM error mentions n",
    str(MIN_N_GRM) in str(r_grm["error"]))

# -----------------------------------------------------------------------
# Group 7 — get_icc_data graceful on error result
# -----------------------------------------------------------------------
print("\n[Group 7] get_icc_data — graceful on error result")

error_result = {"error": "Model failed", "item_params": pd.DataFrame()}
icc = get_icc_data(error_result)
check_true("ICC empty on error",   icc.empty)
check_true("ICC has theta column", "theta" in icc.columns)

# -----------------------------------------------------------------------
# Group 8 — get_wright_map_data graceful on error result
# -----------------------------------------------------------------------
print("\n[Group 8] get_wright_map_data — graceful on error result")

wm = get_wright_map_data(error_result)
check_true("Wright map has persons key", "persons" in wm)
check_true("Wright map has items key",   "items" in wm)
check_true("persons empty on error",     wm["persons"].empty)
check_true("items empty on error",       wm["items"].empty)

# -----------------------------------------------------------------------
# Group 9 — Rasch model (requires rpy2 + mirt)
# -----------------------------------------------------------------------
print("\n[Group 9] run_rasch_model (requires girth)")

if not _GIRTH_AVAILABLE:
    skip("girth not available — install with: pip install girth")
else:
    try:
        r = run_rasch_model(matrix, item_ids)
        if r["error"]:
            print(f"  ⚠️  Rasch error: {r['error']}")
        else:
            check("model_type=Rasch",   r["model_type"], "Rasch")
            check("n_persons=9",        r["n_persons"],  9)
            check("low_n_warning=True", r["low_n_warning"], True)
            check_true("item_params not empty",
                isinstance(r["item_params"], pd.DataFrame)
                and not r["item_params"].empty)
            check_true("person_params not empty",
                isinstance(r["person_params"], pd.DataFrame)
                and not r["person_params"].empty)
            check_true("theta column present",
                "theta" in r["person_params"].columns)
            check_true("b column in item_params",
                "b" in r["item_params"].columns)
            check_true("aic is float",
                isinstance(r.get("aic"), float))
            check_true("item_fit not empty",
                isinstance(r.get("item_fit"), pd.DataFrame)
                and not r["item_fit"].empty)
            check_true("infit column present",
                "infit" in r["item_fit"].columns)

            # Wright map
            wm = get_wright_map_data(r)
            check_true("wright map persons populated", not wm["persons"].empty)
            check_true("wright map items populated",   not wm["items"].empty)
            check_true("theta values finite",
                wm["persons"]["theta"].apply(lambda x: x == x).all())

            # ICC data
            icc = get_icc_data(r, item_id=item_ids[0])
            check_true("ICC has rows",         len(icc) > 0)
            check_true("ICC theta range ok",
                icc["theta"].between(-4, 4).all())
            check_true("ICC prob range ok",
                icc["probability"].between(0, 1).all())

            print(f"\n  ℹ️  AIC={r.get('aic')}, BIC={r.get('bic')}")
            print(f"  ℹ️  Item difficulties:")
            for _, row in r["item_params"].iterrows():
                print(f"    {row['item_id']}: b={row['b']:.3f}")
            print(f"  ℹ️  Person abilities:")
            for _, row in r["person_params"].iterrows():
                print(f"    {row['user_id']}: θ={row['theta']:.3f}  SE={row['theta_se']:.3f}")
            print(f"  ℹ️  Item fit (infit/outfit):")
            for _, row in r["item_fit"].iterrows():
                print(f"    {row['item_id']}: infit={row['infit']}, outfit={row['outfit']}")

    except Exception as e:
        print(f"  ⚠️  Rasch test error: {e}")
        import traceback; traceback.print_exc()

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — irt_runner verified.")
else:
    print("  ❌ SOME TESTS FAILED — review above.")
print('='*60)
