"""
Phase 0 — Step 5 Verification
test_instrument_key_resolver.py

Tests: core/analytics/filters/instrument_key_resolver.py

Run from project root:
    python tests/phase0/test_instrument_key_resolver.py

Expected result: ALL TESTS PASSED
"""

import sys
import os
import importlib.util

# -----------------------------------------------------------------------
# Import resolver directly by file path — works from any working directory
# -----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Try project path first, then sibling directory (local test run)
_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "core", "analytics", "filters", "instrument_key_resolver.py"),
    os.path.join(_THIS_DIR, "instrument_key_resolver.py"),
]

_resolver_path = None
for _p in _CANDIDATES:
    if os.path.exists(os.path.normpath(_p)):
        _resolver_path = os.path.normpath(_p)
        break

if _resolver_path is None:
    print("ERROR: instrument_key_resolver.py not found.")
    print("Expected at: core/analytics/filters/instrument_key_resolver.py")
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("instrument_key_resolver", _resolver_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

resolve_instrument_key       = _mod.resolve_instrument_key
resolve_instrument_keys_series = _mod.resolve_instrument_keys_series
get_survey_base_keys         = _mod.get_survey_base_keys

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------

PASS = 0
FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  ✅ PASS  {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}")
        print(f"           got:      {got!r}")
        print(f"           expected: {expected!r}")
        FAIL += 1

# -----------------------------------------------------------------------
# Test group 1 — Survey prefix stripping (Rule 1)
# -----------------------------------------------------------------------
print("\n[Test Group 1] Survey prefix stripping — module{N}_b4ai_*")

check("module1_b4ai_sccces_survey",
      resolve_instrument_key("module1_b4ai_sccces_survey"),
      "b4ai_sccces_survey")

check("module2_b4ai_sccces_survey",
      resolve_instrument_key("module2_b4ai_sccces_survey"),
      "b4ai_sccces_survey")

check("module7_b4ai_sccces_survey",
      resolve_instrument_key("module7_b4ai_sccces_survey"),
      "b4ai_sccces_survey")

check("module1_b4ai_sims_survey",
      resolve_instrument_key("module1_b4ai_sims_survey"),
      "b4ai_sims_survey")

check("module5_b4ai_sims_survey",
      resolve_instrument_key("module5_b4ai_sims_survey"),
      "b4ai_sims_survey")

# Base key alone (no prefix) also resolves correctly
check("b4ai_sccces_survey (no prefix)",
      resolve_instrument_key("b4ai_sccces_survey"),
      "b4ai_sccces_survey")

check("b4ai_sims_survey (no prefix)",
      resolve_instrument_key("b4ai_sims_survey"),
      "b4ai_sims_survey")

# -----------------------------------------------------------------------
# Test group 2 — Demographics prefix stripping (Rule 2)
# -----------------------------------------------------------------------
print("\n[Test Group 2] Demographics prefix stripping")

check("precourse_demographics_survey",
      resolve_instrument_key("precourse_demographics_survey"),
      "demographics_survey")

check("demographics_survey (no prefix)",
      resolve_instrument_key("demographics_survey"),
      "demographics_survey")

# Hypothetical other prefix — still strips correctly
check("postcourse_demographics_survey",
      resolve_instrument_key("postcourse_demographics_survey"),
      "demographics_survey")

# -----------------------------------------------------------------------
# Test group 3 — Identity pass-through (Rule 3)
# -----------------------------------------------------------------------
print("\n[Test Group 3] Identity pass-through — assessments and MCQs")

check("pre_ai_misconceptions_assessment",
      resolve_instrument_key("pre_ai_misconceptions_assessment"),
      "pre_ai_misconceptions_assessment")

check("post_ai_misconceptions_assessment",
      resolve_instrument_key("post_ai_misconceptions_assessment"),
      "post_ai_misconceptions_assessment")

check("pre_aici_assessment",
      resolve_instrument_key("pre_aici_assessment"),
      "pre_aici_assessment")

check("post_aici_assessment",
      resolve_instrument_key("post_aici_assessment"),
      "post_aici_assessment")

check("module1_content_mcq_assessment",
      resolve_instrument_key("module1_content_mcq_assessment"),
      "module1_content_mcq_assessment")

check("module2_content_mcq_assessment",
      resolve_instrument_key("module2_content_mcq_assessment"),
      "module2_content_mcq_assessment")

check("module7_content_mcq_assessment",
      resolve_instrument_key("module7_content_mcq_assessment"),
      "module7_content_mcq_assessment")

# -----------------------------------------------------------------------
# Test group 4 — Whitespace robustness
# -----------------------------------------------------------------------
print("\n[Test Group 4] Whitespace handling")

check("leading/trailing spaces stripped",
      resolve_instrument_key("  module1_b4ai_sccces_survey  "),
      "b4ai_sccces_survey")

check("leading/trailing spaces on assessment",
      resolve_instrument_key("  pre_aici_assessment  "),
      "pre_aici_assessment")

# -----------------------------------------------------------------------
# Test group 5 — Vectorized Series resolver
# -----------------------------------------------------------------------
print("\n[Test Group 5] Vectorized resolve_instrument_keys_series()")

try:
    import pandas as pd

    series = pd.Series([
        "module1_b4ai_sccces_survey",
        "module2_b4ai_sims_survey",
        "precourse_demographics_survey",
        "pre_ai_misconceptions_assessment",
        "post_aici_assessment",
        "module3_content_mcq_assessment",
    ])

    expected = pd.Series([
        "b4ai_sccces_survey",
        "b4ai_sims_survey",
        "demographics_survey",
        "pre_ai_misconceptions_assessment",
        "post_aici_assessment",
        "module3_content_mcq_assessment",
    ])

    result = resolve_instrument_keys_series(series)

    if result.equals(expected):
        print("  ✅ PASS  vectorized resolver — all 6 values correct")
        PASS += 1
    else:
        print("  ❌ FAIL  vectorized resolver — mismatch:")
        for i, (got, exp) in enumerate(zip(result, expected)):
            status = "✅" if got == exp else "❌"
            print(f"    [{i}] {status}  got={got!r}  expected={exp!r}")
        FAIL += 1

except ImportError:
    print("  ⚠️  SKIP  pandas not available — skipping Series test")

# -----------------------------------------------------------------------
# Test group 6 — get_survey_base_keys()
# -----------------------------------------------------------------------
print("\n[Test Group 6] get_survey_base_keys() contract")

base_keys = get_survey_base_keys()
check("returns frozenset",         isinstance(base_keys, frozenset), True)
check("b4ai_sccces_survey in set", "b4ai_sccces_survey" in base_keys, True)
check("b4ai_sims_survey in set",   "b4ai_sims_survey" in base_keys,   True)

# -----------------------------------------------------------------------
# Test group 7 — No false positive stripping
#               e.g. "module1_content_mcq_assessment" must NOT lose its
#               module prefix, because it IS the canonical key
# -----------------------------------------------------------------------
print("\n[Test Group 7] No false positive stripping on MCQ keys")

check("module1_content_mcq_assessment not stripped",
      resolve_instrument_key("module1_content_mcq_assessment"),
      "module1_content_mcq_assessment")

check("module1_content_mcq_assessment != b4ai_sccces_survey",
      resolve_instrument_key("module1_content_mcq_assessment") == "b4ai_sccces_survey",
      False)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — instrument_key_resolver verified.")
else:
    print("  ❌ SOME TESTS FAILED — review output above.")
print('='*60)
