"""
Phase 0 — Scoring Verification Test 1
test_binary_misconceptions.py

Tests: pre_ai_misconceptions_assessment + post_ai_misconceptions_assessment
Scoring type: binary (correct_answers)
8 items each. Pre and post share identical correct answers.

Run from project root:
    python tests/phase0/test_binary_misconceptions.py

Expected result: ALL TESTS PASSED
"""

import sys
import os

# Allow running from project root OR from this file's directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.scoring_engine import compute_score

# -----------------------------------------------------------------------
# Scoring YAMLs (embedded — mirrors file content exactly)
# -----------------------------------------------------------------------

PRE_SCORING = {
    "scoring_type": "binary",
    "correct_answers": {
        "Q3_1": "False",
        "Q3_2": "False",
        "Q3_3": "False",
        "Q3_4": "False",
        "Q3_5": "No",
        "Q3_6": "Yes",
        "Q3_7": "No",
        "Q3_8": "No",
    }
}

POST_SCORING = {
    "scoring_type": "binary",
    "correct_answers": {
        "Q3_1": "False",
        "Q3_2": "False",
        "Q3_3": "False",
        "Q3_4": "False",
        "Q3_5": "No",
        "Q3_6": "Yes",
        "Q3_7": "No",
        "Q3_8": "No",
    }
}

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------

PASS = 0
FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  ✅ PASS  {label}  →  score={got}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  →  got={got}, expected={expected}")
        FAIL += 1

# -----------------------------------------------------------------------
# Test 1a — All correct (score = 8)
# -----------------------------------------------------------------------
print("\n[Test 1a] All 8 answers correct — expected score: 8")
all_correct = {
    "Q3_1": "False",
    "Q3_2": "False",
    "Q3_3": "False",
    "Q3_4": "False",
    "Q3_5": "No",
    "Q3_6": "Yes",
    "Q3_7": "No",
    "Q3_8": "No",
}
check("pre  all correct", compute_score(all_correct, PRE_SCORING), 8)
check("post all correct", compute_score(all_correct, POST_SCORING), 8)

# -----------------------------------------------------------------------
# Test 1b — All wrong (score = 0)
# -----------------------------------------------------------------------
print("\n[Test 1b] All 8 answers wrong — expected score: 0")
all_wrong = {
    "Q3_1": "True",
    "Q3_2": "True",
    "Q3_3": "True",
    "Q3_4": "True",
    "Q3_5": "Yes",
    "Q3_6": "No",
    "Q3_7": "Yes",
    "Q3_8": "Yes",
}
check("pre  all wrong", compute_score(all_wrong, PRE_SCORING), 0)
check("post all wrong", compute_score(all_wrong, POST_SCORING), 0)

# -----------------------------------------------------------------------
# Test 1c — Mixed: first 5 correct, last 3 wrong (score = 5)
# -----------------------------------------------------------------------
print("\n[Test 1c] First 5 correct, last 3 wrong — expected score: 5")
mixed = {
    "Q3_1": "False",   # correct
    "Q3_2": "False",   # correct
    "Q3_3": "False",   # correct
    "Q3_4": "False",   # correct
    "Q3_5": "No",      # correct
    "Q3_6": "No",      # wrong  (correct is Yes)
    "Q3_7": "Yes",     # wrong  (correct is No)
    "Q3_8": "Yes",     # wrong  (correct is No)
}
check("pre  mixed", compute_score(mixed, PRE_SCORING), 5)
check("post mixed", compute_score(mixed, POST_SCORING), 5)

# -----------------------------------------------------------------------
# Test 1d — Partial submission: only Q3_1 and Q3_6 answered
#           (subset scoring — missing questions score 0)
#           Q3_1 correct + Q3_6 correct = score 2
# -----------------------------------------------------------------------
print("\n[Test 1d] Partial submission (2 of 8 questions) — expected score: 2")
partial = {
    "Q3_1": "False",   # correct
    "Q3_6": "Yes",     # correct
}
check("pre  partial", compute_score(partial, PRE_SCORING), 2)

# -----------------------------------------------------------------------
# Test 1e — Single incorrect answer (score = 0)
# -----------------------------------------------------------------------
print("\n[Test 1e] Single wrong answer — expected score: 0")
single_wrong = {"Q3_1": "True"}
check("pre  single wrong", compute_score(single_wrong, PRE_SCORING), 0)

# -----------------------------------------------------------------------
# Test 1f — Unknown question ID must raise ValueError
# -----------------------------------------------------------------------
print("\n[Test 1f] Unknown question ID — expected: ValueError raised")
try:
    compute_score({"Q3_INVALID": "False"}, PRE_SCORING)
    print("  ❌ FAIL  No error raised for unknown question ID")
    FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError raised correctly: {e}")
    PASS += 1

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — Binary misconceptions scoring verified.")
else:
    print("  ❌ SOME TESTS FAILED — Review output above.")
print('='*55)
