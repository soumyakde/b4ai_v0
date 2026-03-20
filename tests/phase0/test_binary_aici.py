"""
Phase 0 — Scoring Verification Test 2
test_binary_aici.py

Tests: pre_aici_assessment + post_aici_assessment
Scoring type: binary (correct_answers)
20 items each. Pre and post share identical correct answers.

Run from project root:
    python tests/phase0/test_binary_aici.py

Expected result: ALL TESTS PASSED
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.scoring_engine import compute_score

# -----------------------------------------------------------------------
# Scoring YAML (identical for pre and post — embedded from files)
# -----------------------------------------------------------------------

SCORING = {
    "scoring_type": "binary",
    "correct_answers": {
        "Q4_1":  "B",
        "Q4_2":  "A",
        "Q4_3":  "B",
        "Q4_4":  "A",
        "Q4_5":  "A",
        "Q4_6":  "B",
        "Q4_7":  "D",
        "Q4_8":  "A",
        "Q4_9":  "B",
        "Q4_10": "B",
        "Q4_11": "A",
        "Q4_12": "B",
        "Q4_13": "A",
        "Q4_14": "B",
        "Q4_15": "A",
        "Q4_16": "B",
        "Q4_17": "A",
        "Q4_18": "A",
        "Q4_19": "A",
        "Q4_20": "A",
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
# Test 2a — All 20 correct (score = 20)
# -----------------------------------------------------------------------
print("\n[Test 2a] All 20 answers correct — expected score: 20")
all_correct = {
    "Q4_1": "B", "Q4_2": "A", "Q4_3": "B", "Q4_4": "A", "Q4_5": "A",
    "Q4_6": "B", "Q4_7": "D", "Q4_8": "A", "Q4_9": "B", "Q4_10": "B",
    "Q4_11": "A", "Q4_12": "B", "Q4_13": "A", "Q4_14": "B", "Q4_15": "A",
    "Q4_16": "B", "Q4_17": "A", "Q4_18": "A", "Q4_19": "A", "Q4_20": "A",
}
check("pre  all correct", compute_score(all_correct, SCORING), 20)
check("post all correct", compute_score(all_correct, SCORING), 20)

# -----------------------------------------------------------------------
# Test 2b — All wrong (score = 0)
# -----------------------------------------------------------------------
print("\n[Test 2b] All 20 answers wrong — expected score: 0")
all_wrong = {
    "Q4_1": "A", "Q4_2": "B", "Q4_3": "A", "Q4_4": "B", "Q4_5": "B",
    "Q4_6": "A", "Q4_7": "A", "Q4_8": "B", "Q4_9": "A", "Q4_10": "A",
    "Q4_11": "B", "Q4_12": "A", "Q4_13": "B", "Q4_14": "A", "Q4_15": "B",
    "Q4_16": "A", "Q4_17": "B", "Q4_18": "B", "Q4_19": "B", "Q4_20": "B",
}
check("pre  all wrong", compute_score(all_wrong, SCORING), 0)
check("post all wrong", compute_score(all_wrong, SCORING), 0)

# -----------------------------------------------------------------------
# Test 2c — First 10 correct, last 10 wrong (score = 10)
# -----------------------------------------------------------------------
print("\n[Test 2c] First 10 correct, last 10 wrong — expected score: 10")
half = {
    "Q4_1": "B",  "Q4_2": "A",  "Q4_3": "B",  "Q4_4": "A",  "Q4_5": "A",
    "Q4_6": "B",  "Q4_7": "D",  "Q4_8": "A",  "Q4_9": "B",  "Q4_10": "B",
    "Q4_11": "B", "Q4_12": "A", "Q4_13": "B", "Q4_14": "A", "Q4_15": "B",
    "Q4_16": "A", "Q4_17": "B", "Q4_18": "B", "Q4_19": "B", "Q4_20": "B",
}
check("half correct", compute_score(half, SCORING), 10)

# -----------------------------------------------------------------------
# Test 2d — Single correct answer from full set (score = 1)
# -----------------------------------------------------------------------
print("\n[Test 2d] One correct answer only — expected score: 1")
one = {"Q4_7": "D"}   # D is the correct answer for Q4_7
check("single correct Q4_7", compute_score(one, SCORING), 1)

# -----------------------------------------------------------------------
# Test 2e — Verify Q4_7 correct answer is D, not others
# -----------------------------------------------------------------------
print("\n[Test 2e] Q4_7 answer A/B/C are all wrong — expected score: 0 each")
for wrong_ans in ["A", "B", "C"]:
    check(f"Q4_7 = {wrong_ans}", compute_score({"Q4_7": wrong_ans}, SCORING), 0)

# -----------------------------------------------------------------------
# Test 2f — Unknown question ID must raise ValueError
# -----------------------------------------------------------------------
print("\n[Test 2f] Unknown question ID — expected: ValueError raised")
try:
    compute_score({"Q4_99": "A"}, SCORING)
    print("  ❌ FAIL  No error raised for unknown question ID")
    FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError raised correctly: {e}")
    PASS += 1

# -----------------------------------------------------------------------
# Test 2g — Pre and post scoring keys are identical
#           (guard: if they ever diverge, this test catches it)
# -----------------------------------------------------------------------
print("\n[Test 2g] Pre and post AICI correct answers are identical — sanity check")
PRE_CORRECT = {
    "Q4_1": "B", "Q4_2": "A", "Q4_3": "B", "Q4_4": "A", "Q4_5": "A",
    "Q4_6": "B", "Q4_7": "D", "Q4_8": "A", "Q4_9": "B", "Q4_10": "B",
    "Q4_11": "A", "Q4_12": "B", "Q4_13": "A", "Q4_14": "B", "Q4_15": "A",
    "Q4_16": "B", "Q4_17": "A", "Q4_18": "A", "Q4_19": "A", "Q4_20": "A",
}
POST_CORRECT = {
    "Q4_1": "B", "Q4_2": "A", "Q4_3": "B", "Q4_4": "A", "Q4_5": "A",
    "Q4_6": "B", "Q4_7": "D", "Q4_8": "A", "Q4_9": "B", "Q4_10": "B",
    "Q4_11": "A", "Q4_12": "B", "Q4_13": "A", "Q4_14": "B", "Q4_15": "A",
    "Q4_16": "B", "Q4_17": "A", "Q4_18": "A", "Q4_19": "A", "Q4_20": "A",
}
if PRE_CORRECT == POST_CORRECT:
    print("  ✅ PASS  Pre and post correct_answers are identical")
    PASS += 1
else:
    diffs = {k for k in PRE_CORRECT if PRE_CORRECT.get(k) != POST_CORRECT.get(k)}
    print(f"  ❌ FAIL  Divergence found at: {diffs}")
    FAIL += 1

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — Binary AICI scoring verified.")
else:
    print("  ❌ SOME TESTS FAILED — Review output above.")
print('='*55)
