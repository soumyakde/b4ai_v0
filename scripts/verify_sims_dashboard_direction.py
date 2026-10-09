"""
verify_sims_dashboard_direction.py -- checks that the teacher dashboard's SIMS interpretation matches the scoring.

Scoring (b4ai_sims_scoring.yaml): no SIMS item is reverse-scored, so for all four constructs a higher mean = MORE of that
regulation. External Regulation and Amotivation are therefore "high = worse" and must carry reverse_coded=True in the
dashboard's _CONSTRUCT_DEFINITIONS; no SIMS item may be flagged item-reverse-coded; and the interpretation logic must show a
HIGH External Regulation / Amotivation mean as RED. Run:  python scripts/verify_sims_dashboard_direction.py   (exit 0 = all pass)
"""
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml

import streamlit_app.dashboards.teacher_dashboard as td

fail = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        fail.append(name)


sc = yaml.safe_load(open(os.path.join(ROOT, "streamlit_app", "surveys", "b4ai_sims_scoring.yaml"), encoding="utf-8"))
D = td._CONSTRUCT_DEFINITIONS

check("scoring file reverses no SIMS item", sc["reverse_questions"] == [], str(sc["reverse_questions"]))
for c in ("external_regulation", "amotivation"):
    check(f"{c}: reverse_coded=True (high = worse)", D[c]["reverse_coded"] is True)
    check(f"{c}: has a reverse_note and no stale item_reverse_note", bool(D[c].get("reverse_note")) and "item_reverse_note" not in D[c])
    check(f"{c}: scale_high describes the BAD end", not re.search(r"\bNOT\b", D[c]["scale_high"]), D[c]["scale_high"][:60])
    check(f"{c}: scale_low describes the GOOD end", bool(re.search(r"\bNOT\b", D[c]["scale_low"])), D[c]["scale_low"][:60])
for c in ("intrinsic_motivation", "identified_regulation"):
    check(f"{c}: reverse_coded=False (high = better)", D[c]["reverse_coded"] is False)
check("no SIMS item flagged item-reverse-coded", td._SIMS_REVERSE_ITEMS == set(), str(td._SIMS_REVERSE_ITEMS))
check("SIMS item-reverse set matches the scoring file", td._SIMS_REVERSE_ITEMS == set(sc["reverse_questions"]))

# Same branch logic as the 'How to interpret these scores' panel (copied behaviour: reverse_coded -> high = red)
src = open(os.path.join(ROOT, "streamlit_app", "dashboards", "teacher_dashboard.py"), encoding="utf-8").read()
check("interpretation panel: reverse_coded branch shows High as red",
      'if cdef.get("reverse_coded"):\n                        if cmean_val >= 3.0:\n                            interp = f"🔴 High' in src)

print("\n" + ("ALL CHECKS PASSED" if not fail else "FAILED: " + ", ".join(fail)))
sys.exit(1 if fail else 0)
