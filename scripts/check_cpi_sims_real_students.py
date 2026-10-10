"""
check_cpi_sims_real_students.py -- READ-ONLY real-data direction check of the SIMS part of CPI_process.

The dashboard has no screen that shows CPI_process or its SIMS component (the Competency Progression tab is CPI_quant + CPI_qual;
the three-component CPI+ in cpi_engine.compute_cpi_plus is only a fallback inside the PDF report, which prints quant / qual / plus).
This script therefore calls the real engine (cpi_engine.compute_cpi_process) on a temporary copy of responses.db and checks that the
SIMS component (0 to 1, higher = better) moves the right way against each student's own SIMS construct means (1 to 4, all read
"higher = MORE" since the 2026-10-08 fix):

    SIMS component  should correlate  POSITIVELY with Intrinsic and Identified,  NEGATIVELY with External Regulation and Amotivation.

It also prints the three highest and three lowest students so you can read the raw means yourself.   Nothing is written.
Usage (project root, env b4ai_v0):  python scripts/check_cpi_sims_real_students.py        Exit code 0 = all checks pass.
"""
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
from scipy import stats

import core.analytics.datasets.canonical_loader as cl
from core.analytics.cpi.cpi_engine import compute_cpi_process
from core.analytics.descriptive.score_aggregator import compute_construct_means

tmp = os.path.join(tempfile.mkdtemp(), "r.db")
shutil.copyfile(os.path.join(ROOT, "responses.db"), tmp)
df, _, _ = cl.load_canonical_data(tmp)

proc = compute_cpi_process(df).dropna(subset=["sims_mean"]).set_index("user_id")
cm = compute_construct_means(df)
cm = cm[cm.instrument_key.str.lower().str.contains("sims")]
raw = cm.groupby(["user_id", "construct"]).mean_score.mean().unstack("construct")   # student mean across modules, 1-4
d = proc[["sims_mean"]].join(raw, how="inner")
cols = ["intrinsic_motivation", "identified_regulation", "external_regulation", "amotivation"]
d = d[["sims_mean"] + cols].rename(columns={"sims_mean": "SIMS_component(0-1)", "intrinsic_motivation": "Intrinsic",
                                            "identified_regulation": "Identified", "external_regulation": "External",
                                            "amotivation": "Amotivation"})
print(f"Students with a SIMS component: {len(d)}\n")
print("Three HIGHEST SIMS components (should have high Intrinsic/Identified, low External/Amotivation):")
print(d.sort_values("SIMS_component(0-1)", ascending=False).head(3).round(2).to_string())
print("\nThree LOWEST SIMS components (should be the opposite):")
print(d.sort_values("SIMS_component(0-1)").head(3).round(2).to_string())

print("\nSpearman correlation of the SIMS component with each construct mean (student level):")
fail = []
for col, sign in (("Intrinsic", +1), ("Identified", +1), ("External", -1), ("Amotivation", -1)):
    rho, p = stats.spearmanr(d["SIMS_component(0-1)"], d[col])
    ok = (rho * sign) > 0.3
    print(f"   {col:12s} rho = {rho:+.2f}  expected {'positive' if sign > 0 else 'negative'}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fail.append(col)
print("\n" + ("ALL CHECKS PASSED" if not fail and len(d) >= 50 else "FAILED: " + ", ".join(fail) + ("" if len(d) >= 50 else " (fewer than 50 students)")))
sys.exit(1 if fail or len(d) < 50 else 0)
