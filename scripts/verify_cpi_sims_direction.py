"""
verify_cpi_sims_direction.py -- direction check for SIMS-based quantities (CPI_process and RAI).

DEFECT BEING TESTED (found 2026-10-08, NOT yet fixed): the SIMS scoring file (b4ai_sims_scoring.yaml) reverse-scores
every External_regulation (Q4_1-Q4_4) and Amotivation (Q5_1-Q5_3) item, so the stored item_score for those
constructs already runs "high = LESS controlled / LESS amotivated". Two later steps then correct for direction AGAIN:
  * cpi_engine.compute_cpi_process flips external_regulation and amotivation with (5 - mean)
  * correlation_engine.compute_rai subtracts external_regulation and amotivation (formula assumes RAW direction)
Result: a clearly motivated student and a clearly amotivated student get the SAME CPI_process SIMS value (0.50) and
the SAME RAI (0.0).

This script builds three synthetic students from RAW answers through the real DatasetBuilder / scoring files and
checks that the outputs have the correct ordering. Until the defect is fixed it is EXPECTED TO FAIL.

    motivated  : Strongly agree to intrinsic + identified reasons, Strongly disagree to external + amotivation reasons
    controlled : Disagree intrinsic/identified/amotivation, Strongly agree to external reasons
    amotivated : Strongly disagree intrinsic + identified, Strongly agree external + amotivation
Correct behaviour: RAI(motivated) > RAI(controlled) > RAI(amotivated), with RAI(motivated) > 0 > RAI(amotivated);
CPI_process SIMS value ordered the same way.

Usage (project root, env b4ai_v0):  python scripts/verify_cpi_sims_direction.py
Exit code 0 = all pass (defect fixed), 1 = at least one failure.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import yaml

import core.analytics.datasets.canonical_loader as cl
from core.analytics.correlational.correlation_engine import compute_rai
from core.analytics.cpi.cpi_engine import compute_cpi_process
from core.analytics.datasets.dataset_builder import DatasetBuilder
from core.analytics.descriptive.score_aggregator import compute_construct_means

fail = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        fail.append(name)


inst, sc = cl._build_instruments_dict(), cl._build_scoring_dict()
KEY = "module1_b4ai_sims_survey"
y = yaml.safe_load(open(os.path.join(ROOT, "streamlit_app", "surveys", "b4ai_sims_survey.yaml"), encoding="utf-8"))
qs = {s["name"].lower(): [q["id"] for q in s["questions"]] for s in y["sections"]}
SD, D, A, SA = "Strongly disagree", "Disagree", "Agree", "Strongly agree"


def student(uid, intr, ident, ext, amo):
    rows = []
    for cons, ans in (("intrinsic_motivation", intr), ("identified_regulation", ident),
                      ("external_regulation", ext), ("amotivation", amo)):
        for q in qs[cons]:
            rows.append(dict(user_id=uid, instrument_name=KEY, question_id=q, response_value=ans, submitted_at="2026-08-01"))
    return rows


raw = pd.DataFrame(student("motivated", SA, SA, SD, SD) + student("controlled", D, D, SA, D) + student("amotivated", SD, SD, SA, SA))
canon = DatasetBuilder(raw, inst, sc).build()
canon["cohort_id"] = None

# Documented fact (informational): how the scoring file stores the two constructs
cm = compute_construct_means(canon).pivot_table(index="user_id", columns="construct", values="mean_score")
print("Construct means as stored (1-4). Since the 2026-10-08 fix every SIMS construct reads 'higher = MORE'")
print("(before the fix, Strongly agree to an amotivation item was stored as 1, i.e. reversed):")
print(cm.round(2).to_string(), "\n")

rai = compute_rai(canon, "b4ai_sims_survey").groupby("user_id").rai.mean()
cpi = compute_cpi_process(canon).set_index("user_id").sims_mean.astype(float)
print("RAI as computed:        ", rai.round(2).to_dict())
print("CPI_process SIMS value:  ", cpi.round(3).to_dict(), "\n")

check("RAI: motivated student is POSITIVE", rai["motivated"] > 0, f"{rai['motivated']:.2f}")
check("RAI: amotivated student is NEGATIVE", rai["amotivated"] < 0, f"{rai['amotivated']:.2f}")
check("RAI ordering motivated > controlled > amotivated", rai["motivated"] > rai["controlled"] > rai["amotivated"],
      f"{rai['motivated']:.2f} / {rai['controlled']:.2f} / {rai['amotivated']:.2f}")
check("CPI_process SIMS value: motivated > amotivated (must not be equal)", cpi["motivated"] > cpi["amotivated"],
      f"{cpi['motivated']:.3f} vs {cpi['amotivated']:.3f}")
check("CPI_process SIMS ordering motivated > controlled > amotivated", cpi["motivated"] > cpi["controlled"] > cpi["amotivated"],
      f"{cpi['motivated']:.3f} / {cpi['controlled']:.3f} / {cpi['amotivated']:.3f}")

print("\n" + ("ALL CHECKS PASSED" if not fail else "FAILED: " + ", ".join(fail)))
sys.exit(1 if fail else 0)
