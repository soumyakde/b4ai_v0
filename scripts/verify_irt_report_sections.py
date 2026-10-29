"""Guard for the Reports tab "iii. IRT Analysis Report" (defect D-1).
Builds the report sections for every Binary-Assessments model and fails if none is produced.
Run: set PYTHONUTF8=1, then python scripts/verify_irt_report_sections.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np, pandas as pd
import streamlit_app.dashboards.teacher_dashboard as td

fail = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok: fail.append(name)

# Synthetic pre-course misconceptions data: 60 students x 8 items, scored 0/1 (Rasch-like)
rng = np.random.default_rng(7)
theta = rng.normal(size=60); b = np.linspace(-1.5, 1.5, 8)
rows = []
for u, th in enumerate(theta):
    for j, bj in enumerate(b):
        p = 1 / (1 + np.exp(-(th - bj)))
        rows.append({"user_id": f"u{u}", "instrument_key": "precourse_pre_ai_misconceptions_assessment",
                     "question_id": f"Q{j+1}", "item_score": float(rng.random() < p)})
df = pd.DataFrame(rows)

for model in ["Classical item analysis (CTT)", "Random-effects logistic (Python engine)", "Rasch (1PL)"]:
    sections, failed = td._irt_report_sections(df, "Binary Assessments", irt_model=model)
    check(f"{model}: at least one report section is produced", len(sections) > 1, f"{len(sections)-1} sections")
    check(f"{model}: the one instrument with data is not listed as failed", not any("misconceptions" in str(f[0]).lower() and "pre" in str(f[0]).lower() for f in failed), str(failed))

# Instruments without data are skipped quietly (no rows -> not a failure worth listing)
# An instrument that has data but cannot be analysed MUST be listed
bad = df.copy(); bad["item_score"] = 1.0
sections, failed = td._irt_report_sections(bad, "Binary Assessments", irt_model="Random-effects logistic (Python engine)")
check("an instrument that cannot be analysed is listed in 'failed' (never silently dropped)", len(failed) >= 1, str(failed))

print("\n%d check(s) failed" % len(fail) if fail else "\nAll checks passed")
sys.exit(1 if fail else 0)
