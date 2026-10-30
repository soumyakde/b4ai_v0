"""Guards for display fixes D-5 (tied ranks), D-6 (long review flags stay readable) and D-7 (2PL warnings shown).
Run: set PYTHONUTF8=1, then python scripts/verify_display_fixes.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np, pandas as pd
from streamlit.testing.v1 import AppTest
from core.analytics.irt import glmm_item_difficulty as g
from core.analytics.irt.irt_runner import run_2pl_model

fail = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok: fail.append(name)

# ---- D-5: items whose difficulty is the same at the 2 decimals shown must share a rank
r = g.rank_hardest(pd.Series([0.501, 0.504, 1.2, -0.3, 0.499]))
check("D-5: rank_hardest ties on the displayed (2-decimal) difficulty: [0.501,0.504,0.499] all share rank 2",
      list(r) == [2, 2, 1, 5, 2], str(list(r)))
rng = np.random.default_rng(3)
M = pd.DataFrame((rng.random((80, 6)) < np.linspace(.3, .8, 6)).astype(float), columns=[f"Q{i}" for i in range(1, 7)])
res = g.fit_item_difficulty_glmm(M, engine="python")
it = res["items"]
check("D-5: the fitted table's ranks equal the ranks of the rounded difficulties",
      list(it["rank_hardest"]) == list(g.rank_hardest(it["difficulty"])))

# ---- D-7: the engine's 2PL sample-size warnings must be shown in the CPI 2PL expander
rm = pd.DataFrame((rng.random((40, 8)) < .6).astype(float), columns=[f"Q{i}" for i in range(1, 9)])
res2 = run_2pl_model(rm, list(rm.columns))
check("D-7 premise: the 2PL engine returns at least one sample-size warning for 40 students", len(res2.get("warnings", [])) >= 1)


def app():
    import streamlit_app.dashboards.teacher_dashboard as td
    td._show_engine_warnings({"warnings": ["WARN-ONE small sample", "WARN-TWO few items"]})


at = AppTest.from_function(app, default_timeout=60).run()
txt = " ".join(str(c.value) for c in at.caption)
check("D-7: _show_engine_warnings prints every warning", not at.exception and "WARN-ONE" in txt and "WARN-TWO" in txt,
      str([e.value for e in at.exception]))

# ---- D-6: a long flags text must be fully readable (full list under the table + wide column)
import streamlit_app.dashboards.teacher_dashboard as td
check("D-6: helper that lists full flag text exists", hasattr(td, "_show_flag_details"))

print("\n%d check(s) failed" % len(fail) if fail else "\nAll checks passed")
sys.exit(1 if fail else 0)
