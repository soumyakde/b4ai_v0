"""UI guard for the Item Review tab (defects D-2 and D-4), using Streamlit's AppTest on a throw-away database.
D-2: the Item dropdown must stay on the chosen item after "Lock my rating" and after saving.
D-4: the "Saved ..." confirmations must still be on screen after the page reloads.
Run: set PYTHONUTF8=1, then python scripts/verify_item_review_ui.py"""
import os, sys, tempfile, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from pathlib import Path
from streamlit.testing.v1 import AppTest
from core.analytics.item_review import store

fail = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok: fail.append(name)

store._DB_PATH = Path(tempfile.mkdtemp()) / "ui_test.db"      # never touch the real responses.db


def app():
    import pandas as pd
    import streamlit_app.dashboards.teacher_dashboard as td
    td._render_item_review_tab("ui_tester", pd.DataFrame())


at = AppTest.from_function(app, default_timeout=60).run()
check("tab renders without an exception", not at.exception, str([e.value for e in at.exception]))
opts_before = list(at.selectbox(key="ir_item").options)
ids = [o.split(" ", 1)[-1] for o in opts_before]   # (older layout: options carried a status icon)
check("at least 3 items to pick from", len(ids) >= 3, str(len(ids)))
target = ids[2]

at.selectbox(key="ir_item").select(target).run()
check("picking the 3rd item works", at.selectbox(key="ir_item").value == target)

lock = [b for b in at.button if b.label.startswith("🔒 Lock")]
check("a Lock button is shown", len(lock) == 1)
lock[0].click().run()
check("D-2: after 'Lock my rating' the Item dropdown still shows the same item",
      at.selectbox(key="ir_item").value == target, f"now {at.selectbox(key='ir_item').value}, expected {target}")

check("D-2: the dropdown's option labels did not change after locking (a changing label makes the browser reset the widget)",
      list(at.selectbox(key="ir_item").options) == opts_before,
      f"{opts_before[2]!r} -> {list(at.selectbox(key='ir_item').options)[2]!r}")

save = [b for b in at.button if b.label.startswith("💾 Save final")]
check("a Save final decision button is shown after locking", len(save) == 1)
if save:
    save[0].click().run()
    check("D-2: after saving the final decision the dropdown still shows the same item",
          at.selectbox(key="ir_item").value == target, f"now {at.selectbox(key='ir_item').value}")
    check("D-4: 'Saved (...)' confirmation is still visible after the reload",
          any(str(s.value).startswith("Saved") for s in at.success), str([s.value for s in at.success]))
    at.run()    # a second, automatic rerun with no user action (timer, fragment, websocket reconnect)
    check("D-4: ...and still visible after a SECOND rerun with no user action",
          any(str(s.value).startswith("Saved") for s in at.success), str([s.value for s in at.success]))

ksave = [b for b in at.button if b.label.startswith("💾 Save key")]
if ksave:
    ksave[0].click().run()
    check("D-4: key decision 'Saved.' confirmation is still visible after the reload",
          any(str(s.value).startswith("Saved") for s in at.success), str([s.value for s in at.success]))
    at.run()
    check("D-4: ...key decision confirmation survives a second rerun",
          any(str(s.value).startswith("Saved") for s in at.success), str([s.value for s in at.success]))
    check("D-2: dropdown unchanged after key decision", at.selectbox(key="ir_item").value == target)

# a confirmation must not linger forever: moving to another item clears it
at.selectbox(key="ir_item").select(ids[0]).run()
check("confirmation is cleared when another item is opened",
      not any(str(s.value).startswith("Saved") for s in at.success))

print("\n%d check(s) failed" % len(fail) if fail else "\nAll checks passed")
sys.exit(1 if fail else 0)
