"""Checks for admin-imported cohorts (core/analytics/datasets/imported_cohorts.py + loader switch + dashboard filter).
Uses a temporary copy of a tiny responses.db; never touches a real database.
Run: python -X utf8 scripts/verify_imported_cohorts.py"""
import os, sys, sqlite3, tempfile, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import pandas as pd
import core.analytics.datasets.imported_cohorts as ic

fail = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def canon(users=("UB1-01", "UB1-02"), cohort="UB_test", n_mod=2):
    rows = []
    for u in users:
        for m in range(1, n_mod + 1):
            for q, lab, sc in (("Q2_1", "Agree", 3.0), ("Q2_2", "Disagree", 2.0)):
                rows.append(dict(user_id=u, module_id=f"module_{m}", instrument_key=f"module{m}_b4ai_sims_survey", question_id=q,
                                 response_value=lab, item_score=sc, construct="intrinsic_motivation", grade=None,
                                 submitted_at="2026-02-16T10:00:00", completed_at="", cohort_id=cohort))
    return pd.DataFrame(rows)


demo = pd.DataFrame([dict(user_id="UB1-01", grade="Fifth (5th) grade", gender="Female", first_language_english=False),
                     dict(user_id="UB1-02", grade="Adult", gender="Male", first_language_english=True)])
tmp = tempfile.mkdtemp()
db = os.path.join(tmp, "responses.db")
sqlite3.connect(db).executescript("CREATE TABLE responses(user_id TEXT, instrument_name TEXT, question_id TEXT, response_value TEXT, submitted_at TEXT);").close()

# --- validation
v = ic.validate_import(canon(), demo, {"alice"}, {"cohortA"}, known_instruments={"module1_b4ai_sims_survey", "module2_b4ai_sims_survey"})
check("valid file has no errors", not v["errors"], str(v["errors"]))
check("summary counts", int(v["summary"].students.iloc[0]) == 2 and int(v["summary"].modules.iloc[0]) == 2)
check("missing column blocks", bool(ic.validate_import(canon().drop(columns=["construct"]), demo, set(), set())["errors"]))
check("name-like ID blocked", bool(ic.validate_import(canon(users=("Jane Doe",)), None, set(), set())["errors"]))
check("clash with registered user blocked", bool(ic.validate_import(canon(), demo, {"UB1-01"}, set())["errors"]))
check("clash with registered cohort blocked", bool(ic.validate_import(canon(), demo, set(), {"UB_test"})["errors"]))
check("duplicate rows blocked", bool(ic.validate_import(pd.concat([canon(), canon()]), demo, set(), set())["errors"]))
check("unknown instrument blocked", bool(ic.validate_import(canon(), demo, set(), set(), known_instruments={"x"})["errors"]))
check("student in two cohorts blocked", bool(ic.validate_import(pd.concat([canon(), canon(cohort="other")]), demo, set(), set())["errors"]))
bad = canon(); bad.loc[0, "item_score"] = 4.0
check("inconsistent score for same answer blocked", bool(ic.validate_import(bad, demo, set(), set())["errors"]))
v2 = ic.validate_import(canon(), demo.iloc[:1], set(), set())
check("missing demographics = warning not error", not v2["errors"] and any("no demographics" in w for w in v2["warnings"]))

# --- save / list / load / replace / delete
check("nothing imported yet", ic.list_imports(db).empty)
r = ic.save_import(db, canon(), demo, "batch1")
check("save returns counts", r == {"rows": 8, "students": 2, "cohorts": 1}, str(r))
li = ic.list_imports(db)
check("list shows cohort", list(li.cohort_id) == ["UB_test"] and int(li.students[0]) == 2)
c, d, cm = ic.load_imported(db)
check("canonical shape", list(c.columns) == ic.CANON_COLS and len(c) == 8)
check("grade carried onto canonical rows", set(c.grade) == {"Fifth (5th) grade", "Adult"})
check("demographics decoded", d.set_index("user_id").loc["UB1-01", "first_language_english"] is False
      and d.set_index("user_id").loc["UB1-02", "first_language_english"] is True
      and d.set_index("user_id").loc["UB1-01", "grade_level"] == 5
      and d.set_index("user_id").loc["UB1-02", "grade_level"] == "Adult")
check("cohort map", cm == {"UB1-01": "UB_test", "UB1-02": "UB_test"})
ic.save_import(db, canon(users=("UB1-01", "UB1-03")), demo, "batch2")
c2, d2, _ = ic.load_imported(db)
check("re-import replaces the cohort (no duplicates)", len(c2) == 8 and set(c2.user_id) == {"UB1-01", "UB1-03"} and set(d2.user_id) == {"UB1-01", "UB1-03"})
ic.save_import(db, canon(users=("V-01",), cohort="Valley"), None, "batch3")
check("two cohorts coexist", set(ic.list_imports(db).cohort_id) == {"UB_test", "Valley"})
vd = ic.load_imported(db, ["Valley"])[1]
check("student without demographics gets an empty row", list(vd.user_id) == ["V-01"] and vd.gender.isna().all())
check("responses table untouched", sqlite3.connect(db).execute("select count(*) from responses").fetchone()[0] == 0)
check("delete removes only that cohort", ic.delete_imported_cohort(db, "Valley") == 4 and list(ic.list_imports(db).cohort_id) == ["UB_test"])

# --- loader switch (platform data empty -> imported still loads only when asked)
import core.analytics.datasets.canonical_loader as cl
cl_ = cl.load_canonical_data(db_path=__import__("pathlib").Path(db))
check("default load excludes imported", cl_[0].empty and not cl_[2])
cl2 = cl.load_canonical_data(db_path=__import__("pathlib").Path(db), include_imported=True)
check("include_imported adds rows, demographics, cohort map",
      len(cl2[0]) == 8 and set(cl2[1].user_id) == {"UB1-01", "UB1-03"} and set(cl2[2].values()) == {"UB_test"})

# --- dashboard filter picks the imported cohort up
import streamlit_app.dashboards.teacher_dashboard as td
fc, fd = td._apply_filters(cl2[0], cl2[1], [], [], [], "All", ["UB_test"], "All students", cl2[2])
check("cohort filter works on imported cohort", len(fc) == 8)
fc, fd = td._apply_filters(cl2[0], cl2[1], [], ["Fifth (5th) grade"], [], "All", [], "All students", cl2[2])
check("grade filter works on imported students", set(fc.user_id) == {"UB1-01"} and len(fc) == 4)

print("\n%d failure(s)" % len(fail))
sys.exit(1 if fail else 0)
