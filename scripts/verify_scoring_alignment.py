"""
verify_scoring_alignment.py

Checks that the submission-time scoring path (core.scoring_engine.compute_score)
and the dashboard path (DatasetBuilder via load_canonical_data) agree, and that
the research export returns one row per response.

Usage (from the project root, conda env b4ai_v0):
    python scripts/verify_scoring_alignment.py [path/to/responses.db]

It works on a temporary COPY of the database, so it never writes to the real one.
Exit code 0 = all checks passed.
"""
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.scoring_engine import compute_score, normalize_choice  # noqa: E402
from utils.yaml_loader import load_yaml  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        failures.append(name)


# 1. normalize_choice unit cases
check("normalize_choice('A: Artificial Intelligence') == 'A'", normalize_choice("A: Artificial Intelligence") == "A")
check("normalize_choice('B') == 'B'", normalize_choice("B") == "B")
check("normalize_choice('False') == 'False'", normalize_choice(" False ") == "False")
check("normalize_choice(None) is None", normalize_choice(None) is None)

# 2. compute_score unit cases (the original defect: 'A: text' never equalled 'A')
key = {"correct_answers": {"Q1": "A", "Q2": "B", "Q3": "True"}}
check("compute_score scores option-style answers", compute_score({"Q1": "A: yes", "Q2": "A: no", "Q3": "True"}, key) == 2)
check("compute_score scores letter answers", compute_score({"Q1": "A", "Q2": "B"}, key) == 2)
check("compute_score subset scoring unchanged", compute_score({"Q2": "B"}, key) == 1)

# 3. replay every real binary instrument through BOTH paths on a copy of the database
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "responses.db"
tmp = Path(tempfile.mkdtemp()) / "responses_copy.db"
shutil.copyfile(src, tmp)

from core.analytics.datasets.canonical_loader import load_canonical_data  # noqa: E402

canonical_df, _, _ = load_canonical_data(tmp)

binary = [n for n in canonical_df["instrument_key"].unique()
          if n.endswith("_assessment") and "mcq" in n or n in (
              "precourse_pre_aici_assessment", "postcourse_post_aici_assessment",
              "precourse_pre_ai_misconceptions_assessment", "postcourse_post_ai_misconceptions_assessment")]

builder_tot = (canonical_df[canonical_df["instrument_key"].isin(binary)]
               .groupby(["instrument_key", "user_id"])["item_score"].sum())

conn = sqlite3.connect(tmp)
resp = defaultdict(dict)
first_seen = {}
for inst, user, q, v, ts in conn.execute(
        "SELECT instrument_name, user_id, question_id, response_value, submitted_at FROM responses"):
    if inst in binary:
        resp[(inst, user)][q] = v
        first_seen[(inst, user)] = min(ts, first_seen.get((inst, user), ts))

mismatch = compared = legacy_used = 0
yaml_cache = {}
surveys_dir = ROOT / "streamlit_app" / "surveys"
for (inst, user), answers in resp.items():
    base = inst.split("_", 1)[1] if re.match(r"^(precourse|postcourse)_", inst) else inst
    fn = "surveys/%s_scoring.yaml" % base
    if fn not in yaml_cache:
        yaml_cache[fn] = load_yaml(fn)
    key = yaml_cache[fn]
    # Responses submitted before a legacy key's valid_until are scored with that key
    # (same rule the dashboard applies), taking the latest applicable legacy file.
    for lp in sorted(surveys_dir.glob("%s_scoring_legacy_*.yaml" % base), reverse=True):
        lf = load_yaml("surveys/" + lp.name)
        if first_seen[(inst, user)] < str(lf["valid_until"]):
            key = lf
            legacy_used += 1
    # score only items the key knows about, as the dashboard does
    answers = {q: v for q, v in answers.items() if q in key["correct_answers"]}
    engine = compute_score(answers, key)
    compared += 1
    if abs(engine - builder_tot.get((inst, user), float("nan"))) > 1e-9:
        mismatch += 1
check("scoring engine == DatasetBuilder for every student/instrument", mismatch == 0 and compared > 0,
      "%d compared, %d mismatches, %d scored with a legacy key" % (compared, mismatch, legacy_used))

# no binary response may be left unscorable (NaN) now that legacy keys exist
bin_rows = canonical_df[canonical_df["instrument_key"].isin(binary)]
nan_rows = int(bin_rows["item_score"].isna().sum())
check("every binary-assessment response is scorable", nan_rows == 0, "%d unscorable of %d" % (nan_rows, len(bin_rows)))

# 4. research export returns one row per response (the old query returned ~17.6M)
n_resp = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
check("canonical data has one row per response", len(canonical_df) == n_resp, "%d rows vs %d responses" % (len(canonical_df), n_resp))
dups = canonical_df.duplicated(["user_id", "instrument_key", "question_id", "submitted_at"]).sum()
check("no duplicated response rows", dups == 0, "%d duplicates" % dups)

conn.close()
print("\n%s" % ("ALL CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures)))
sys.exit(1 if failures else 0)
