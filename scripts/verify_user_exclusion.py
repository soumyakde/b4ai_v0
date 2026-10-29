"""
verify_user_exclusion.py -- checks the B4AI_EXCLUDE_USERS mechanism (core/exclusions.py).

Uses a temporary COPY of responses.db (never writes to the real one). The excluded name is chosen from the data,
so the script works on any database that has at least two responders.

    python scripts/verify_user_exclusion.py [path/to/responses.db]      (exit 0 = all pass)
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

fail = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        fail.append(name)

import pandas as pd
from core import exclusions as ex
from core.analytics.datasets.canonical_loader import load_canonical_data

src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "responses.db"
tmp = Path(tempfile.mkdtemp()) / "responses_copy.db"
shutil.copyfile(src, tmp)
users = [r[0] for r in sqlite3.connect(tmp).execute("select distinct user_id from responses order by 1")]
check("database has at least 2 responders", len(users) >= 2, str(len(users)))
target = users[0]

# unit behaviour
os.environ.pop(ex.ENV_VAR, None)
ex._BASE_DIR = Path(tempfile.mkdtemp())          # no .env to fall back to
check("unset variable -> empty set", ex.excluded_user_ids() == frozenset())
os.environ[ex.ENV_VAR] = f" {target.upper()} , other "
check("case-insensitive, trims spaces, comma list", ex.excluded_user_ids() == frozenset({target.lower(), "other"}))
check("is_excluded true for target (any case)", ex.is_excluded(target) and ex.is_excluded(target.swapcase()))
check("is_excluded false for someone else", not ex.is_excluded("someone_else_entirely"))
d = pd.DataFrame({"user_id": [target, "x", target.upper(), "y"], "v": [1, 2, 3, 4]})
check("drop_excluded removes the user's rows only", list(ex.drop_excluded(d)["user_id"]) == ["x", "y"])
os.environ.pop(ex.ENV_VAR, None)
check("drop_excluded with no exclusions returns frame unchanged", len(ex.drop_excluded(d)) == 4)

# end to end through the real loader
base_df, base_demo, base_map = load_canonical_data(tmp)
check("without exclusion the user is present in the canonical data", target in set(base_df["user_id"]))
os.environ[ex.ENV_VAR] = target
df, demo, cmap = load_canonical_data(tmp)
check("with exclusion the user is absent from the canonical data", target not in set(df["user_id"]))
check("demographics exclude the user", target not in set(demo["user_id"]))
check("cohort map excludes the user", target not in cmap)
check("everyone else is unchanged (user count -1)", df["user_id"].nunique() == base_df["user_id"].nunique() - 1,
      f"{base_df['user_id'].nunique()} -> {df['user_id'].nunique()}")
rest_base = base_df[base_df["user_id"] != target].sort_values(["user_id", "instrument_key", "question_id"]).reset_index(drop=True)
rest_new = df.sort_values(["user_id", "instrument_key", "question_id"]).reset_index(drop=True)
check("remaining rows identical to the baseline minus the user", len(rest_base) == len(rest_new)
      and (rest_base["item_score"].fillna(-1).values == rest_new["item_score"].fillna(-1).values).all())

# the raw database is untouched
n_raw = sqlite3.connect(tmp).execute("select count(*) from responses where user_id=?", (target,)).fetchone()[0]
check("raw rows of the excluded user remain in the database", n_raw > 0, str(n_raw))

print()
print("ALL CHECKS PASSED" if not fail else f"{len(fail)} FAILED: {fail}")
sys.exit(1 if fail else 0)
