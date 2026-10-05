"""
clear_item_review_test_data.py -- remove Item Review test entries from a database.

Item Review stores three kinds of rows in responses.db:
  item_human_ratings : a reviewer's ratings        (identified by the reviewer's username)
  item_llm_runs      : one row per AI run          (identified by who started it: created_by)
  item_llm_results   : the AI answers of each run  (linked to a run by run_id)

DRY RUN by default: it only COUNTS what would be deleted. Add --apply to delete.
A backup copy of the database is written first (responses.db.item_review_backup_<timestamp>).

Usage (project root, env b4ai_v0):
    python scripts/clear_item_review_test_data.py --user tester1 [--user tester2 ...]            (dry run)
    python scripts/clear_item_review_test_data.py --user tester1 --user tester2 --apply          (delete)
    python scripts/clear_item_review_test_data.py --db path\\to\\responses.db --user tester1      (other database)

Only rows belonging to the usernames you list are touched; nothing else in the database is changed.
"""
import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "responses.db"))
    ap.add_argument("--user", action="append", required=True, help="username to clear (repeatable)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default is a dry run)")
    a = ap.parse_args()
    if not os.path.exists(a.db):
        print("Database not found:", a.db)
        return 1
    c = sqlite3.connect(a.db)
    have = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"item_human_ratings", "item_llm_runs", "item_llm_results"} <= have:
        print("No Item Review tables in this database; nothing to clear.")
        return 0
    ph = ",".join("?" * len(a.user))
    runs = [r[0] for r in c.execute(f"SELECT run_id FROM item_llm_runs WHERE created_by IN ({ph})", a.user)]
    n_h = c.execute(f"SELECT COUNT(*) FROM item_human_ratings WHERE rater IN ({ph})", a.user).fetchone()[0]
    n_res = (c.execute(f"SELECT COUNT(*) FROM item_llm_results WHERE run_id IN ({','.join('?' * len(runs))})", runs).fetchone()[0]
             if runs else 0)
    print(f"Database: {a.db}\nUsers: {', '.join(a.user)}")
    print(f"  ratings by these users          : {n_h}")
    print(f"  AI runs started by these users  : {len(runs)}")
    print(f"  AI answers in those runs        : {n_res}")
    if not a.apply:
        print("\nDRY RUN: nothing deleted. Re-run with --apply to delete.")
        return 0
    backup = f"{a.db}.item_review_backup_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copyfile(a.db, backup)
    print("\nBackup written:", backup)
    if runs:
        c.execute(f"DELETE FROM item_llm_results WHERE run_id IN ({','.join('?' * len(runs))})", runs)
        c.execute(f"DELETE FROM item_llm_runs WHERE run_id IN ({','.join('?' * len(runs))})", runs)
    c.execute(f"DELETE FROM item_human_ratings WHERE rater IN ({ph})", a.user)
    c.commit()
    print("Deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
