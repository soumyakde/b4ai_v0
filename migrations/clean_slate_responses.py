"""
migrations/clean_slate_responses.py
====================================
Phase R1 — Delete all research data from responses.db before pilot launch.

Purpose
-------
Removes all rows from research tables so the pilot starts with a clean
database where every new row is written with a RID from day one (after
Migrations 2 and 3 are applied).

The rid columns added by Migration 1 are preserved — schema is unchanged.
users.db is NOT touched — existing users and their RIDs are kept.

Safety
------
- Prints pre-deletion counts for every table.
- Requires typing CONFIRM to proceed.
- All deletes run in a single transaction — rolled back if anything fails.
- Prints post-deletion counts to verify.
- Run smoke_test.py after this script to confirm DB structure is intact.

Usage
-----
    python migrations/clean_slate_responses.py
    python migrations/clean_slate_responses.py --dry-run
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

_HERE         = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

RESPONSES_DB = Path(os.getenv("SQLITE_PATH", str(_PROJECT_ROOT / "responses.db")))

# Tables to clear — in dependency order (analytics outputs first,
# then core student data). Schema and rid columns are preserved.
TABLES = [
    # Analytics output tables (derived, can be regenerated)
    "dta_lo_results",
    "dta_results",
    "cpi_qual_scores",
    "cpi_summary",
    "ita_results",
    "cpi_runs",
    "dta_runs",
    "ita_runs",
    # Transcript store
    "transcripts",
    # Core student write tables
    "assessment_scores",
    "survey_scores",
    "completions",
    "responses",
]


def get_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {}
    for table in TABLES:
        try:
            counts[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            counts[table] = -1  # table doesn't exist
    return counts


def print_counts(label: str, counts: dict[str, int]) -> None:
    print(f"\n  {label}:")
    total = 0
    for table, n in counts.items():
        if n == -1:
            print(f"    {table:<26} (table not found)")
        else:
            print(f"    {table:<26} {n:>6} rows")
            total += n
    print(f"    {'TOTAL':<26} {total:>6} rows")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase R1 — Clean slate deletion of research data."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show counts without deleting anything.",
    )
    args = parser.parse_args()

    if not RESPONSES_DB.exists():
        print(f"\n❌ responses.db not found at: {RESPONSES_DB}")
        return 1

    print(f"\n{'='*55}")
    print(f"  Phase R1 — Clean Slate Deletion")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  DB: {RESPONSES_DB}")
    print(f"{'='*55}")

    conn = sqlite3.connect(RESPONSES_DB)

    pre_counts = get_counts(conn)
    print_counts("Pre-deletion counts", pre_counts)

    total_rows = sum(n for n in pre_counts.values() if n > 0)

    if total_rows == 0:
        print("\n  ✅ All tables already empty — nothing to delete.")
        conn.close()
        return 0

    if args.dry_run:
        print(f"\n  DRY RUN — {total_rows} rows would be deleted. No data written.")
        conn.close()
        return 0

    # Require explicit confirmation
    print(f"\n  ⚠️  This will permanently delete {total_rows} rows.")
    print(f"  users.db is NOT affected — all users and RIDs are preserved.")
    print(f"  Schema and rid columns are preserved.")
    print()
    confirm = input("  Type CONFIRM to proceed, anything else to abort: ").strip()
    if confirm != "CONFIRM":
        print("\n  Aborted — no data deleted.")
        conn.close()
        return 0

    # Delete in a single transaction
    try:
        conn.execute("BEGIN")
        for table in TABLES:
            try:
                conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass  # table doesn't exist — skip silently

        post_counts = get_counts(conn)
        print_counts("Post-deletion counts (before commit)", post_counts)

        # Verify all tables are empty
        non_empty = {t: n for t, n in post_counts.items() if n > 0}
        if non_empty:
            conn.rollback()
            print(f"\n  ❌ Verification failed — rolling back:")
            for t, n in non_empty.items():
                print(f"     {t}: {n} rows remain")
            conn.close()
            return 1

        conn.commit()
        print(f"\n  ✅ All {total_rows} rows deleted and verified.")
        print(f"  Schema intact — rid columns preserved.")
        print(f"  Run smoke_test.py to confirm DB structure.\n")

    except Exception as e:
        conn.rollback()
        print(f"\n  ❌ Error — rolled back: {e}")
        conn.close()
        return 1

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
