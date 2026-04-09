"""
migrations/backfill_rids.py
===========================
Migration 1b — Generate and backfill Research Identifiers (RIDs)

Purpose
-------
After migration_1_add_rid_columns.sql has been run, this script:
  1. Generates a deterministic, non-reversible RID for every existing user
  2. Writes RIDs to users.db (users.rid)
  3. Backfills RIDs to every research table in responses.db by joining
     on the existing user_id (which stores the username string)
  4. Verifies backfill completeness before committing
  5. Prints a summary report

RID Algorithm
-------------
HMAC-SHA256(secret_salt, username) → first 16 hex chars, uppercased.

Properties:
  - Deterministic: same username + same salt → same RID always.
    This allows safe re-runs and future data corrections.
  - Non-reversible: without RID_SALT, the username cannot be recovered
    from the RID. HMAC is a one-way function.
  - Collision-resistant: 16 hex chars = 64 bits of output. At N=1000
    users the collision probability is negligible (~2.7 × 10⁻¹⁵).
  - IRB-compatible: satisfies "indirect identifier" requirements when
    RID_SALT is stored separately from research data.

Prerequisites
-------------
  1. Run migration_1_add_rid_columns.sql against both databases.
  2. Set RID_SALT in your .env file:
       RID_SALT=<random 32+ char string, never committed to git>
     Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
  3. Take a backup of both databases before running this script.

Usage
-----
  From project root:
    python migrations/backfill_rids.py [--dry-run] [--verify-only]

  --dry-run      : Print what would happen without writing anything.
  --verify-only  : Skip generation, only verify existing RID coverage.

Safety
------
  - All writes are wrapped in a single transaction per database.
  - If any verification check fails, the transaction is rolled back.
  - The script is idempotent: re-running it will not change existing RIDs.
    Only NULL rid rows are updated.
  - Existing data is never deleted.
"""

import argparse
import hmac
import hashlib
import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Bootstrap: load .env before anything else
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv not available — rely on environment variables being set

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USERS_DB_PATH     = Path(os.getenv("USERS_DB_PATH",  str(_PROJECT_ROOT / "users.db")))
RESPONSES_DB_PATH = Path(os.getenv("SQLITE_PATH",    str(_PROJECT_ROOT / "responses.db")))
RID_SALT          = os.getenv("RID_SALT", "")
RESOLVER_VERSION  = "v1"

# Research tables: (table_name, subject_column_that_holds_username)
# participant_id columns hold the same username strings as user_id columns.
RESEARCH_TABLES = [
    ("responses",         "user_id"),
    ("completions",       "user_id"),
    ("assessment_scores", "user_id"),
    ("survey_scores",     "user_id"),
    ("cpi_summary",       "participant_id"),
    ("cpi_qual_scores",   "participant_id"),
    ("dta_results",       "participant_id"),
    ("dta_lo_results",    "participant_id"),
    ("transcripts",       "participant_id"),
]

# ---------------------------------------------------------------------------
# RID generation
# ---------------------------------------------------------------------------

def generate_rid(username: str, salt: str) -> str:
    """
    Generate a deterministic, non-reversible Research Identifier.

    HMAC-SHA256(salt, username) → first 16 hex chars, uppercased.
    Same inputs always produce the same output.
    Without the salt, the username cannot be recovered.
    """
    return hmac.new(
        salt.encode("utf-8"),
        username.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16].upper()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_environment() -> list[str]:
    """Return list of blocking errors. Empty list = safe to proceed."""
    errors = []

    if not RID_SALT:
        errors.append(
            "RID_SALT is not set in environment or .env file.\n"
            "  Generate one: python -c \"import secrets; print(secrets.token_hex(32))\"\n"
            "  Add to .env:  RID_SALT=<generated value>"
        )
    elif len(RID_SALT) < 32:
        errors.append(
            f"RID_SALT is only {len(RID_SALT)} characters. "
            "Use at least 32 characters for adequate entropy."
        )

    if not USERS_DB_PATH.exists():
        errors.append(f"users.db not found at: {USERS_DB_PATH}")

    if not RESPONSES_DB_PATH.exists():
        errors.append(f"responses.db not found at: {RESPONSES_DB_PATH}")

    return errors


def check_rid_column_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Return True if the 'rid' column exists in the given table."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return "rid" in cols


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------

def get_all_usernames(users_conn: sqlite3.Connection) -> list[str]:
    """Return all usernames from users.db."""
    rows = users_conn.execute("SELECT username FROM users").fetchall()
    return [r[0] for r in rows]


def build_username_to_rid_map(usernames: list[str], salt: str) -> dict[str, str]:
    """Generate RID for every username. Returns {username: rid}."""
    return {username: generate_rid(username, salt) for username in usernames}


def backfill_users_table(
    users_conn: sqlite3.Connection,
    username_to_rid: dict[str, str],
    dry_run: bool,
) -> tuple[int, int]:
    """
    Write RIDs to users.rid where currently NULL.
    Returns (updated_count, already_had_rid_count).
    """
    rows = users_conn.execute(
        "SELECT username, rid FROM users"
    ).fetchall()

    updated = 0
    already_set = 0

    for username, existing_rid in rows:
        if existing_rid is not None:
            already_set += 1
            continue
        rid = username_to_rid.get(username)
        if rid is None:
            continue
        if not dry_run:
            users_conn.execute(
                "UPDATE users SET rid = ? WHERE username = ?",
                (rid, username),
            )
        updated += 1

    return updated, already_set


def backfill_research_table(
    responses_conn: sqlite3.Connection,
    table: str,
    subject_col: str,
    username_to_rid: dict[str, str],
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Write RIDs to a research table where currently NULL.
    Returns (updated, already_set, unknown_username).
    unknown_username = rows whose subject_col value has no RID mapping.
    """
    if not check_rid_column_exists(responses_conn, table):
        print(f"  ⚠️  Table '{table}' does not have a 'rid' column — "
              f"run migration_1_add_rid_columns.sql first. Skipping.")
        return 0, 0, 0

    rows = responses_conn.execute(
        f"SELECT id, {subject_col}, rid FROM {table}"
    ).fetchall()

    updated = 0
    already_set = 0
    unknown = 0

    for row_id, subject_value, existing_rid in rows:
        if existing_rid is not None:
            already_set += 1
            continue
        rid = username_to_rid.get(subject_value)
        if rid is None:
            unknown += 1
            continue
        if not dry_run:
            responses_conn.execute(
                f"UPDATE {table} SET rid = ? WHERE id = ?",
                (rid, row_id),
            )
        updated += 1

    return updated, already_set, unknown


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_backfill(
    users_conn: sqlite3.Connection,
    responses_conn: sqlite3.Connection,
) -> list[str]:
    """
    Run post-backfill verification checks.
    Returns list of failure messages. Empty = all checks passed.
    """
    failures = []

    # Check 1: No NULL rids in users table
    null_users = users_conn.execute(
        "SELECT COUNT(*) FROM users WHERE rid IS NULL"
    ).fetchone()[0]
    if null_users > 0:
        failures.append(
            f"users.rid: {null_users} rows still NULL after backfill."
        )

    # Check 2: RIDs are unique in users table
    total_users = users_conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    unique_rids = users_conn.execute(
        "SELECT COUNT(DISTINCT rid) FROM users WHERE rid IS NOT NULL"
    ).fetchone()[0]
    if unique_rids != total_users:
        failures.append(
            f"users.rid: collision detected — "
            f"{total_users} users but only {unique_rids} unique RIDs."
        )

    # Check 3: Core research tables have no NULL rids
    # (NULL is acceptable in analytics output tables which may be empty)
    core_tables = [
        ("responses",         "user_id"),
        ("completions",       "user_id"),
        ("assessment_scores", "user_id"),
        ("survey_scores",     "user_id"),
    ]
    for table, _ in core_tables:
        try:
            total = responses_conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            if total == 0:
                continue  # empty table is fine
            null_rids = responses_conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE rid IS NULL"
            ).fetchone()[0]
            if null_rids > 0:
                failures.append(
                    f"{table}.rid: {null_rids}/{total} rows still NULL. "
                    f"These rows have user_id values not in users.db."
                )
        except sqlite3.OperationalError as e:
            failures.append(f"{table}: query error — {e}")

    return failures


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(
    username_to_rid: dict[str, str],
    users_result: tuple,
    table_results: dict,
    failures: list[str],
    dry_run: bool,
    elapsed_ms: float,
) -> None:
    mode = "DRY RUN — " if dry_run else ""
    print(f"\n{'='*60}")
    print(f"  {mode}RID Backfill Summary")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")
    print(f"\n  RIDs generated for {len(username_to_rid)} users")
    print(f"\n  users.db:")
    print(f"    Updated:       {users_result[0]}")
    print(f"    Already set:   {users_result[1]}")
    print(f"\n  responses.db:")
    for table, (updated, already_set, unknown) in table_results.items():
        print(f"    {table:<22} updated={updated:>5}  "
              f"already_set={already_set:>5}  unknown={unknown:>3}")
    print(f"\n  Elapsed: {elapsed_ms:.0f}ms")
    if failures:
        print(f"\n  {'='*40}")
        print(f"  ❌ VERIFICATION FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"     • {f}")
        print(f"  Transaction rolled back. No data was modified.")
        print(f"  {'='*40}")
    else:
        if dry_run:
            print(f"\n  ✅ Dry run complete — no data written.")
        else:
            print(f"\n  ✅ Backfill complete and verified.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill RIDs for existing users and research data."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without writing anything.",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Skip generation — only verify existing RID coverage.",
    )
    args = parser.parse_args()

    # Validate environment
    errors = validate_environment()
    if errors:
        print("\n❌ Cannot proceed — fix these issues first:\n")
        for e in errors:
            print(f"  • {e}\n")
        return 1

    start = datetime.utcnow()

    print(f"\nUsers DB:    {USERS_DB_PATH}")
    print(f"Responses DB: {RESPONSES_DB_PATH}")
    print(f"Mode: {'verify-only' if args.verify_only else 'dry-run' if args.dry_run else 'LIVE WRITE'}")

    # Connect
    users_conn     = sqlite3.connect(USERS_DB_PATH)
    responses_conn = sqlite3.connect(RESPONSES_DB_PATH)

    try:
        # Build RID map
        usernames      = get_all_usernames(users_conn)
        username_to_rid = build_username_to_rid_map(usernames, RID_SALT)

        if args.verify_only:
            failures = verify_backfill(users_conn, responses_conn)
            if failures:
                print(f"\n❌ Verification failed:")
                for f in failures:
                    print(f"  • {f}")
                return 1
            else:
                print(f"\n✅ Verification passed — all RIDs present and unique.")
                return 0

        # Backfill users.db in a transaction
        users_conn.execute("BEGIN")
        users_result = backfill_users_table(
            users_conn, username_to_rid, args.dry_run
        )

        # Backfill responses.db in a transaction
        responses_conn.execute("BEGIN")
        table_results = {}
        for table, subject_col in RESEARCH_TABLES:
            table_results[table] = backfill_research_table(
                responses_conn, table, subject_col,
                username_to_rid, args.dry_run,
            )

        # Verify before committing
        failures = [] if args.dry_run else verify_backfill(users_conn, responses_conn)

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000

        print_summary(
            username_to_rid, users_result, table_results,
            failures, args.dry_run, elapsed,
        )

        if failures:
            users_conn.rollback()
            responses_conn.rollback()
            return 1

        if not args.dry_run:
            users_conn.commit()
            responses_conn.commit()
        else:
            users_conn.rollback()
            responses_conn.rollback()

        return 0

    except Exception as e:
        users_conn.rollback()
        responses_conn.rollback()
        print(f"\n❌ Unexpected error — transaction rolled back: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        users_conn.close()
        responses_conn.close()


if __name__ == "__main__":
    sys.exit(main())
