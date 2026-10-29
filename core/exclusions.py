"""
Participants excluded from every calculation (but never deleted from the database).

The list comes from the environment variable B4AI_EXCLUDE_USERS (comma-separated usernames, case-insensitive),
set in the local .env file and as a Railway variable. It is deliberately NOT a file in the repository: the
repository is public and a participant's username must not be committed to it.

Typical reason: a participant who left mid-session. Raw rows stay in responses.db; the data-loading paths
(canonical dataset, demographics, reflection notes, research export) skip them.
"""
import os
from pathlib import Path

ENV_VAR = "B4AI_EXCLUDE_USERS"
_BASE_DIR = Path(__file__).resolve().parents[1]


def excluded_user_ids() -> frozenset:
    """Lower-cased usernames to leave out of all calculations (empty set when the variable is unset)."""
    if ENV_VAR not in os.environ:
        try:
            from dotenv import load_dotenv
            load_dotenv(_BASE_DIR / ".env")  # does not override variables that are already set
        except Exception:
            pass
    raw = os.environ.get(ENV_VAR, "")
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def is_excluded(user_id) -> bool:
    return str(user_id).strip().lower() in excluded_user_ids()


def drop_excluded(df, column: str = "user_id"):
    """Return df without the rows of excluded participants (df is returned unchanged when none are excluded)."""
    ex = excluded_user_ids()
    if not ex or df is None or column not in getattr(df, "columns", []):
        return df
    return df[~df[column].astype(str).str.strip().str.lower().isin(ex)].reset_index(drop=True)
