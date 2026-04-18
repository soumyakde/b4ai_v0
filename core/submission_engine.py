# core/submission_engine.py

from datetime import datetime

from modules.resolution.learning_unit_resolver import LearningUnitResolver
from core.progress_engine import (
    get_required_instruments,
    get_instrument_type
)
from core.db_utils import (
    get_connection,
    mark_instrument_complete
)

# Resolver instance (safe — lightweight, no state mutation)
resolver = LearningUnitResolver()


def submit_instrument(
    user_id: str,
    module_id: str,
    instrument_key: str,
    responses: dict,
    score: float | None = None,
    rid: str | None = None,
):
    """
    Handles submission of a survey or assessment instrument.

    ARCHITECTURE RULES:
    - Instrument semantics come from progress_engine
    - Resolver provides validated module access
    - Registry is NOT used for completion logic

    Migration 3: rid written alongside user_id in all research tables.
    Auto-scoring: surveys scored from YAML when score=None.
    """
    # Migration 3 — resolve RID if caller did not supply one
    if rid is None:
        try:
            from auth.user_manager import get_user_rid
            rid = get_user_rid(user_id)
        except Exception:
            rid = None

    # --------------------------------------------------
    # 1 — Resolve module
    # --------------------------------------------------
    module = resolver.get(module_id)
    required = get_required_instruments(module)

    if instrument_key not in required:
        raise ValueError(
            f"Instrument '{instrument_key}' is not defined "
            f"for module '{module_id}'."
        )

    conn = get_connection()
    cur = conn.cursor()
    timestamp = datetime.utcnow().isoformat()

    # --------------------------------------------------
    # 2 — Store individual responses
    # --------------------------------------------------
    module_prefix = module_id.replace("_", "")
    stored_name = (
        instrument_key
        if instrument_key.startswith(module_prefix)
        else f"{module_prefix}_{instrument_key}"
    )

    for question_id, response_value in responses.items():
        cur.execute("""
            INSERT INTO responses
            (user_id, rid, instrument_name, question_id, response_value, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, rid, stored_name, question_id, str(response_value), timestamp))

    # --------------------------------------------------
    # 3 — Store score (auto-compute for surveys if score=None)
    # --------------------------------------------------
    instrument_type = get_instrument_type(module, instrument_key)

    # Auto-compute survey score when caller passes score=None.
    # Reflections have instrument_type="reflection" and are skipped.
    if score is None and instrument_type == "survey":
        try:
            from core.scoring_engine import compute_score
            from pathlib import Path
            import yaml
            _scoring_dir = Path(__file__).resolve().parents[1] / "streamlit_app" / "surveys"
            _file_key = instrument_key.removesuffix("_survey")
            _scoring_file = _scoring_dir / f"{_file_key}_scoring.yaml"
            if _scoring_file.exists():
                with open(_scoring_file, "r", encoding="utf-8") as _f:
                    _scoring_yaml = yaml.safe_load(_f)
                score = float(compute_score(responses, _scoring_yaml))
        except Exception:
            score = None  # YAML missing or parse error — store None, no crash

    if score is not None:

        if instrument_type == "survey":
            cur.execute("""
                INSERT INTO survey_scores
                (user_id, rid, survey_key, score, calculated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, survey_key)
                DO UPDATE SET
                    rid = excluded.rid,
                    score = excluded.score,
                    calculated_at = excluded.calculated_at
            """, (user_id, rid, instrument_key, score, timestamp))

        elif instrument_type == "assessment":
            cur.execute("""
                INSERT INTO assessment_scores
                (user_id, rid, assessment_code, score, calculated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, assessment_code)
                DO UPDATE SET
                    rid = excluded.rid,
                    score = excluded.score,
                    calculated_at = excluded.calculated_at
            """, (user_id, rid, instrument_key, score, timestamp))

        else:
            raise ValueError(f"Unsupported instrument type '{instrument_type}'.")

    conn.commit()
    conn.close()

    # --------------------------------------------------
    # 4 — Mark instrument complete
    # --------------------------------------------------
    mark_instrument_complete(user_id, module_id, instrument_key)

    return {"status": "success", "instrument": instrument_key, "module": module_id}
