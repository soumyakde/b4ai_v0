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

# ✅ Resolver instance (safe — lightweight, no state mutation)
resolver = LearningUnitResolver()


def submit_instrument(
    user_id: str,
    module_id: str,
    instrument_key: str,
    responses: dict,
    score: float | None = None
):
    """
    Handles submission of a survey or assessment instrument.

    ARCHITECTURE RULES:
    - Instrument semantics come from progress_engine
    - Resolver provides validated module access
    - Registry is NOT used for completion logic
    """

    # --------------------------------------------------
    # 1️⃣ Resolve module once (single source of truth)
    # --------------------------------------------------
    module = resolver.get(module_id)

    # Validate instrument exists in module definition
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
    # 2️⃣ Store individual responses
    # --------------------------------------------------
    for question_id, response_value in responses.items():
        cur.execute("""
            INSERT INTO responses
            (user_id, instrument_name, question_id, response_value, submitted_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            instrument_key,
            question_id,
            str(response_value),
            timestamp
        ))

    # --------------------------------------------------
    # 3️⃣ Store score (if provided)
    # --------------------------------------------------
    if score is not None:

        instrument_type = get_instrument_type(module, instrument_key)

        if instrument_type == "survey":
            cur.execute("""
                INSERT INTO survey_scores
                (user_id, survey_key, score, calculated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, survey_key)
                DO UPDATE SET
                    score = excluded.score,
                    calculated_at = excluded.calculated_at
            """, (user_id, instrument_key, score, timestamp))

        elif instrument_type == "assessment":
            cur.execute("""
                INSERT INTO assessment_scores
                (user_id, assessment_code, score, calculated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, assessment_code)
                DO UPDATE SET
                    score = excluded.score,
                    calculated_at = excluded.calculated_at
            """, (user_id, instrument_key, score, timestamp))

        else:
            raise ValueError(
                f"Unsupported instrument type '{instrument_type}'."
            )

    conn.commit()
    conn.close()

    # --------------------------------------------------
    # 4️⃣ Mark instrument complete (authoritative path)
    # --------------------------------------------------
    # mark_instrument_complete internally re-validates via resolver + progress_engine
    mark_instrument_complete(user_id, module_id, instrument_key)

    return {
        "status": "success",
        "instrument": instrument_key,
        "module": module_id
    }