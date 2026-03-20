"""
Scoring Engine

Responsibilities:
- Deterministically compute score from:
    - responses dict
    - scoring YAML dict

Design Principles:
- Pure function (no DB, no UI, no side effects)
- Strict validation (fail fast on misconfiguration)
- Deterministic output
- Defensive against schema drift
- Clear error messaging

Supported YAML Formats:

OPTION A (Correct Answers — Binary)
-------------------------------------
correct_answers:
  Q1: "A"
  Q2: "B"

NOTE: responses may be a SUBSET of correct_answers keys.
This supports randomised MCQ draws (e.g. 10 questions from
a bank of 57). Unknown question IDs in responses are
rejected; missing questions simply score 0.

OPTION B (Point Mapping — Weighted)
-------------------------------------
points:
  Q1:
    A: 1
    B: 0
  Q2:
    Yes: 2
    No: 0

OPTION C (Likert — Default + Reverse Scales)
----------------------------------------------
scoring_type: likert

default_scale:
  Strongly disagree: 1
  Disagree: 2
  Agree: 3
  Strongly agree: 4

reverse_scale:
  Strongly disagree: 4
  Disagree: 3
  Agree: 2
  Strongly agree: 1

reverse_questions:       # questions scored with reverse_scale
  - Q9_1
  - Q9_2

Non-Likert responses (e.g. text initials like Q1_1) are
silently skipped — they will not be in either scale.
"""

from typing import Dict, Any


def compute_score(
    responses: Dict[str, Any],
    scoring_yaml: Dict[str, Any]
) -> int:
    """
    Compute total score for an instrument.

    Parameters
    ----------
    responses : Dict[str, Any]
        Dictionary mapping question_id -> selected answer.
        For randomised MCQ, this may be a strict subset of
        the keys defined in correct_answers.

    scoring_yaml : Dict[str, Any]
        Parsed YAML scoring definition (one of Options A/B/C).

    Returns
    -------
    int
        Total computed score.

    Raises
    ------
    ValueError
        If scoring format is invalid or data integrity fails.
    """

    # ----------------------------------------------------------
    # Defensive: Ensure inputs are dictionaries
    # ----------------------------------------------------------

    if not isinstance(responses, dict):
        raise ValueError("`responses` must be a dictionary.")

    if not isinstance(scoring_yaml, dict):
        raise ValueError("`scoring_yaml` must be a dictionary.")

    total_score = 0

    # ----------------------------------------------------------
    # Detect scoring mode
    # ----------------------------------------------------------

    has_correct_answers = "correct_answers" in scoring_yaml
    has_points          = "points" in scoring_yaml
    has_likert          = scoring_yaml.get("scoring_type") == "likert"

    # Prevent ambiguous YAML configuration
    active_modes = sum([has_correct_answers, has_points, has_likert])
    if active_modes > 1:
        raise ValueError(
            "Scoring YAML must specify exactly one mode: "
            "'correct_answers', 'points', or 'scoring_type: likert'."
        )

    # ==========================================================
    # OPTION A: CORRECT ANSWERS (Binary — supports MCQ subsets)
    # ==========================================================

    if has_correct_answers:

        correct_answers = scoring_yaml["correct_answers"]

        if not isinstance(correct_answers, dict):
            raise ValueError("'correct_answers' must be a dictionary.")

        scoring_keys  = set(correct_answers.keys())
        response_keys = set(responses.keys())

        # Reject any response IDs that don't exist in the scoring key.
        # This catches schema drift / mis-labelled questions.
        unknown = response_keys - scoring_keys
        if unknown:
            raise ValueError(
                f"Response contains question IDs not in scoring key: "
                f"{sorted(unknown)}."
            )

        # Score only the questions that were answered (subset scoring).
        # Questions in the bank but not answered contribute 0.
        for question_id, correct_value in correct_answers.items():
            if question_id in responses:
                if responses[question_id] == correct_value:
                    total_score += 1

        return total_score

    # ==========================================================
    # OPTION B: POINT MAPPING (Weighted Scoring)
    # ==========================================================

    if has_points:

        points_map = scoring_yaml["points"]

        if not isinstance(points_map, dict):
            raise ValueError("'points' must be a dictionary.")

        scoring_keys  = set(points_map.keys())
        response_keys = set(responses.keys())

        # Same subset logic: reject unknown IDs, allow missing questions.
        unknown = response_keys - scoring_keys
        if unknown:
            raise ValueError(
                f"Response contains question IDs not in points map: "
                f"{sorted(unknown)}."
            )

        for question_id, selected_answer in responses.items():

            question_points = points_map[question_id]

            if not isinstance(question_points, dict):
                raise ValueError(
                    f"Points mapping for question '{question_id}' "
                    f"must be a dictionary."
                )

            answer_points = question_points.get(selected_answer)

            if answer_points is None:
                raise ValueError(
                    f"No scoring value for answer '{selected_answer}' "
                    f"in question '{question_id}'."
                )

            total_score += answer_points

        return total_score

    # ==========================================================
    # OPTION C: LIKERT (Default scale + optional reverse scoring)
    # ==========================================================

    if has_likert:

        default_scale     = scoring_yaml.get("default_scale", {})
        reverse_scale     = scoring_yaml.get("reverse_scale", {})
        reverse_questions = set(scoring_yaml.get("reverse_questions", []))

        if not isinstance(default_scale, dict):
            raise ValueError("'default_scale' must be a dictionary.")

        if not isinstance(reverse_scale, dict):
            raise ValueError("'reverse_scale' must be a dictionary.")

        for question_id, selected_answer in responses.items():

            # Choose scale
            scale = reverse_scale if question_id in reverse_questions \
                    else default_scale

            # Skip questions whose answer is not in any scale
            # (e.g. text-input initials like Q1_1).
            if selected_answer not in scale:
                continue

            total_score += scale[selected_answer]

        return total_score

    # ==========================================================
    # INVALID FORMAT
    # ==========================================================

    raise ValueError(
        "Scoring YAML must contain 'correct_answers', 'points', "
        "or 'scoring_type: likert'."
    )
