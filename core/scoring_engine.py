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

OPTION A (Correct Answers)
---------------------------
correct_answers:
  Q1: "A"
  Q2: "B"

OPTION B (Point Mapping)
---------------------------
points:
  Q1:
    A: 1
    B: 0
  Q2:
    Yes: 2
    No: 0
"""

from typing import Dict, Any


def compute_score(
    responses: Dict[str, Any],
    scoring_yaml: Dict[str, Any]
) -> int:
    """
    Compute total score for assessment.

    Parameters
    ----------
    responses : Dict[str, Any]
        Dictionary mapping question_id -> selected answer

    scoring_yaml : Dict[str, Any]
        Parsed YAML scoring definition

    Returns
    -------
    int
        Total computed score

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
    has_points = "points" in scoring_yaml

    # Prevent ambiguous YAML configuration
    if has_correct_answers and has_points:
        raise ValueError(
            "Scoring YAML cannot contain both "
            "'correct_answers' and 'points'."
        )

    # ==========================================================
    # OPTION A: CORRECT ANSWERS (Binary Scoring)
    # ==========================================================

    if has_correct_answers:

        correct_answers = scoring_yaml["correct_answers"]

        if not isinstance(correct_answers, dict):
            raise ValueError("'correct_answers' must be a dictionary.")

        # ------------------------------------------------------
        # Strict key validation:
        # Ensure responses EXACTLY match scoring definition
        # Prevents:
        # - Missing responses
        # - Extra unexpected questions
        # - Schema drift
        # ------------------------------------------------------

        response_keys = set(responses.keys())
        scoring_keys = set(correct_answers.keys())

        if response_keys != scoring_keys:
            raise ValueError(
                f"Response keys {response_keys} do not match "
                f"scoring keys {scoring_keys}."
            )

        # ------------------------------------------------------
        # Compute score
        # ------------------------------------------------------

        for question_id, correct_value in correct_answers.items():

            # Compare values exactly (deterministic)
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

        # ------------------------------------------------------
        # Strict key validation (same reasoning as above)
        # ------------------------------------------------------

        response_keys = set(responses.keys())
        scoring_keys = set(points_map.keys())

        if response_keys != scoring_keys:
            raise ValueError(
                f"Response keys {response_keys} do not match "
                f"scoring keys {scoring_keys}."
            )

        # ------------------------------------------------------
        # Compute score
        # ------------------------------------------------------

        for question_id, selected_answer in responses.items():

            # Ensure question has scoring rule
            if question_id not in points_map:
                raise ValueError(
                    f"No scoring rule for question '{question_id}'."
                )

            question_points = points_map[question_id]

            if not isinstance(question_points, dict):
                raise ValueError(
                    f"Points mapping for question '{question_id}' "
                    f"must be a dictionary."
                )

            # Lookup selected answer value
            answer_points = question_points.get(selected_answer)

            if answer_points is None:
                raise ValueError(
                    f"No scoring value for answer '{selected_answer}' "
                    f"in question '{question_id}'."
                )

            total_score += answer_points

        return total_score

    # ==========================================================
    # INVALID FORMAT
    # ==========================================================

    raise ValueError(
        "Scoring YAML must contain either "
        "'correct_answers' or 'points'."
    )