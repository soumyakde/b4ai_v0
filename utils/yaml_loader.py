"""
YAML Loader Utility

Responsibilities:
- Load YAML files safely
- Fail fast on missing files
- Fail fast on malformed YAML
- Return pure dict (no side effects)

This module has:
- No DB access
- No Streamlit
- No resolver logic
- No scoring logic
"""

import os
import yaml
from typing import Dict, Any


BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "streamlit_app"
)


def load_yaml(relative_path: str) -> Dict[str, Any]:
    """
    Load a YAML file relative to project root.

    Example:
        load_yaml("surveys/demographics_survey.yaml")

    Raises:
        FileNotFoundError
        ValueError (if YAML malformed)
    """

    full_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"YAML file not found: {full_path}")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {relative_path}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dictionary: {relative_path}")

    return data