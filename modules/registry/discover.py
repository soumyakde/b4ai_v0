"""
Module Discovery System
------------------------

Discovers and validates curriculum modules located in:

    modules/definitions/

Only files ending with:

    *_definition.py

are treated as valid module definition files.

Each definition file MUST expose:

    MODULE_DEFINITION (dict)

Each MODULE_DEFINITION MUST contain:

    {
        "meta": {
            "module_id": "<unique_id>",
            ...
        },
        ...
    }

This ensures:
    ✅ Strict validation
    ✅ Deterministic loading
    ✅ No silent overrides
    ✅ Clean plugin-style architecture
"""

import importlib
import pkgutil
from typing import List, Dict, Any

import modules.definitions


def discover_module_definitions(include_inactive: bool = False) -> List[Dict[str, Any]]:
    """
    Discover and validate all module definitions.

    Args:
        include_inactive:
            False (default) — skip modules whose meta.status is not
            "active". This is what the student-facing registry
            (module_registry, unlock sequencing) must use.

            True — return every module definition found on disk
            regardless of status. Intended for callers that need the
            FULL module inventory (e.g. Teacher Dashboard / analytics),
            which should keep showing paused/disabled modules rather
            than only what's currently visible to students.

    Returns:
        A list of validated MODULE_DEFINITION dictionaries.

    Raises:
        ValueError if:
            - MODULE_DEFINITION is missing
            - MODULE_DEFINITION is not a dict
            - 'meta.module_id' is missing
            - Duplicate module IDs are found
    """

    discovered_modules: List[Dict[str, Any]] = []
    seen_module_ids = set()

    package = modules.definitions

    # Iterate through all Python files in modules/definitions/
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):

        # ✅ Only load files that follow the naming convention
        # Example: module1_definition.py
        if not module_name.endswith("_definition"):
            continue

        full_module_name = f"{package.__name__}.{module_name}"

        # Dynamically import module
        module = importlib.import_module(full_module_name)

        # ------------------------------------------------------------------
        # Validate existence of MODULE_DEFINITION
        # ------------------------------------------------------------------
        if not hasattr(module, "MODULE_DEFINITION"):
            raise ValueError(
                f"{full_module_name} does not define MODULE_DEFINITION"
            )

        module_definition = module.MODULE_DEFINITION

        # ------------------------------------------------------------------
        # Validate type
        # ------------------------------------------------------------------
        if not isinstance(module_definition, dict):
            raise ValueError(
                f"{full_module_name}.MODULE_DEFINITION must be a dictionary"
            )

        # ------------------------------------------------------------------
        # Validate meta section
        # ------------------------------------------------------------------
        meta = module_definition.get("meta")

        if not isinstance(meta, dict):
            raise ValueError(
                f"{full_module_name} must define a 'meta' dictionary"
            )

        module_id = meta.get("module_id")

        if not module_id:
            raise ValueError(
                f"{full_module_name} missing required key: meta['module_id']"
            )

        # ------------------------------------------------------------------
        # Skip modules explicitly marked inactive
        # ------------------------------------------------------------------
        status = meta.get("status", "active")

        if status != "active" and not include_inactive:
            print(f"[ModuleRegistry] ⏸️  Skipped module '{module_id}' (status='{status}')")
            continue

        # ------------------------------------------------------------------
        # Enforce unique module IDs
        # ------------------------------------------------------------------
        if module_id in seen_module_ids:
            raise ValueError(
                f"Duplicate module_id detected: '{module_id}'"
            )

        seen_module_ids.add(module_id)
        discovered_modules.append(module_definition)

        print(f"[ModuleRegistry] ✅ Registered module: {module_id}")

    # ----------------------------------------------------------------------
    # Ensure at least one module was found
    # ----------------------------------------------------------------------
    if not discovered_modules:
        raise ValueError(
            "No valid module definitions were discovered in modules/definitions/"
        )

    return discovered_modules


def discover_all_module_numbers() -> List[int]:
    """
    Return every numbered module (1..N) with a definition file on disk,
    regardless of active/disabled status.

    Intended for consumers that must see the FULL module inventory even
    when some modules are temporarily paused — e.g. the Teacher Dashboard
    and the analytics pipeline (core/analytics/datasets/canonical_loader.py),
    which should keep reflecting a disabled module's existing data rather
    than only what's currently visible to students. Replaces what used to
    be a hardcoded range(1, 8) in both of those files, so adding module 8+
    (or removing/re-enabling one) requires no further code changes there.

    pre_course / post_course are intentionally excluded — callers that need
    those already reference them by fixed name, not by number.
    """
    numbers = []
    for definition in discover_module_definitions(include_inactive=True):
        module_id = definition["meta"]["module_id"]
        if module_id.startswith("module_"):
            try:
                numbers.append(int(module_id.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(numbers)