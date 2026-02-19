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


def discover_module_definitions() -> List[Dict[str, Any]]:
    """
    Discover and validate all module definitions.

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