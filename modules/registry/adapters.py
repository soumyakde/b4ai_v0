"""
adapters.py

Purpose:
--------
Transforms raw MODULE_DEFINITION dictionaries (pure declarative metadata)
into normalized module objects that the registry and dashboards can safely use.

This adapter:
    ✅ Normalizes metadata
    ✅ Applies defensive defaults
    ✅ Injects runtime render() function
    ✅ Enforces v6 execution contract
"""

from copy import deepcopy
import importlib


def adapt_module_definition(defn: dict) -> dict:
    """
    Convert a raw MODULE_DEFINITION into a normalized registry-safe structure.
    """

    # ------------------------------------------------------------------
    # Defensive copy (prevents mutation of original definition)
    # ------------------------------------------------------------------
    defn = deepcopy(defn)

    # ------------------------------------------------------------------
    # Extract meta safely
    # ------------------------------------------------------------------
    meta = defn.get("meta", {})

    if "module_id" not in meta:
        raise ValueError("MODULE_DEFINITION is missing required field: meta.module_id")

    module_id = meta["module_id"]

    # ------------------------------------------------------------------
    # Normalize metadata
    # ------------------------------------------------------------------
    normalized_meta = {
        "module_id": module_id,
        "title": meta.get("title") or meta.get("module_name"),
        "version": meta.get("version", "1.0.0"),
        "description": meta.get("description", ""),
        "author": meta.get("author", ""),
        "status": meta.get("status", "active"),
        "order": meta.get("order", 999),
    }

    if not normalized_meta["title"]:
        raise ValueError(
            f"Module '{module_id}' must define either 'title' or 'module_name'"
        )

    # ------------------------------------------------------------------
    # Normalize Pedagogy
    # ------------------------------------------------------------------
    pedagogy = defn.get("pedagogy", {})

    normalized_pedagogy = {
        "learning_objectives": pedagogy.get("learning_objectives", []),
        "prerequisite_knowledge": pedagogy.get("prerequisite_knowledge", ""),
        "estimated_time_minutes": pedagogy.get("estimated_time_minutes", None),
    }

    # ------------------------------------------------------------------
    # Normalize Instruments
    # ------------------------------------------------------------------
    instruments = defn.get("instruments", {})

    normalized_instruments = {
        "surveys": instruments.get("surveys", {}),
        "assessments": instruments.get("assessments", {}),
        "labs": instruments.get("labs", {}),
    }

    # ------------------------------------------------------------------
    # Normalize UI Block
    # ------------------------------------------------------------------
    ui = defn.get("ui", {})

    normalized_ui = {
        "show_progress": ui.get("show_progress", True),
    }

    # ------------------------------------------------------------------
    # Legacy Support
    # ------------------------------------------------------------------
    legacy = defn.get("legacy", {})

    # ------------------------------------------------------------------
    # ✅ Inject Runtime Render Function (v6 Contract Enforcement)
    # ------------------------------------------------------------------
    try:
        runtime_module = importlib.import_module(f"modules.{module_id}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Runtime module 'modules.{module_id}' not found for module_id '{module_id}'."
        ) from e

    render_fn = getattr(runtime_module, "render", None)

    if not callable(render_fn):
        raise ValueError(
            f"Module '{module_id}' does not define a callable render(username) function."
        )

    # ------------------------------------------------------------------
    # Final Normalized Module Object
    # ------------------------------------------------------------------
    adapted_module = {
        "meta": normalized_meta,
        "pedagogy": normalized_pedagogy,
        "instruments": normalized_instruments,
        "ui": normalized_ui,
        "legacy": legacy,
        "render": render_fn,  # ✅ Injected executable entry point
    }

    return adapted_module