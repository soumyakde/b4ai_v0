"""
learning_unit_resolver.py

Purpose:
--------
Read-only, deterministic resolver for registered learning units.

This class:
- Does NOT mutate registry
- Does NOT contain progress logic
- Does NOT contain unlock logic
- Does NOT interact with database
- Does NOT import Streamlit

It only:
- Retrieves units
- Guarantees curricular ordering
- Resolves next/previous units safely

This ensures:
- Deterministic navigation
- Stable unlock sequencing foundation
- Clean architectural separation
"""

from typing import Dict, Any, Optional, List
from modules.registry.module_registry import module_registry


class LearningUnitResolver:
    """
    Read-only resolver for registered learning units.

    The resolver acts as a stability layer between:
        - module_registry (source of truth)
        - progress/unlock engines
        - UI navigation

    It enforces deterministic ordering and safe access.
    """

    def __init__(self, registry=module_registry):
        """
        Allow registry injection for:
        - Testing
        - Future extensibility
        """
        self._registry = registry

    # ---------------------------------------------------------------------
    # INTERNAL: Deterministic Ordering
    # ---------------------------------------------------------------------

    def _ordered_units(self) -> List[Dict[str, Any]]:
        """
        Return learning units sorted by meta.order.

        Why this exists:
        ----------------
        We do NOT assume registry guarantees order.
        Even if registry currently sorts internally,
        this defensive sort guarantees deterministic behavior.

        This is critical because:
        - Unlock logic depends on order
        - next_unit / previous_unit depend on order
        - Progress sequencing depends on order

        Fail-fast behavior:
        -------------------
        If a module is malformed (missing meta or order),
        we raise a clear KeyError immediately.
        """

        units = self._registry.list_learning_units()

        try:
            return sorted(units, key=lambda u: u["meta"]["order"])
        except KeyError as e:
            raise KeyError(
                f"Malformed learning unit missing required meta key: {e}"
            )

    # ---------------------------------------------------------------------
    # PUBLIC: Basic Accessors
    # ---------------------------------------------------------------------

    def get(self, module_id: str) -> Dict[str, Any]:
        """
        Resolve a learning unit by module_id.

        Raises:
            KeyError if module_id not found.
        """
        return self._registry.get(module_id)

    def exists(self, module_id: str) -> bool:
        """
        Check whether a learning unit exists.
        """
        return self._registry.has(module_id)

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Return all learning units in deterministic curricular order.
        """
        return self._ordered_units()

    # ---------------------------------------------------------------------
    # PUBLIC: Sequential Navigation
    # ---------------------------------------------------------------------

    def next_unit(self, module_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the next learning unit in curricular order.

        Returns:
            - Dict for next unit
            - None if this is the last unit

        Raises:
            KeyError if module_id is invalid
        """

        units = self._ordered_units()
        ids = [u["meta"]["module_id"] for u in units]

        if module_id not in ids:
            raise KeyError(f"Unknown module_id: {module_id}")

        idx = ids.index(module_id)

        if idx + 1 < len(units):
            return units[idx + 1]

        return None  # Last unit

    def previous_unit(self, module_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the previous learning unit in curricular order.

        Returns:
            - Dict for previous unit
            - None if this is the first unit

        Raises:
            KeyError if module_id is invalid
        """

        units = self._ordered_units()
        ids = [u["meta"]["module_id"] for u in units]

        if module_id not in ids:
            raise KeyError(f"Unknown module_id: {module_id}")

        idx = ids.index(module_id)

        if idx > 0:
            return units[idx - 1]

        return None  # First unit