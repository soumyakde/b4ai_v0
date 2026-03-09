##modules/registry/module_registry.py
from typing import Dict, Any, List

class ModuleRegistry:
    """
    Central registry for all learning unit definitions.

    IMPORTANT DESIGN NOTE:
    ----------------------
    Streamlit reruns the entire script on user interactions
    (login, logout, button clicks, etc.).

    Therefore:
    - Registration MUST be idempotent
    - Duplicate registrations must NOT crash the app
    - The registry must behave like a singleton

    This implementation guarantees safe repeated calls to register().
    """

    def __init__(self):
        # Internal storage:
        # module_id -> module_definition
        self._modules: Dict[str, Dict[str, Any]] = {}

        # Preserves curricular order of registration
        self._ordered_ids: List[str] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, module_definition: Dict[str, Any]) -> None:
        """
        Register a validated learning unit definition.

        Safe for repeated calls (Streamlit reruns).

        Behavior:
        - First registration → stored
        - Duplicate registration → silently skipped
        """

        meta = module_definition.get("meta", {})
        module_id = meta.get("module_id")

        if not module_id:
            raise ValueError("module_definition missing meta.module_id")

        # ✅ FIX: Make registration idempotent (do NOT crash on duplicates)
        if module_id in self._modules:
            # Silently skip duplicate registration
            # (expected during Streamlit reruns)
            return

        self._modules[module_id] = module_definition
        self._ordered_ids.append(module_id)

    

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, module_id: str) -> Dict[str, Any]:
        """
        Retrieve a learning unit definition by ID.
        """
        if module_id not in self._modules:
            raise KeyError(f"Module not found: {module_id}")

        return self._modules[module_id]

    def has(self, module_id: str) -> bool:
        """
        Check if a module is registered.
        """
        return module_id in self._modules

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_modules(self) -> Dict[str, Dict[str, Any]]:
        """
        Return all registered learning units (unordered dictionary).
        """
        return dict(self._modules)

    def list_learning_units(self) -> List[Dict[str, Any]]:
        """
        Return learning units in curricular registration order.
        """
        return [self._modules[mid] for mid in self._ordered_ids]

    def get_ordered_modules(self) -> List[Dict[str, Any]]:
        """
        Return modules sorted by display order (meta.order).

        Why this exists:
        ----------------
        The student dashboard must display modules in a logical
        curricular sequence:

            Pre-Course → Module 1 → Module 2 → ...

        Each module may define:

            module_dict["meta"]["order"]

        Behavior:
        ---------
        - Sort ascending by meta.order
        - If order is missing → defaults to 999
        - Safe even if meta is missing
        - Returns a LIST of module definitions
        """

        if not self._modules:
            return []

        return sorted(
            self._modules.values(),
            key=lambda module: module.get("meta", {}).get("order", 999)
        )

    # ------------------------------------------------------------------
    # Survey Accessor (Reflection Fix)
    # ------------------------------------------------------------------

    def get_survey(self, module_id: str, survey_key: str) -> Dict[str, Any]:
        """
        Retrieve a survey for a module.

        survey_key may be either:
        - registry key (e.g. 'reflection')
        - internal survey_key (e.g. 'module1_reflection')
        """

        module = self.get(module_id)

        instruments = module.get("instruments", {})
        surveys = instruments.get("surveys", {})

        # 1️⃣ Direct registry-key lookup (preferred)
        if survey_key in surveys:
            return surveys[survey_key]

        # 2️⃣ Fallback: match by internal survey_key
        for survey in surveys.values():
            if survey.get("survey_key") == survey_key:
                return survey

        # 3️⃣ Clear error for debugging
        raise KeyError(
            f"Survey '{survey_key}' not found for module '{module_id}'. "
            f"Available surveys: {list(surveys.keys())}"
        )


# ✅ Global singleton registry (safe for Streamlit reruns)
module_registry = ModuleRegistry()

__all__ = ["module_registry"]