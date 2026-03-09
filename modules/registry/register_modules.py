#modules/registry/register_modules.py
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))



from modules.registry.module_registry import module_registry
from modules.registry.adapters import adapt_module_definition
from modules.registry.discover import discover_module_definitions


def register_all_modules():
    """
    Discovers, adapts, and registers all modules.
    """

    for raw_definition in discover_module_definitions():
        adapted = adapt_module_definition(raw_definition)
        print("Registering:", adapted["meta"]["module_id"])
        module_registry.register(adapted)