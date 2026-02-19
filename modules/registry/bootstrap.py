from modules.registry.module_registry import module_registry
from modules.registry.register_modules import register_all_modules


def build_registry():
    """
    Builds and returns the singleton registry.
    Safe to call multiple times.
    """

    if getattr(module_registry, "_initialized", False):
        return module_registry

    register_all_modules()
    module_registry._initialized = True
    return module_registry