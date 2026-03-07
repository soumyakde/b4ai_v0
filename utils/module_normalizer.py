def normalize_module_id(module_id: str) -> str:
    """
    Converts dataset module IDs into ontology-compatible keys.

    module1  -> module_1
    module2  -> module_2
    global   -> global
    """

    if module_id.startswith("module") and "_" not in module_id:
        number = module_id.replace("module", "")
        return f"module_{number}"

    return module_id