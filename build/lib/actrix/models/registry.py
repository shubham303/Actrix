from typing import Dict, Type, Tuple
from dataclasses import fields

_registry: Dict[str, Tuple[Type, Type]] = {}

# Modified registry to handle multiple configs
_registry = {}

def register_model(config_class):
    def decorator(model_class):
        # Use config name as registry key
        _registry[config_class().name] = (model_class, config_class)
        return model_class
    return decorator

def model(name: str, **kwargs):
    model_class, config_class = _registry[name]
    config = config_class()
    config.__dict__.update(kwargs)  # Apply custom parameters
    return model_class(config)


def get_model_class(name: str) -> Tuple[Type, Type]:
    if name not in _registry:
        raise ValueError(f"Model '{name}' not found")
    return _registry[name]

def list_models() -> list:
    return list(_registry.keys())

def get_model_params(name: str) -> dict:
    """
    Retrieves the parameters and their metadata for a given model configuration.

    Args:
        name (str): Name of the model configuration.

    Returns:
        dict: A dictionary containing parameter names as keys and their metadata (type and help) as values.
    """
    _, config_cls = get_model_class(name)
    params = {}
    for f in fields(config_cls):
        param_info = {
            "type": str(f.type),
            "help": f.metadata.get("help", "No description available.")
        }
        params[f.name] = param_info
    return params
