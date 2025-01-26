from .models.registry import get_model_class
from .models.base import ModelConfig


def model(name: str, **kwargs):
    try:
        model_class, config_class = get_model_class(name)
        config = config_class()
        config.update(**kwargs)
        return model_class(config)
    except ValueError as e:
        raise RuntimeError(f"Model creation failed: {str(e)}") from e

__all__ = ['model', 'ModelConfig']

