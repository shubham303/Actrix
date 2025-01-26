from dataclasses import dataclass, fields
from typing import Type, Dict, Any
import torch.nn as nn

@dataclass
class ModelConfig:
    name: str
    num_classes: int = 1000
    img_size: tuple = (224, 224)

    def update(self, **kwargs):
        valid_fields = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key not in valid_fields:
                raise ValueError(f"Invalid config parameter '{key}'")
            setattr(self, key, value)

class Model():
    def __init__(self, config: ModelConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(config)
    
