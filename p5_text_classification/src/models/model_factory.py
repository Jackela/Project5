"""
Model Factory

A simple factory function to create model instances based on the application's
configuration object. This approach centralizes model creation logic and
integrates seamlessly with the typed configuration system.
"""

import torch.nn as nn
from typing import Dict, Any

from .densenet import DenseNet
from .vdcnn_model import VDCNN
from ..config.base_config import ModelConfig

SUPPORTED_MODELS = {
    "vdcnn": VDCNN,
    "densenet": DenseNet,
}

def create_model(model_config: ModelConfig) -> nn.Module:
    """
    Factory function to create a model instance based on a ModelConfig object.

    Args:
        model_config: A dataclass object containing the model's name and
                      all necessary parameters for its construction.

    Returns:
        An instance of the requested model (nn.Module).
        
    Raises:
        ValueError: If the requested model name is not supported.
    """
    model_name = model_config.name.lower()
    params = model_config.params

    if model_name == 'vdcnn':
        return VDCNN(**params)
    
    elif model_name == 'densenet':
        return DenseNet(**params)
        
    else:
        raise ValueError(f"Unsupported model type: '{model_name}'. Supported types are 'vdcnn', 'densenet'.") 