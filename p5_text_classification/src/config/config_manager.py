"""
Configuration Manager implementation.
Implements Single Source of Truth (SSOT) principle and follows SOLID principles.
"""

import yaml
from typing import Dict, Any, Optional, MutableMapping, List
from pathlib import Path
from copy import deepcopy

from .base_config import (
    AppConfig, DataConfig, TrainingConfig, LoggingConfig, 
    DeviceConfig, RandomConfig, ValidationConfig, ModelConfig
)
from ..utils.constants import CONFIG_DIR

def deep_merge(d1: MutableMapping, d2: MutableMapping) -> MutableMapping:
    """
    Recursively merges two dictionaries. d2 values override d1 values.
    """
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            d1[k] = deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1

class ConfigManager:
    """
    Manages loading, merging, and validation of application configurations.
    It constructs a type-safe AppConfig object using a centralized CONFIG_DIR.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initializes the ConfigManager.
        
        Args:
            config_dir (Optional[Path]): The path to the configuration directory.
                                         If None, defaults to the project's standard
                                         CONFIG_DIR. This allows for testing with
                                         different configurations.
        """
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
        
    def load_config(self, model_name: str, dataset_name: str, training_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
        """
        Loads configurations from multiple YAML files, merges them with a specific
        priority, and returns a validated AppConfig object.
        
        Priority order (lowest to highest):
        1. global_defaults.yaml
        2. dataset config
        3. model config
        4. training config (optional)
        5. overrides (e.g., from command line)
        """
        # 1. Load base configurations
        global_config = self._load_yaml_file(self.config_dir / "global_defaults.yaml")
        dataset_config = self._load_yaml_file(self.config_dir / "datasets" / f"{dataset_name}.yaml")
        model_config = self._load_yaml_file(self.config_dir / "models" / f"{model_name}.yaml")
        
        # 2. Start with the lowest priority config
        merged_config = global_config
        
        # 3. Sequentially merge higher priority configs
        deep_merge(merged_config, dataset_config)
        deep_merge(merged_config, model_config)
        
        # 4. Merge optional training config if provided
        if training_name:
            training_config = self._load_yaml_file(self.config_dir / "training" / f"{training_name}.yaml")
            deep_merge(merged_config, training_config)

        # 5. Apply highest-priority overrides
        if overrides:
            deep_merge(merged_config, overrides)
            
        return self._create_app_config_from_dict(merged_config, model_name)

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Loads a single YAML file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _create_app_config_from_dict(self, config_dict: Dict[str, Any], model_name: str) -> AppConfig:
        """
        Intelligently creates an AppConfig instance from a merged dictionary.
        """
        # The dictionary is now flat, so we can directly initialize the dataclasses.
        data_conf = DataConfig(**config_dict.get('data', {}))
        model_conf = ModelConfig(**config_dict.get('model', {}))
        train_conf = TrainingConfig(**config_dict.get('training', {}))
        device_conf = DeviceConfig(**config_dict.get('device', {}))
        val_conf = ValidationConfig(**config_dict.get('validation', {}))
        log_conf = LoggingConfig(**config_dict.get('logging', {}))
        random_conf = RandomConfig(**config_dict.get('random', {}))

        return AppConfig(
            data=data_conf,
            model=model_conf,
            training=train_conf,
            device=device_conf,
            validation=val_conf,
            logging=log_conf,
            random=random_conf
        )
    
    # This method is now invalid due to dataclass changes and is not currently used.
    # def save_config...
    # It is removed to prevent future confusion.

    def get_available_configs(self) -> Dict[str, List[str]]:
        """
        Get list of available configuration files.
        
        Returns:
            Dict[str, List[str]]: Available configurations by category
        """
        result = {
            'models': [],
            'datasets': []
        }
        
        # Get available models
        models_dir = self.config_dir / "models"
        if models_dir.exists():
            result['models'] = [f.stem for f in models_dir.glob("*.yaml")]
            
        # Get available datasets
        datasets_dir = self.config_dir / "datasets"
        if datasets_dir.exists():
            result['datasets'] = [f.stem for f in datasets_dir.glob("*.yaml")]
            
        return result 