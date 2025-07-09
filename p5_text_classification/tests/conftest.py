"""
Configuration for pytest.

This file defines shared fixtures that can be used across multiple test files.
Fixtures defined here are automatically discovered by pytest.
"""
import pytest
import yaml
from pathlib import Path

@pytest.fixture(scope="session")
def test_config_dir(tmp_path_factory) -> Path:
    """
    Creates a temporary directory structure with a flexible set of YAML files 
    for robustly testing the ConfigManager.
    This fixture has a 'session' scope, meaning it's created once for the
    entire test run, improving efficiency.
    """
    config_dir = tmp_path_factory.mktemp("configs", numbered=False)
    
    # Create subdirectories
    (config_dir / "datasets").mkdir(parents=True)
    (config_dir / "models").mkdir(parents=True)
    (config_dir / "training").mkdir(parents=True)

    # --- Create YAML Files ---
    # Global defaults matching production structure
    global_defaults = {
        "logging": {"level": "INFO"},
        "device": {"preferred": "cuda"},
        "random": {"seed": 42},
        "validation": {"split_ratio": 0.2},
        "training": {"epochs": 1, "lr": 0.1, "momentum": 0.9}
    }
    with open(config_dir / "global_defaults.yaml", "w") as f:
        yaml.dump(global_defaults, f)

    # Dataset config, mimicking the nested structure of a real config
    dataset_conf = {
        "dataset": {
            "name": "test_dataset",
            "path": "placeholder/path", # This will be overridden in specific tests
            "train_file": "train.csv",
            "test_file": "test.csv",
            "class_file": "classes.txt",
            "sequence_length": 512,
            "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{} "
        },
        "training": { # Dataset-specific override
            "batch_size": 64
        }
    }
    with open(config_dir / "datasets" / "test_dataset.yaml", "w") as f:
        yaml.dump(dataset_conf, f)

    # Model config, defines model-specific params and overrides training params
    model_conf = {
        "model": {
            "num_classes": 10, 
            "depth": 9
        },
        "training": { # Model-specific override
            "lr": 0.01
        }
    }
    with open(config_dir / "models" / "vdcnn.yaml", "w") as f:
        yaml.dump(model_conf, f)
        
    # Another model for testing get_available_configs
    with open(config_dir / "models" / "densenet.yaml", "w") as f:
        yaml.dump({"model": {"num_classes": 15}}, f)

    # Training config, overrides lr to its own value
    training_conf = {
        "training": {
            "lr": 0.005, 
            "epochs": 20
        }
    }
    with open(config_dir / "training" / "test_training.yaml", "w") as f:
        yaml.dump(training_conf, f)
        
    return config_dir 