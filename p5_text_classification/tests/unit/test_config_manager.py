import pytest
from pathlib import Path
from p5_text_classification.src.config.config_manager import ConfigManager
from p5_text_classification.src.config.base_config import AppConfig
import yaml

@pytest.fixture
def test_config_dir(tmp_path):
    """
    Creates a temporary directory with a complete set of mock configuration files
    for robust testing of the ConfigManager.
    """
    configs_path = tmp_path / "configs"
    models_path = configs_path / "models"
    datasets_path = configs_path / "datasets"
    training_path = configs_path / "training"
    
    models_path.mkdir(parents=True)
    datasets_path.mkdir(parents=True)
    training_path.mkdir(parents=True)

    # Global Defaults
    with open(configs_path / "global_defaults.yaml", "w") as f:
        yaml.dump({
            "training": {"lr": 0.01, "batch_size": 128, "epochs": 10},
            "data": {"alphabet": "abc", "sequence_length": 1014},
            "device": {"preferred": "cpu"},
             "validation": {"split_ratio": 0.2},
             "logging": {"level": "INFO"}
        }, f)

    # Models
    with open(models_path / "vdcnn.yaml", "w") as f:
        yaml.dump({"model": {"name": "vdcnn", "params": {"depth": 9}}}, f)
    with open(models_path / "densenet.yaml", "w") as f:
        yaml.dump({"model": {"name": "densenet", "params": {"growth_rate": 12}}}, f)

    # Datasets - Ensure the test dataset is complete
    with open(datasets_path / "test_dataset.yaml", "w") as f:
        yaml.dump({
            "data": {
                "name": "test_dataset",
                "path": "data/test",
                "train_file": "train.csv",
                "test_file": "test.csv",
                "class_file": "classes.txt"
                # sequence_length and alphabet will be picked up from global_defaults
            }
        }, f)

    # Training
    with open(training_path / "test_training.yaml", "w") as f:
        yaml.dump({"training": {"lr": 0.005}}, f)
        
    return configs_path

def test_load_config_successfully(test_config_dir):
    """
    Tests that the ConfigManager can successfully load and merge configurations
    and create a valid AppConfig object. This test will initially fail due
    to the known bug, driving the fix.
    """
    # Arrange
    manager = ConfigManager(config_dir=test_config_dir)
    
    # Act
    # This is the action that is expected to fail initially.
    config = manager.load_config(
        model_name="vdcnn",
        dataset_name="test_dataset"
    )
    
    # Assert
    assert isinstance(config, AppConfig)
    assert config.model.name == "vdcnn"
    # LR should come from model config (0.01), overriding dataset and global
    assert config.training.lr == 0.01
    # Batch size should come from dataset config's training block (64)
    assert config.training.batch_size == 64
    # Check a value from the dataset block
    assert config.data.sequence_length == 512
    # Check that model-specific params were loaded into the params dict
    assert 'num_classes' in config.model.params
    assert config.model.params['num_classes'] == 10
    assert config.model.params['depth'] == 9

def test_config_merge_priority(test_config_dir):
    """
    Tests that the configuration merge priority is correctly handled.
    Priority: overrides > training > model > dataset > global
    """
    # Arrange
    manager = ConfigManager(config_dir=test_config_dir)
    overrides = {
        "training": {"lr": 0.0001},  # Highest priority
    }
    
    # Act
    config = manager.load_config(
        model_name="vdcnn",
        dataset_name="test_dataset",
        training_name="test_training",
        overrides=overrides
    )
    
    # Assert
    assert isinstance(config, AppConfig)
    # Learning rate should be from `overrides`
    assert config.training.lr == 0.0001
    # Batch size should be from `training` config (test_training.yaml doesn't define it, so it falls back to dataset)
    assert config.training.batch_size == 64
    # Epochs should be from `training` config, as it's not in overrides
    assert config.training.epochs == 20
    # Num classes should be from `model` config
    assert config.model.params['num_classes'] == 10

def test_load_config_raises_file_not_found(test_config_dir):
    """
    Tests that FileNotFoundError is raised for missing config files.
    """
    manager = ConfigManager(config_dir=test_config_dir)
    
    # Test for missing model file
    with pytest.raises(FileNotFoundError, match="models"):
        manager.load_config(model_name="non_existent_model", dataset_name="test_dataset")
        
    # Test for missing dataset file
    with pytest.raises(FileNotFoundError, match="datasets"):
        manager.load_config(model_name="vdcnn", dataset_name="non_existent_dataset")

def test_get_available_configs(test_config_dir):
    """
    Tests that the get_available_configs method correctly finds and lists
    all available model and dataset configuration files.
    """
    # Arrange
    manager = ConfigManager(config_dir=test_config_dir)
    
    # Act
    available_configs = manager.get_available_configs()
    
    # Assert
    assert isinstance(available_configs, dict)
    assert "models" in available_configs
    assert "datasets" in available_configs
    # The list might not be in a specific order, so use set for comparison
    assert set(available_configs['models']) == {"vdcnn", "densenet"}
    assert set(available_configs['datasets']) == {"test_dataset"} 