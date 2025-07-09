"""
Integration tests for the DataManager.

These tests verify that the DataManager can correctly interact with the
configuration system and the filesystem to load, preprocess, and batch
data into DataLoaders.
"""
import pytest
import pandas as pd
from pathlib import Path
import torch

from torch.utils.data import DataLoader

from p5_text_classification.src.config.config_manager import ConfigManager
from p5_text_classification.src.config.base_config import AppConfig
from p5_text_classification.src.data.data_manager import DataManager

@pytest.fixture(scope="module")
def temp_data_dir(tmpdir_factory) -> Path:
    """Creates a temporary directory with a dummy train.csv file."""
    temp_dir = Path(tmpdir_factory.mktemp("data"))
    train_data = {
        'label': list(range(1, 101)),
        'text': [f"This is test sentence number {i}" for i in range(100)]
    }
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(temp_dir / "train.csv", index=False, header=False)
    
    # Create an empty test.csv as it's expected by the config
    (temp_dir / "test.csv").touch()
    
    return temp_dir

@pytest.fixture(scope="module")
def test_app_config(test_config_dir, temp_data_dir: Path) -> AppConfig:
    """
    Provides a fully initialized AppConfig object pointing to the
    temporary data directory.
    """
    manager = ConfigManager(config_dir=test_config_dir)
    # Override the data path to point to our temporary directory
    # Note: str(temp_data_dir) is important for YAML compatibility if we were writing it,
    # but for overrides, Path object works fine with deep_merge. However, our
    # current implementation reads it from yaml, so we modify the yaml loading logic instead.
    # The better way is to create a specific test config file.
    # For now, let's use overrides.
    
    # We need to use relative path for the config loader to find it from project root
    # but the path in config is relative to project root
    # So we can just pass the path string
    
    # Using an absolute path in the override will work because Path()
    # joining a path with an absolute path results in the absolute path.
    # Path('any/path') / Path('C:/abs/path') -> Path('C:/abs/path')
    abs_path_str = str(temp_data_dir.resolve())

    overrides = {
        "dataset": {
            "path": abs_path_str,
            "train_file": "train.csv",
            "test_file": "test.csv"
        }
    }
    
    config = manager.load_config(
        model_name="vdcnn",
        dataset_name="test_dataset",
        overrides=overrides
    )
    return config

def test_get_data_loaders_successful(test_app_config: AppConfig):
    """
    Tests the main success path of get_data_loaders.
    Verifies loader types, batch contents, and dataset size.
    """
    # Arrange
    dm = DataManager(config=test_app_config)
    
    # Act
    train_loader, val_loader = dm.get_data_loaders()
    
    # Assert
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # Check sizes based on 100 samples and 0.2 validation split
    # 80 train, 20 val
    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 20
    
    # Check a batch
    train_batch_data, train_batch_labels = next(iter(train_loader))
    
    assert train_batch_data.dtype == torch.long
    assert train_batch_labels.dtype == torch.long
    assert train_batch_data.shape == (test_app_config.training.batch_size, test_app_config.data.sequence_length)
    assert train_batch_labels.shape == (test_app_config.training.batch_size,)

def test_max_samples_limit(test_app_config: AppConfig):
    """Tests that the max_samples configuration is respected."""
    # Arrange
    # Modify the config to set a max_samples limit
    test_app_config.training.max_samples = 10
    
    dm = DataManager(config=test_app_config)
    
    # Act
    train_loader, val_loader = dm.get_data_loaders()
    
    # Assert
    # 10 samples total, 0.2 split -> 8 train, 2 val
    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 2
    assert len(train_loader.dataset) + len(val_loader.dataset) == 10 