"""
Integration test for the ExperimentRunner.

This test verifies that all major components of the application can be
successfully integrated and can run a full end-to-end training experiment.
"""
import pytest
from unittest.mock import MagicMock, patch

import torch
from pathlib import Path

from p5_text_classification.src.config.config_manager import ConfigManager
from p5_text_classification.src.data.data_manager import DataManager
from p5_text_classification.src.models.model_factory import create_model
from p5_text_classification.src.training.experiment_runner import ExperimentRunner


@pytest.fixture(scope="module")
def full_test_setup(test_config_dir, tmpdir_factory):
    """
    Sets up a complete environment for an end-to-end integration test.
    This includes loading configuration, creating temporary data,
    and preparing data loaders.
    """
    # 1. Load Config
    # We create a specific test config file to avoid complex overrides
    data_dir = Path(tmpdir_factory.mktemp("data"))
    
    # Create dummy data files
    # 20 samples, batch size 4 -> 5 batches for train, val split will also work
    with open(data_dir / "train.csv", "w") as f:
        for i in range(20):
            f.write(f"{i % 2 + 1},'some text {i}'\n") # 2 classes
    with open(data_dir / "test.csv", "w") as f:
        f.write("1,'test text'\n")
    with open(data_dir / "classes.txt", "w") as f:
        f.write("class1\nclass2\n")

    # Use a specific config for this integration test
    manager = ConfigManager(config_dir=test_config_dir)
    overrides = {"dataset": {"path": str(data_dir.resolve())}, "training": {"epochs": 1, "batch_size": 4}}
    config = manager.load_config(
        model_name="vdcnn", 
        dataset_name="test_dataset",
        overrides=overrides
    )
    
    # 2. Setup Data
    data_manager = DataManager(config=config)
    train_loader, val_loader = data_manager.get_data_loaders()
    
    # 3. Create Model
    # Add vocab_size to the model params from the data config
    config.model.params['num_classes'] = 2
    config.model.params['vocab_size'] = len(config.data.alphabet) + 2
    model = create_model(config.model)
    
    return config, model, train_loader, val_loader

def test_full_experiment_run(full_test_setup):
    """
    Tests a full, end-to-end experiment run.
    """
    # Arrange
    config, model, train_loader, val_loader = full_test_setup
    device = config.device.effective_device
    model.to(device)
    
    runner = ExperimentRunner(config=config, model=model, device=device)
    
    # Act
    results = runner.run_experiment(train_loader=train_loader, val_loader=val_loader)
    
    # Assert
    assert isinstance(results, dict)
    assert "best_accuracy" in results
    assert "history" in results
    
    assert isinstance(results['best_accuracy'], float)
    assert isinstance(results['history'], list)
    assert len(results['history']) == 1 # We ran for 1 epoch
    
    history_epoch_1 = results['history'][0]
    assert "epoch" in history_epoch_1
    assert "train_loss" in history_epoch_1
    assert "val_loss" in history_epoch_1
    assert "val_accuracy" in history_epoch_1
    assert history_epoch_1['epoch'] == 1 