"""
Unit tests for the Trainer class.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY

import torch

from p5_text_classification.src.training.trainer import Trainer
from p5_text_classification.src.config.config_manager import ConfigManager
from p5_text_classification.src.training.evaluator import ClassificationEvaluator


@pytest.fixture
def mock_config(test_config_dir):
    """Provides a mocked AppConfig object for testing."""
    manager = ConfigManager(config_dir=test_config_dir)
    config = manager.load_config(model_name="vdcnn", dataset_name="test_dataset")
    config.training.epochs = 2 # Test over 2 epochs
    return config

@pytest.fixture
def mock_dependencies(mock_config):
    """Mocks all external dependencies of the Trainer."""
    device = mock_config.device.effective_device
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
    mock_evaluator = MagicMock()
    
    # Mock the return value of the model call to be a tensor on the correct device
    mock_output = torch.randn(1, 1, requires_grad=True).to(device)
    mock_model.return_value = mock_output
    
    # Mock the return value of the evaluator
    mock_evaluator.evaluate.return_value = (0.5, 0.9, [0, 1], [0, 1]) # loss, acc, preds, labels

    return {
        "model": mock_model,
        "optimizer": mock_optimizer,
        "config": mock_config,
        "scheduler": mock_scheduler,
        "evaluator": mock_evaluator
    }

@patch('src.training.trainer.setup_logger')
def test_trainer_initialization(mock_setup_logger, mock_dependencies):
    """Tests that the Trainer initializes correctly."""
    trainer = Trainer(
        model=mock_dependencies['model'],
        optimizer=mock_dependencies['optimizer'],
        config=mock_dependencies['config'],
        scheduler=mock_dependencies['scheduler']
    )
    mock_dependencies['model'].to.assert_called_once_with(mock_dependencies['config'].device.effective_device)
    mock_setup_logger.assert_called_once()
    assert trainer.criterion is not None

@patch('src.training.trainer.setup_logger')
@patch('torch.Tensor.backward')
def test_train_epoch_logic(mock_backward, mock_setup_logger, mock_dependencies):
    """
    Tests the core logic of a single training epoch.
    Verifies that model is set to train, and all relevant methods are called.
    """
    trainer = Trainer(**{k: v for k, v in mock_dependencies.items() if k != 'evaluator'})
    device = mock_dependencies['config'].device.effective_device
    
    # Mock data loader to return a single batch on the correct device
    # Target tensor must be Long type for CrossEntropyLoss
    mock_data_loader = [(torch.randn(1, 1).to(device), torch.randint(0, 1, (1,)).to(device))]

    # Act
    avg_loss = trainer.train_epoch(mock_data_loader)

    # Assert
    mock_dependencies['model'].train.assert_called_once()
    mock_dependencies['optimizer'].zero_grad.assert_called_once()
    # Check that the model was called
    mock_dependencies['model'].assert_called_once()
    # Check that backward was called on the loss tensor
    mock_backward.assert_called_once()
    mock_dependencies['optimizer'].step.assert_called_once()
    assert isinstance(avg_loss, float)

@patch('src.training.trainer.setup_logger')
def test_full_training_loop(mock_setup_logger, mock_dependencies):
    """
    Tests the full `train` method loop over multiple epochs.
    Verifies that train_epoch, evaluate, and scheduler.step are called correctly.
    """
    trainer = Trainer(
        model=mock_dependencies['model'],
        optimizer=mock_dependencies['optimizer'],
        config=mock_dependencies['config'],
        scheduler=mock_dependencies['scheduler']
    )
    device = mock_dependencies['config'].device.effective_device
    
    # Mock data loaders on the correct device
    # Target tensor must be Long type for CrossEntropyLoss
    mock_train_loader = [(torch.randn(1, 1).to(device), torch.randint(0, 1, (1,)).to(device))] * 2
    mock_val_loader = [(torch.randn(1, 1).to(device), torch.randint(0, 1, (1,)).to(device))]

    # Act
    results = trainer.train(
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        evaluator=mock_dependencies['evaluator']
    )

    # Assert
    num_epochs = mock_dependencies['config'].training.epochs
    assert mock_dependencies['evaluator'].evaluate.call_count == num_epochs
    assert mock_dependencies['scheduler'].step.call_count == num_epochs
    
    # Check that best_accuracy is correctly reported
    assert results['best_accuracy'] == 0.9
    assert len(results['history']) == num_epochs 