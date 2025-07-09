"""
Unit tests for the ClassificationEvaluator.
"""
import pytest
from unittest.mock import MagicMock, patch

import torch

from p5_text_classification.src.training.evaluator import ClassificationEvaluator
from p5_text_classification.src.config.base_config import AppConfig


@pytest.fixture
def mock_config(test_config_dir):
    """Provides a mocked AppConfig object for testing."""
    manager = AppConfig(config_dir=test_config_dir)
    return manager.load_config(model_name="vdcnn", dataset_name="test_dataset")

@pytest.fixture
def mock_model_and_data(mock_config):
    """Mocks the model and a data loader for the evaluator."""
    device = mock_config.device.effective_device
    
    # Mock Model
    mock_model = MagicMock(spec=torch.nn.Module)
    
    # Mock Data: 2 batches
    # Batch 1: 1 correct, 2 incorrect
    # Batch 2: 1 correct, 2 incorrect
    # Total: 2 correct, 4 incorrect -> 33.3% accuracy
    
    # Batch 1 outputs (logits) and targets
    outputs1 = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]).to(device) # preds: 1, 0, 1
    targets1 = torch.tensor([1, 1, 0]).to(device) # expected: 1, 1, 0 -> correct: True, False, False
    
    # Batch 2 outputs and targets
    outputs2 = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]).to(device) # preds: 0, 1, 0
    targets2 = torch.tensor([0, 0, 1]).to(device) # expected: 0, 0, 1 -> correct: True, False, False

    mock_model.side_effect = [outputs1, outputs2]
    
    # Mock DataLoader
    mock_data_loader = [
        (torch.randn_like(outputs1), targets1),
        (torch.randn_like(outputs2), targets2)
    ]

    return {
        "model": mock_model,
        "data_loader": mock_data_loader,
        "config": mock_config,
        "device": device
    }

@patch('src.training.evaluator.setup_logger')
def test_evaluator_initialization(mock_setup_logger, mock_model_and_data):
    """Tests that the Evaluator initializes correctly."""
    evaluator = ClassificationEvaluator(
        model=mock_model_and_data['model'],
        config=mock_model_and_data['config'],
        device=mock_model_and_data['device']
    )
    mock_setup_logger.assert_called_once()
    assert evaluator.criterion is not None

@patch('src.training.evaluator.setup_logger')
def test_evaluate_logic_and_metrics(mock_setup_logger, mock_model_and_data):
    """
    Tests the core evaluation logic.
    Verifies model.eval() is called, and accuracy/loss are calculated correctly.
    """
    evaluator = ClassificationEvaluator(
        model=mock_model_and_data['model'],
        config=mock_model_and_data['config'],
        device=mock_model_and_data['device']
    )
    
    # Act
    avg_loss, accuracy, all_preds, all_targets = evaluator.evaluate(
        data_loader=mock_model_and_data['data_loader']
    )

    # Assert
    # Check that model.eval() was called
    mock_model_and_data['model'].eval.assert_called_once()
    
    # Check accuracy (2 correct / 6 total)
    assert pytest.approx(accuracy) == 1/3
    
    # Check that loss is a float (we don't check the exact value as it depends on nn.CrossEntropyLoss)
    assert isinstance(avg_loss, float)
    
    # Check collected predictions and targets
    expected_preds = [1, 0, 1, 0, 1, 0]
    expected_targets = [1, 1, 0, 0, 0, 1]
    assert all_preds == expected_preds
    assert all_targets == expected_targets 