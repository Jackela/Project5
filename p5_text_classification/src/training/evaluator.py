"""
Evaluator Module

This module defines the `ClassificationEvaluator` class, which encapsulates
all logic related to model evaluation. It is responsible for calculating
metrics like loss and accuracy on a given dataset.
"""

import torch
import torch.nn as nn
from typing import Tuple, List
from torch.utils.data import DataLoader

from ..config.base_config import AppConfig
from ..utils.logger import setup_logger


class ClassificationEvaluator:
    """
    Handles the evaluation of a classification model.
    """
    def __init__(self, model: nn.Module, config: AppConfig, device: str):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.logger = setup_logger(config=config, name=self.__class__.__name__)

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, List[int], List[int]]:
        """
        Evaluates the model on a given dataset.

        Args:
            data_loader: DataLoader for the validation or test set.

        Returns:
            A tuple containing:
            - Average loss.
            - Accuracy (as a fraction).
            - A list of all predictions.
            - A list of all true labels.
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        num_batches = len(data_loader)
        self.logger.info(f"🔍 Evaluating on {num_batches} batches...")

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total_samples += target.size(0)
                correct_predictions += (predicted == target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        
        self.logger.info(f"✅ Evaluation complete. Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")

        return avg_loss, accuracy, all_preds, all_targets 