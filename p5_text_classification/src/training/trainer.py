"""
Training Module Implementation

Implements comprehensive training and evaluation logic for PyTorch deep learning models.
Provides structured training loops, validation cycles, and progress monitoring with
extensive logging capabilities following SOLID design principles.

@module trainer
@version 1.0.0
@author Text Classification Framework Team
@since 2025-07-04

@requires torch
@requires typing
@requires tqdm

@example
# Basic trainer usage
>>> trainer = Trainer(model, optimizer, config)
>>> results = trainer.train(train_loader, val_loader, epochs=10)

@example
# Trainer with learning rate scheduler
>>> trainer = Trainer(model, optimizer, config, scheduler=scheduler)
>>> results = trainer.train(train_loader, val_loader, epochs=20)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

# The interface is no longer needed in the simplified structure.
# from .trainer_interface import TrainerInterface
from ..utils.logger import setup_logger

class Trainer: # Removed inheritance from TrainerInterface
    """
    Deep learning model trainer implementing comprehensive training and evaluation workflows.
    
    This class orchestrates the complete training process including forward/backward passes,
    loss computation, validation cycles, and progress monitoring. Follows Single Responsibility
    Principle by focusing exclusively on training-related operations.
    
    @class Trainer
    @implements Training Loop Management
    
    @param {nn.Module} model - PyTorch model to be trained
    @param {torch.optim.Optimizer} optimizer - Optimization algorithm instance
    @param {BaseConfig} config - Configuration object containing training parameters
    @param {torch.optim.lr_scheduler._LRScheduler} [scheduler=None] - Learning rate scheduler
    
    @precondition model is properly initialized and parameterized
    @precondition optimizer is configured with model parameters
    @precondition config contains valid training and device specifications
    @invariant model remains on consistent device throughout training
    @postcondition Training history is preserved for analysis
    
    @example
    >>> trainer = Trainer(model, optimizer, config)
    >>> trainer.model.to(trainer.device)  # Automatic device placement
    >>> results = trainer.train(train_loader, val_loader, epochs=5)
    """
    
    def __init__(self, config: 'AppConfig', model: nn.Module, optimizer: torch.optim.Optimizer, 
                 evaluator: 'ClassificationEvaluator', device: str,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Initialize the trainer with model, optimizer, and configuration.

        @constructor
        @param {nn.Module} model - The PyTorch model to be trained
        @param {torch.optim.Optimizer} optimizer - The optimizer for parameter updates
        @param {BaseConfig} config - The configuration object containing training settings
        @param {torch.optim.lr_scheduler._LRScheduler} [scheduler=None] - Optional learning rate scheduler

        @precondition model parameters are properly initialized
        @precondition optimizer is linked to model parameters
        @precondition config.device.effective_device is valid
        @postcondition All components are properly configured and device-synchronized
        @postcondition Logger is established for training monitoring

        @throws {ValueError} If configuration parameters are invalid
        @throws {RuntimeError} If device placement fails
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.logger = setup_logger(config=self.config, name=self.__class__.__name__)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Execute a single training epoch with comprehensive batch processing.

        This method performs the complete training cycle for one epoch, including
        forward propagation, loss computation, backpropagation, and parameter updates
        with detailed progress monitoring and logging.

        @function train_epoch
        @param {DataLoader} train_loader - The DataLoader containing training data batches
        @returns {float} Average training loss across all batches in the epoch

        @precondition train_loader contains valid training data
        @precondition model is in training mode
        @postcondition Model parameters are updated based on training data
        @postcondition Training loss is computed and returned

        @throws {RuntimeError} If forward/backward pass fails
        @throws {OutOfMemoryError} If batch size exceeds available memory

        @example
        >>> avg_loss = trainer.train_epoch(train_loader)
        >>> print(f"Training loss: {avg_loss:.4f}")
        """
        self.model.train() # Set the model to training mode
        total_loss = 0.0
        num_batches = len(train_loader)
        
        self.logger.info(f"🚂 Training on {num_batches} batches...")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass: compute predicted outputs by passing inputs to the model
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress periodically for monitoring
            if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                self.logger.info(f"  📈 Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Execute the complete training process over a specified number of epochs.
        Delegates evaluation to a dedicated Evaluator object.
        """
        self.logger.info(f"🚀 STARTING TRAINING for {self.config.training.epochs} epochs")
        self.logger.info(f"{'='*50}")
        
        best_accuracy = 0.0
        best_predictions = []
        best_true_labels = []
        history = []
        
        for epoch in range(1, self.config.training.epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{self.config.training.epochs}")
            
            avg_train_loss = self.train_epoch(train_loader)
            
            # --- Evaluation Phase ---
            self.logger.info(f"--- Validating Epoch {epoch} ---")
            val_loss, val_accuracy, predictions, true_labels = self.evaluator.evaluate(val_loader)
            self.logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            if self.scheduler:
                self.scheduler.step()

            # --- History and Best Model Tracking ---
            epoch_results = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }
            history.append(epoch_results)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_predictions = predictions
                best_true_labels = true_labels
                self.logger.info(f"🎉 New best validation accuracy: {best_accuracy:.4f}")
                # Save best model checkpoint
                from ..utils.constants import REPORTS_DIR
                model_path = REPORTS_DIR / f"{self.config.model.name}_{self.config.data.name}_best.pth"
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"💾 Saved best model to {model_path}")

        self.logger.info(f"\n{'='*50}")
        self.logger.info("🏁 TRAINING FINISHED")
        
        # Generate predictions CSV file
        import pandas as pd
        from ..utils.constants import REPORTS_DIR
        predictions_df = pd.DataFrame({
            'true_label': best_true_labels,
            'predicted_label': best_predictions
        })
        predictions_path = REPORTS_DIR / f"{self.config.model.name}_{self.config.data.name}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"✅ Saved predictions to {predictions_path}")
        
        final_results = {
            "best_accuracy": best_accuracy,
            "history": history,
            "best_predictions": best_predictions,
            "best_true_labels": best_true_labels
        }
        
        return final_results 