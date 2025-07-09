"""
Trainer and evaluator interface definitions.
Follows the Single Responsibility Principle (SRP) and Interface Segregation Principle (ISP).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging


class TrainerInterface(ABC):
    """
    Abstract interface for model trainers.
    
    Responsible for training logic only, following SRP.
    """
    
    @abstractmethod
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ) -> float:
        """
        Train the model for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            logger: Optional logger for progress tracking
            
        Returns:
            float: Average training loss for the epoch
        """
        pass
    
    @abstractmethod
    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            config: Training configuration
            device: Device to train on
            logger: Optional logger for progress tracking
            
        Returns:
            Dict[str, List[float]]: Training history with losses and metrics
        """
        pass


class EvaluatorInterface(ABC):
    """
    Abstract interface for model evaluators.
    
    Responsible for evaluation logic only, following SRP.
    """
    
    @abstractmethod
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[float, float]:
        """
        Evaluate the model on given data.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            criterion: Loss function
            device: Device to evaluate on
            logger: Optional logger for progress tracking
            
        Returns:
            Tuple[float, float]: (loss, accuracy)
        """
        pass
    
    @abstractmethod
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metrics: List[str]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metrics: List of metrics to compute
            
        Returns:
            Dict[str, float]: Computed metrics
        """
        pass
    
    @abstractmethod
    def generate_report(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            device: Device to evaluate on
            class_names: Optional class names for detailed report
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation report
        """
        pass


class CheckpointManagerInterface(ABC):
    """
    Abstract interface for checkpoint management.
    
    Separated from trainer following ISP principle.
    """
    
    @abstractmethod
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        loss: float,
        accuracy: float,
        config: Dict[str, Any],
        filepath: str
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            loss: Current loss
            accuracy: Current accuracy
            config: Training configuration
            filepath: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            
        Returns:
            Dict[str, Any]: Checkpoint metadata
        """
        pass 