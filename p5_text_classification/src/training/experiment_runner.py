"""
Experiment Runner Module

This module contains the high-level logic to run a full training and evaluation experiment.
It orchestrates the data loading, model creation, training, and result reporting.
"""

from typing import Dict, Any, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
from datetime import datetime
from ..utils.constants import TRAINING_HISTORY_DIR

from ..config.base_config import AppConfig
from ..utils.logger import setup_logger
from .trainer import Trainer
from .evaluator import ClassificationEvaluator
from ..data.data_manager import DataManager
from ..models.model_factory import create_model

class ExperimentRunner:
    """
    Orchestrates the entire experiment lifecycle based on a given configuration.
    This includes setting up logging, data, model, trainer, and running the
    training/evaluation process.
    """
    def __init__(self, config: AppConfig):
        """
        Initializes the runner with a fully-resolved configuration object
        and sets up all necessary components for the experiment.
        
        Args:
            config (AppConfig): The application configuration.
        """
        self.config = config
        self.logger = setup_logger(config, name=self.__class__.__name__)
        # Prefer GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.results: Dict[str, Any] = {}
        
        self.logger.info("🚀 Initializing experiment components...")
        self.logger.info(f"Model: {self.config.model.name}, Dataset: {self.config.data.name}")
        
        # 1. Prepare DataLoaders
        data_manager = DataManager(self.config)
        self.train_loader, self.val_loader = data_manager.get_data_loaders()
        
        # 2. Create Model
        self.config.model.params['vocab_size'] = data_manager.preprocessor.vocab_size
        if data_manager.num_classes:
            self.config.model.params['num_classes'] = data_manager.num_classes
        else:
            raise ValueError("Number of classes not determined by DataManager. Check your data.")
            
        model = create_model(self.config.model).to(self.device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model '{self.config.model.name}' created with {num_params:,} parameters.")
        
        # 3. Initialize Trainer and Evaluator
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.config.training.lr, momentum=self.config.training.momentum
        )
        evaluator = ClassificationEvaluator(model=model, config=self.config, device=self.device)
        
        # 4. Create the Trainer instance
        self.trainer = Trainer(self.config, model, optimizer, evaluator, self.device)

    def run(self):
        """
        Executes the entire experiment from start to finish.
        """
        self.logger.info("▶️ Running training...")
        self.logger.info(f"Epochs: {self.config.training.epochs}, LR: {self.config.training.lr}, Batch Size: {self.config.training.batch_size}")

        self.results = self.trainer.train(self.train_loader, self.val_loader)
        # Save training history to CSV for plotting and analysis
        history = self.results.get('history', [])
        if history:
            df_history = pd.DataFrame(history)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = f"{self.config.model.name}_{self.config.data.name}_{timestamp}.csv"
            csv_path = TRAINING_HISTORY_DIR / csv_name
            df_history.to_csv(csv_path, index=False)
            self.logger.info(f"✅ Training history saved to {csv_path}")
        
        # Log Final Results
        final_accuracy = self.results.get('best_accuracy', 0) * 100
        self.logger.info(f"✅ Experiment finished. Final validation accuracy: {final_accuracy:.2f}%")
        return self.results 