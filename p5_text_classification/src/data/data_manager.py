"""
Data Manager Module

Handles all data-related tasks: loading, preprocessing, and creating DataLoaders.
This encapsulates data logic, decoupling it from the main training script.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..config.base_config import AppConfig
from .text_preprocessor import TextPreprocessor
from ..utils.constants import PROJECT_ROOT, DATA_DIR

class TextClassificationDataset(Dataset):
    """Custom PyTorch Dataset for text classification."""
    def __init__(self, texts: List[str], labels: List[int], preprocessor: TextPreprocessor, max_length: int):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        # Encode text using preprocessor's internal sequence_length
        encoded_text = self.preprocessor.encode_text(text)
        return torch.tensor(encoded_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class DataManager:
    """
    Orchestrates data loading, preprocessing, and DataLoader creation.
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.preprocessor = TextPreprocessor(
            alphabet=config.data.alphabet,
            sequence_length=config.data.sequence_length
        )
        self.num_classes: Optional[int] = None

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Loads data from CSV, preprocesses it, and prepares train/validation DataLoaders.
        """
        # Locate dataset directory under p5_text_classification/data using DATA_DIR
        base_path = DATA_DIR / self.config.data.name
        train_path = base_path / self.config.data.train_file
        test_path = base_path / self.config.data.test_file
        
        # Load dataframes with specific dtypes to prevent pandas errors
        # and handle potential malformed rows in the CSV.
        # Yahoo dataset has a different format that needs special handling.
        if self.config.data.name == 'yahoo':
            train_df = pd.read_csv(train_path, header=None, names=['label', 'q_title', 'q_content', 'answer'])
            test_df = pd.read_csv(test_path, header=None, names=['label', 'q_title', 'q_content', 'answer'])
            # Combine text columns into a single 'text' column
            train_df['text'] = train_df['q_title'].fillna('') + ' ' + train_df['q_content'].fillna('') + ' ' + train_df['answer'].fillna('')
            test_df['text'] = test_df['q_title'].fillna('') + ' ' + test_df['q_content'].fillna('') + ' ' + test_df['answer'].fillna('')
        else:
            # Default handling for Yelp and other 2-column datasets
            try:
                train_df = pd.read_csv(train_path, header=None, names=['label', 'text'], dtype={'label': 'Int64', 'text': 'str'})
                test_df = pd.read_csv(test_path, header=None, names=['label', 'text'], dtype={'label': 'Int64', 'text': 'str'})
            except Exception as e:
                self.logger.error(f"Failed to load data files: {e}")
                raise

        # Data cleaning: fill NaNs and report issues.
        if train_df['label'].isnull().any():
            self.logger.warning("NaN values found in training labels. Filling with 1 and continuing.")
            train_df['label'] = train_df['label'].fillna(1)
        
        if test_df['label'].isnull().any():
            self.logger.warning("NaN values found in test labels. Filling with 1 and continuing.")
            test_df['label'] = test_df['label'].fillna(1)
            
        # Determine num_classes from the data
        self.num_classes = len(train_df['label'].unique())

        # Apply max_samples limit for quick testing
        if self.config.training.max_samples:
            train_df = train_df.head(self.config.training.max_samples)
        
        # Create a unified dataset and then split it
        train_texts = train_df['text'].astype(str).tolist()
        train_labels = (train_df['label'] - 1).tolist() # Assuming labels are 1-based

        full_dataset = TextClassificationDataset(
            texts=train_texts,
            labels=train_labels,
            preprocessor=self.preprocessor,
            max_length=self.config.data.sequence_length
        )

        # Split the training data to create a validation set
        val_size = int(len(full_dataset) * self.config.validation.split_ratio)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create DataLoaders
        # BUG FIX: DataLoader params now come from the TrainingConfig object.
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers
        )

        return train_loader, val_loader 