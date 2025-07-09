"""
Base configuration classes (dataclasses) for the project.

This module defines the schema for the application's configuration,
ensuring that all configuration values are type-safe and have a single,
predictable structure. Using dataclasses allows for easy creation,
validation, and access to configuration parameters.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch

# This is the full and corrected content for base_config.py

@dataclass
class SchedulerConfig:
    """Configuration for the learning rate scheduler."""
    type: str = "MultiStepLR"
    milestones_ratio: Optional[List[float]] = None
    milestones: Optional[List[int]] = None
    gamma: float = 0.1

    def __post_init__(self):
        # Ensure correct types after initialization from dict
        self.gamma = float(self.gamma)
        
        # Validate that at least one milestone format is provided
        if self.milestones_ratio is None and self.milestones is None:
            # Set default if neither is provided
            self.milestones_ratio = [0.8]
        
        # Type validation
        if self.milestones_ratio is not None and not isinstance(self.milestones_ratio, list):
            raise TypeError(f"milestones_ratio must be a list, but got {type(self.milestones_ratio)}")
        if self.milestones is not None and not isinstance(self.milestones, list):
            raise TypeError(f"milestones must be a list, but got {type(self.milestones)}")

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    epochs: int = 1
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-5
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    # Moved from DataConfig as they are training-time parameters
    batch_size: int = 128
    num_workers: int = 0
    shuffle: bool = True
    max_samples: Optional[int] = None
    expected_duration_hours: Optional[float] = None # For timeout calculation

    def __post_init__(self):
        # Ensures that if a dictionary is passed for 'scheduler',
        # it gets converted into a SchedulerConfig instance.
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)
        # Basic type coercion
        self.epochs = int(self.epochs)
        self.lr = float(self.lr)
        self.momentum = float(self.momentum)
        self.weight_decay = float(self.weight_decay)
        self.batch_size = int(self.batch_size)
        self.num_workers = int(self.num_workers)
        if self.expected_duration_hours is not None:
            self.expected_duration_hours = float(self.expected_duration_hours)

@dataclass
class DataConfig:
    """Data-related configuration parameters."""
    name: str # Added field
    path: str
    train_file: str
    test_file: str
    class_file: str
    sequence_length: int
    alphabet: str
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    pad_token_id: int = 0
    unk_token_id: int = 1
    # BATCH_SIZE and other training-related params moved to TrainingConfig

@dataclass
class ModelConfig:
    """Model-specific configuration parameters."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging-related configuration parameters."""
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

@dataclass
class DeviceConfig:
    """Device-related configuration parameters."""
    preferred: str = "cuda"
    effective_device: str = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available() and self.preferred == "cuda":
            self.effective_device = 'cuda'
        else:
            self.effective_device = 'cpu'

@dataclass
class RandomConfig:
    """Reproducibility-related configuration parameters."""
    seed: int = 42

@dataclass
class ValidationConfig:
    """Validation-related configuration parameters."""
    split_ratio: float = 0.2

@dataclass
class AppConfig:
    """
    Root configuration class that aggregates all configuration sections.
    This provides a single, type-safe object to pass around the application.
    """
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    device: DeviceConfig
    random: RandomConfig
    validation: ValidationConfig
    # The 'dataset' field is now removed. DataConfig is the SSOT for data params.

    @property
    def vocab_size(self) -> int:
        """Calculate the vocabulary size based on alphabet and special tokens."""
        # +2 for PAD and UNK tokens
        return len(self.data.alphabet) + 2 