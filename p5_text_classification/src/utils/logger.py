"""
Logger Setup Module

Provides a simplified, configuration-driven function for creating a standardized logger.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from ..config.base_config import AppConfig
from ..utils.constants import LOGS_DIR

# Keep track of configured loggers to avoid duplicate handlers
_loggers = {}

def setup_logger(config: AppConfig, name: str = "default") -> logging.Logger:
    """
    Sets up a logger based on the provided AppConfig.

    It creates a logger that outputs to both the console and a timestamped
    log file inside the directory defined by the LOGS_DIR constant.
    The logger is configured only once per name.

    Args:
        config: The application configuration object.
        name: The name for the logger.

    Returns:
        A fully configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    
    # Set level from config
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter from config
    formatter = logging.Formatter(config.logging.format)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    # Ensure the main logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Construct a descriptive log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # BUG FIX: Read dataset name from the correct location in the AppConfig.
    log_file_name = f"{config.model.name}_{config.data.name}_{timestamp}.log"
    log_file_path = LOGS_DIR / log_file_name

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Cache the logger
    _loggers[name] = logger
    
    # Log the path for user convenience
    logger.info(f"Logging to file: {log_file_path}")

    return logger 