"""
Logging utility for NBA ETL Pipeline
"""

import logging
import os
from pathlib import Path
from config.settings import LogConfig


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(LogConfig.FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LogConfig.LEVEL.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        LogConfig.FORMAT,
        datefmt=LogConfig.DATE_FORMAT
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(LogConfig.FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get logger instance (creates if doesn't exist)
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name)