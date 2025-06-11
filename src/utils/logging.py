import logging
import sys
from pathlib import Path
from typing import Optional
import mlflow
from lightning.pytorch.loggers import MLFlowLogger

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """Set up a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")

    return logger

def get_mlflow_logger(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    log_model: bool = True,
    tags: Optional[dict] = None
) -> MLFlowLogger:
    """Get MLflow logger for PyTorch Lightning.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI
        run_name: Name of the MLflow run
        log_model: Whether to log model checkpoints
        tags: Additional tags for the run
        
    Returns:
        MLFlowLogger instance
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        log_model=log_model,
        tags=tags
    ) 