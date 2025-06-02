import logging
import sys
from pathlib import Path
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient
import os

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

    # File handler if log_file is provided and not in Databricks notebook
    if log_file and not os.environ.get('DATABRICKS_RUNTIME_VERSION'):
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")

    return logger

class MLflowLogger:
    """Wrapper for MLflow logging functionality."""
    
    def __init__(self, experiment_name: str):
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        self._setup_experiment()

    def _setup_experiment(self):
        """Set up MLflow experiment."""
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

    def log_params(self, params: dict):
        """Log parameters to MLflow."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, artifact_path: str):
        """Log model to MLflow."""
        mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()

def get_metric_logger(experiment_name: str) -> MLflowLogger:
    """Get MLflow logger for metrics."""
    return MLflowLogger(experiment_name)

def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    metrics: dict,
    mlflow_logger: Optional[MLflowLogger] = None
):
    """Log training progress to both console and MLflow."""
    # Log to console
    logger.info(f"Epoch {epoch}: {metrics}")

    # Log to MLflow if logger is provided
    if mlflow_logger:
        mlflow_logger.log_metrics(metrics, step=epoch) 