from typing import Dict, Any, Optional, List
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import torch
import numpy as np
from datetime import datetime
import json

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "databricks",
        run_name: Optional[str] = None
    ):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id
            
    def start_run(self) -> None:
        """Start a new MLflow run."""
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name
        )
        
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log training parameters."""
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log training metrics."""
        mlflow.log_metrics(metrics, step=step)
        
    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_example: Optional[torch.Tensor] = None
    ) -> None:
        """Log model to MLflow."""
        if input_example is None:
            input_example = torch.randn(1, 3, 224, 224)
            
        signature = infer_signature(
            input_example.numpy(),
            model(input_example).detach().numpy()
        )
        
        mlflow.pytorch.log_model(
            model,
            model_name,
            registered_model_name=model_name,
            signature=signature
        )
        
    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """Log dataset information."""
        mlflow.log_dict(dataset_info, "dataset_info.json")
        
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log local directory as artifacts."""
        mlflow.log_artifacts(local_dir, artifact_path)
        
    def log_visualization(self, fig, name: str) -> None:
        """Log visualization artifacts."""
        mlflow.log_figure(fig, f"{name}.png")
        
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        
    def get_best_run(self, metric: str, mode: str = "min") -> Dict[str, Any]:
        """Get the best run based on a metric."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"metrics.{metric} IS NOT NULL"
        )
        
        if mode == "min":
            best_run = runs.loc[runs[f"metrics.{metric}"].idxmin()]
        else:
            best_run = runs.loc[runs[f"metrics.{metric}"].idxmax()]
            
        return best_run.to_dict() 