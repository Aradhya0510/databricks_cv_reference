from typing import Dict, Any, Optional
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
import json
import logging
import os
from datetime import datetime

class DeploymentPipeline:
    def __init__(
        self,
        workspace_url: str,
        token: str,
        model_name: str,
        experiment_name: str
    ):
        self.workspace = WorkspaceClient(
            host=workspace_url,
            token=token
        )
        self.model_name = model_name
        self.experiment_name = experiment_name
        
    def run_pipeline(
        self,
        metrics_threshold: Dict[str, float],
        endpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete CI/CD pipeline."""
        # Get the latest model version
        latest_version = self._get_latest_model_version()
        
        # Evaluate model
        metrics = self._evaluate_model(latest_version)
        
        # Check if metrics meet threshold
        if not self._check_metrics_threshold(metrics, metrics_threshold):
            raise ValueError("Model metrics do not meet threshold requirements")
            
        # Deploy model
        if endpoint_name is None:
            endpoint_name = f"{self.model_name}_endpoint"
            
        endpoint = self._deploy_model(latest_version, endpoint_name)
        
        return {
            "model_version": latest_version,
            "metrics": metrics,
            "endpoint_name": endpoint_name,
            "deployment_time": datetime.now().isoformat()
        }
        
    def _get_latest_model_version(self) -> str:
        """Get the latest model version from MLflow."""
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{self.model_name}'")
        return max(versions, key=lambda x: x.version).version
        
    def _evaluate_model(self, model_version: str) -> Dict[str, float]:
        """Evaluate model performance."""
        # Load model
        model_uri = f"models:/{self.model_name}/versions/{model_version}"
        model = mlflow.pytorch.load_model(model_uri)
        
        # Run evaluation
        # This should be implemented based on your specific evaluation needs
        metrics = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1": 0.94
        }
        
        return metrics
        
    def _check_metrics_threshold(
        self,
        metrics: Dict[str, float],
        threshold: Dict[str, float]
    ) -> bool:
        """Check if metrics meet threshold requirements."""
        for metric, value in metrics.items():
            if metric in threshold and value < threshold[metric]:
                return False
        return True
        
    def _deploy_model(
        self,
        model_version: str,
        endpoint_name: str
    ) -> str:
        """Deploy model to serving endpoint."""
        config = EndpointCoreConfigInput(
            served_models=[
                {
                    "model_name": self.model_name,
                    "model_version": model_version,
                    "workload_size": "Standard_DS3_v2",
                    "scale_to_zero_enabled": True
                }
            ],
            traffic_config={
                "routes": [
                    {
                        "served_model_name": self.model_name,
                        "traffic_percentage": 100
                    }
                ]
            }
        )
        
        try:
            endpoint = self.workspace.serving_endpoints.create(
                name=endpoint_name,
                config=config
            )
        except Exception as e:
            # If endpoint exists, update it
            if "already exists" in str(e):
                endpoint = self.workspace.serving_endpoints.update_config(
                    endpoint_name=endpoint_name,
                    config=config
                )
            else:
                raise e
                
        return endpoint.name 