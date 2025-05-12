from typing import Dict, Any, Optional
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
import json
import logging

class ModelServer:
    def __init__(
        self,
        workspace_url: str,
        token: str,
        model_name: str,
        model_version: Optional[str] = None
    ):
        self.workspace = WorkspaceClient(
            host=workspace_url,
            token=token
        )
        self.model_name = model_name
        self.model_version = model_version
        
    def load_model(self) -> torch.nn.Module:
        """Load the model from MLflow."""
        model_uri = f"models:/{self.model_name}"
        if self.model_version:
            model_uri += f"/versions/{self.model_version}"
            
        return mlflow.pytorch.load_model(model_uri)
        
    def create_endpoint(
        self,
        endpoint_name: str,
        instance_type: str = "Standard_DS3_v2",
        min_instances: int = 1,
        max_instances: int = 5,
        scale_to_zero: bool = True
    ) -> str:
        """Create a new model serving endpoint."""
        config = EndpointCoreConfigInput(
            served_models=[
                {
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "workload_size": instance_type,
                    "scale_to_zero_enabled": scale_to_zero
                }
            ],
            traffic_config={
                "routes": [
                    {
                        "served_model_name": self.model_name,
                        "traffic_percentage": 100
                    }
                ]
            },
            auto_capture_config={
                "enabled": True,
                "catalog_name": "hive_metastore",
                "schema_name": "default",
                "table_name": f"{endpoint_name}_predictions"
            }
        )
        
        endpoint = self.workspace.serving_endpoints.create(
            name=endpoint_name,
            config=config
        )
        
        return endpoint.name
        
    def update_endpoint(
        self,
        endpoint_name: str,
        new_model_version: str,
        traffic_percentage: int = 100
    ) -> None:
        """Update an existing endpoint with a new model version."""
        endpoint = self.workspace.serving_endpoints.get(endpoint_name)
        
        config = EndpointCoreConfigInput(
            served_models=[
                {
                    "model_name": self.model_name,
                    "model_version": new_model_version,
                    "workload_size": endpoint.config.served_models[0].workload_size,
                    "scale_to_zero_enabled": endpoint.config.served_models[0].scale_to_zero_enabled
                }
            ],
            traffic_config={
                "routes": [
                    {
                        "served_model_name": self.model_name,
                        "traffic_percentage": traffic_percentage
                    }
                ]
            }
        )
        
        self.workspace.serving_endpoints.update_config(
            endpoint_name=endpoint_name,
            config=config
        )
        
    def monitor_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Get monitoring metrics for an endpoint."""
        metrics = self.workspace.serving_endpoints.get_metrics(endpoint_name)
        return {
            "total_requests": metrics.total_requests,
            "average_latency": metrics.average_latency,
            "requests_per_second": metrics.requests_per_second,
            "error_rate": metrics.error_rate
        }
        
    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete a model serving endpoint."""
        self.workspace.serving_endpoints.delete(endpoint_name) 