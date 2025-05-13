from typing import Dict, Any, Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
from ...data.unity_catalog.catalog_manager import CatalogManager
import mlflow
import json
import logging
import os
from datetime import datetime

class DeploymentPipeline:
    """Handles model deployment with Unity Catalog integration."""
    
    def __init__(
        self,
        workspace_url: str,
        token: str,
        catalog_name: str,
        schema_name: str,
        model_name: str,
        experiment_name: str
    ):
        self.workspace = WorkspaceClient(
            host=workspace_url,
            token=token
        )
        self.catalog_manager = CatalogManager(
            workspace_url=workspace_url,
            token=token,
            catalog_name=catalog_name,
            schema_name=schema_name
        )
        self.model_name = model_name
        self.experiment_name = experiment_name
        
    def run_pipeline(
        self,
        metrics_threshold: Dict[str, float],
        endpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the deployment pipeline with Unity Catalog integration."""
        # Get the latest model version
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{self.model_name}'")
        latest_version = max(versions, key=lambda x: int(x.version))
        
        # Check if metrics meet threshold
        run = client.get_run(latest_version.run_id)
        metrics = run.data.metrics
        
        for metric_name, threshold in metrics_threshold.items():
            if metric_name not in metrics or metrics[metric_name] < threshold:
                raise ValueError(
                    f"Metric {metric_name} ({metrics.get(metric_name, 'N/A')}) "
                    f"does not meet threshold {threshold}"
                )
                
        # Create or update endpoint
        endpoint_name = endpoint_name or f"{self.model_name}_endpoint"
        config = EndpointCoreConfigInput(
            served_models=[
                {
                    "model_name": self.model_name,
                    "model_version": latest_version.version,
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
            },
            auto_capture_config={
                "enabled": True,
                "catalog_name": self.catalog_manager.catalog_name,
                "schema_name": self.catalog_manager.schema_name,
                "table_name": f"{endpoint_name}_predictions"
            }
        )
        
        try:
            endpoint = self.workspace.serving_endpoints.create(
                name=endpoint_name,
                config=config
            )
        except Exception as e:
            if "already exists" in str(e):
                endpoint = self.workspace.serving_endpoints.update(
                    name=endpoint_name,
                    config=config
                )
            else:
                raise e
                
        # Register deployment metadata in Unity Catalog
        deployment_metadata = {
            "endpoint_name": endpoint_name,
            "model_name": self.model_name,
            "model_version": latest_version.version,
            "metrics": metrics,
            "deployment_time": run.info.end_time,
            "config": json.dumps(config.to_dict())
        }
        
        self.catalog_manager.register_model_metadata(
            table_name=f"{self.model_name}_deployments",
            model_name=self.model_name,
            version=latest_version.version,
            metrics=metrics,
            parameters=deployment_metadata
        )
        
        return {
            "endpoint_name": endpoint_name,
            "model_version": latest_version.version,
            "metrics": metrics,
            "deployment_time": run.info.end_time
        }
        
    def monitor_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Monitor endpoint performance and log metrics to Unity Catalog."""
        endpoint = self.workspace.serving_endpoints.get(name=endpoint_name)
        
        # Get endpoint metrics
        metrics = {
            "total_requests": endpoint.total_requests,
            "average_latency": endpoint.average_latency,
            "error_rate": endpoint.error_rate
        }
        
        # Log metrics to Unity Catalog
        self.catalog_manager.register_model_metadata(
            table_name=f"{endpoint_name}_metrics",
            model_name=self.model_name,
            version="latest",
            metrics=metrics,
            parameters={"endpoint_name": endpoint_name}
        )
        
        return metrics 