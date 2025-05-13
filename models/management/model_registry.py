from typing import Dict, Any, Optional
import mlflow
from mlflow.models import infer_signature
import torch
from databricks.sdk import WorkspaceClient
from ...data.unity_catalog.catalog_manager import CatalogManager

class ModelRegistry:
    """Manages model registration and versioning with Unity Catalog integration."""
    
    def __init__(
        self,
        workspace_url: str,
        token: str,
        catalog_name: str,
        schema_name: str,
        model_name: str
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
        
    def register_model(
        self,
        model: torch.nn.Module,
        version: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        input_example: Optional[torch.Tensor] = None
    ) -> str:
        """Register a model in MLflow and Unity Catalog."""
        # Set up MLflow
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(f"/Users/{self.workspace.current_user().user_name}/{self.model_name}")
        
        with mlflow.start_run(run_name=f"version_{version}"):
            # Log parameters and metrics
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            
            # Prepare input example if not provided
            if input_example is None:
                input_example = torch.randn(1, 3, 224, 224)
                
            # Infer model signature
            signature = infer_signature(
                input_example.numpy(),
                model(input_example).detach().numpy()
            )
            
            # Log model to MLflow
            model_uri = mlflow.pytorch.log_model(
                model,
                self.model_name,
                registered_model_name=self.model_name,
                signature=signature
            ).model_uri
            
            # Register model metadata in Unity Catalog
            self.catalog_manager.register_model_metadata(
                table_name=f"{self.model_name}_metadata",
                model_name=self.model_name,
                version=version,
                metrics=metrics,
                parameters=parameters
            )
            
            return model_uri
            
    def get_model_version(self, version: str) -> torch.nn.Module:
        """Get a specific version of the model."""
        model_uri = f"models:/{self.model_name}/{version}"
        return mlflow.pytorch.load_model(model_uri)
        
    def list_model_versions(self) -> Dict[str, Any]:
        """List all versions of the model."""
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{self.model_name}'")
        
        return {
            version.version: {
                "status": version.status,
                "run_id": version.run_id,
                "metrics": client.get_run(version.run_id).data.metrics,
                "parameters": client.get_run(version.run_id).data.params
            }
            for version in versions
        }
        
    def transition_model_version(
        self,
        version: str,
        stage: str
    ) -> None:
        """Transition a model version to a specific stage."""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage
        )
        
    def delete_model_version(self, version: str) -> None:
        """Delete a specific version of the model."""
        client = mlflow.tracking.MlflowClient()
        client.delete_model_version(
            name=self.model_name,
            version=version
        ) 