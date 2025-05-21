from typing import Dict, Any, Optional
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from trainer.base_trainer import BaseTrainer
from schemas.model import ModelConfig
from schemas.data import BatchData
import os
import uuid
import pytorch_lightning as pl
import ray.train.lightning

class RayTrainer:
    """Distributed trainer using Ray with PyTorch Lightning integration."""
    
    def __init__(
        self,
        trainer: BaseTrainer,
        num_workers: int = 4,
        use_gpu: bool = True,
        resources_per_worker: Optional[Dict[str, float]] = None,
        databricks_host: str = None,
        databricks_token: str = None
    ):
        self.trainer = trainer
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.resources_per_worker = resources_per_worker or {"CPU": 1}
        if use_gpu:
            self.resources_per_worker["GPU"] = 1
        
        # Store Databricks credentials
        self.databricks_host = databricks_host or os.environ.get("DATABRICKS_HOST")
        self.databricks_token = databricks_token or os.environ.get("DATABRICKS_TOKEN")
    
    def train_func(self, config: Dict[str, Any]):
        """Training function executed on each worker using PyTorch Lightning."""
        # Set Databricks environment variables for MLflow if needed
        if "databricks_host" in config and "databricks_token" in config:
            os.environ["DATABRICKS_HOST"] = config["databricks_host"]
            os.environ["DATABRICKS_TOKEN"] = config["databricks_token"]
        
        # Get the model from the trainer
        model = self.trainer.get_model()
        
        # Prepare model for distributed training
        model = train.torch.prepare_model(model)
        
        # Prepare data loaders
        train_loader = train.torch.prepare_data_loader(config["train_loader"])
        val_loader = train.torch.prepare_data_loader(config["val_loader"]) if "val_loader" in config else None
        
        # Create Lightning callbacks
        callbacks = [
            ray.train.lightning.RayTrainReportCallback()
        ]
        
        # Add any additional callbacks from the trainer
        if hasattr(self.trainer, "get_callbacks"):
            trainer_callbacks = self.trainer.get_callbacks()
            if trainer_callbacks:
                callbacks.extend(trainer_callbacks)
        
        # Initialize Lightning trainer with Ray-specific configuration
        lightning_trainer = pl.Trainer(
            max_epochs=config.get("max_epochs", 10),
            devices="auto",
            accelerator="auto",
            strategy=ray.train.lightning.RayDDPStrategy(),
            plugins=[ray.train.lightning.RayLightningEnvironment()],
            callbacks=callbacks,
            enable_checkpointing=True,
            use_distributed_sampler=False
        )
        
        # Prepare the Lightning trainer for Ray
        lightning_trainer = ray.train.lightning.prepare_trainer(lightning_trainer)
        
        # Start training
        if val_loader:
            lightning_trainer.fit(model, train_loader, val_loader)
        else:
            lightning_trainer.fit(model, train_loader)
    
    def train(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start distributed training with Ray."""
        # Add Databricks credentials to config
        if self.databricks_host and self.databricks_token:
            config["databricks_host"] = self.databricks_host
            config["databricks_token"] = self.databricks_token
        
        # Create scaling config
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker=self.resources_per_worker
        )
        
        # Try to get experiment name from notebook context or use provided one
        experiment_name = config.get("experiment_name", "ray-training")
        try:
            from dbutils import DBUtils
            dbutils = DBUtils()
            notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
            if notebook_path:
                experiment_name = notebook_path
        except:
            pass
        
        # Create a unique tag for this run
        tag_id = f"ray-training-{uuid.uuid4().hex[:8]}"
        
        # Create callbacks for RunConfig
        callbacks = []
        
        # Add MLflow callback
        callbacks.append(
            MLflowLoggerCallback(
                tracking_uri="databricks",
                experiment_name=experiment_name,
                save_artifact=True,
                tags={"tag_id": tag_id, **config.get("tags", {})}
            )
        )
        
        # Add any additional callbacks from config
        if "callbacks" in config and isinstance(config["callbacks"], list):
            callbacks.extend(config["callbacks"])
        
        # Create RunConfig with callbacks
        run_config = RunConfig(
            storage_path=config.get("storage_path", None),
            name=config.get("run_name", "ray_train_run"),
            callbacks=callbacks
        )
        
        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=self.train_func,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=run_config
        )
        
        # Start training
        result = trainer.fit()
        
        # Process and return results
        final_results = {
            "best_checkpoint": result.checkpoint,
            "metrics": {}
        }
        
        # Extract metrics from the result
        if hasattr(result, "metrics_dataframe") and result.metrics_dataframe is not None:
            final_results["metrics"] = result.metrics_dataframe.to_dict(orient='records')[-1]
        
        # Get best model path if available
        if result.checkpoint:
            with result.checkpoint.as_directory() as checkpoint_dir:
                final_results["best_model_path"] = os.path.join(
                    checkpoint_dir, 
                    ray.train.lightning.RayTrainReportCallback.CHECKPOINT_NAME
                )
        
        return final_results
