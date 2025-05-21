from typing import Dict, Any, Optional
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import prepare_model, prepare_data_loader
from ray.air.integrations.mlflow import MLflowLoggerCallback
from .base_trainer import BaseTrainer
from ..schemas.model import ModelConfig
from ..schemas.data import BatchData
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import torch
import os
import ray.train.lightning
import uuid

class RayTrainer:
    def __init__(
        self,
        model: pl.LightningModule,
        num_workers: int = 4,
        use_gpu: bool = True,
        resources_per_worker: Dict[str, float] = None,
        databricks_host: str = None,
        databricks_token: str = None
    ):
        self.model = model
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.resources_per_worker = resources_per_worker or {"CPU": 1}
        if use_gpu:
            self.resources_per_worker["GPU"] = 1
        
        # Store Databricks credentials
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
            
    def train_func(self, config: Dict[str, Any]):
        """Training function that will be executed on each worker."""
        # Set up Databricks authentication for MLflow if needed
        if "databricks_host" in config and "databricks_token" in config:
            os.environ["DATABRICKS_HOST"] = config["databricks_host"]
            os.environ["DATABRICKS_TOKEN"] = config["databricks_token"]
        
        # Create a copy of the model for this worker
        model_copy = type(self.model)(**config.get("model_params", {}))
        
        # Prepare model and data
        model = prepare_model(model_copy)
        train_loader = prepare_data_loader(config["train_loader"])
        val_loader = prepare_data_loader(config["val_loader"]) if "val_loader" in config else None
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=config.get("checkpoint_dir", "./checkpoints"),
                filename="{epoch}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min"
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=config.get("patience", 5),
                mode="min"
            ),
            ray.train.lightning.RayTrainReportCallback()
        ]
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=config.get("max_epochs", 10),
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            strategy=ray.train.lightning.RayDDPStrategy(),
            plugins=[ray.train.lightning.RayLightningEnvironment()],
            enable_checkpointing=True
        )
        
        # Prepare trainer for Ray
        trainer = ray.train.lightning.prepare_trainer(trainer)
        
        # Train model
        if val_loader:
            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader)
        
        # Report metrics to Ray
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                train.report({key: value.item()})
            else:
                train.report({key: value})
        
        # Save model artifact if needed
        if hasattr(model, "save_for_ray"):
            model.save_for_ray()
            
    def train(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start distributed training using Ray."""
        # Add Databricks credentials to config if available
        if self.databricks_host and self.databricks_token:
            config["databricks_host"] = self.databricks_host
            config["databricks_token"] = self.databricks_token
        
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker=self.resources_per_worker
        )
        
        # Get experiment name - try to get from notebook context or use provided one
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
        tag_id = f"ray-training-{uuid.uuid4()}"
        
        # Create RunConfig with MLflowLoggerCallback
        run_config = RunConfig(
            storage_path=config.get("storage_path", None),
            name=config.get("run_name", "ray_train_run"),
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri="databricks",  # or config.get("mlflow_tracking_uri", "databricks")
                    experiment_name=experiment_name,
                    save_artifact=True,
                    tags={"tag_id": tag_id, **config.get("tags", {})}
                )
            ]
        )
        
        trainer = TorchTrainer(
            train_loop_per_worker=self.train_func,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=run_config
        )
        
        result = trainer.fit()
        
        # Process and return results
        final_results = {
            "best_checkpoint": result.checkpoint,
            "metrics": {}
        }
        
        # Extract metrics from the result
        if hasattr(result, "metrics_dataframe") and result.metrics_dataframe is not None:
            final_results["metrics"] = result.metrics_dataframe.to_dict(orient='records')[-1]
        
        # Get best model path
        with result.checkpoint.as_directory() as checkpoint_dir:
            final_results["best_model_path"] = os.path.join(
                checkpoint_dir, 
                ray.train.lightning.RayTrainReportCallback.CHECKPOINT_NAME
            )
        
        return final_results
