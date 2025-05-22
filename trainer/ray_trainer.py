from typing import Dict, Any, Optional
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from tasks.common.factory import make_module
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class RayTrainer:
    """Distributed trainer using Ray."""
    
    def __init__(
        self,
        task: str,
        model_ckpt: str,
        num_workers: int = 4,
        use_gpu: bool = True,
        resources_per_worker: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Initialize Ray trainer.
        
        Args:
            task: One of "classification", "detection", "segmentation"
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            num_workers: Number of Ray workers
            use_gpu: Whether to use GPU
            resources_per_worker: Resources per worker
            **kwargs: Additional arguments passed to make_module
        """
        self.task = task
        self.model_ckpt = model_ckpt
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.resources_per_worker = resources_per_worker or {"CPU": 1}
        if use_gpu:
            self.resources_per_worker["GPU"] = 1
        self.kwargs = kwargs
    
    def train_func(self, config: Dict[str, Any]):
        """Training function executed on each worker."""
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(config["experiment_name"])
        
        with mlflow.start_run(run_name=config["run_name"]):
            # Log parameters
            mlflow.log_params(config)
            
            # Create module using factory
            module = make_module(
                task=self.task,
                model_ckpt=self.model_ckpt,
                **self.kwargs
            )
            
            # Prepare model and data
            model = train.torch.prepare_model(module)
            train_loader = train.torch.prepare_data_loader(config["train_loader"])
            val_loader = train.torch.prepare_data_loader(config["val_loader"])
            
            # Setup callbacks
            callbacks = [
                ModelCheckpoint(
                    dirpath=config["checkpoint_dir"],
                    filename="{epoch}-{val_loss:.2f}",
                    save_top_k=3,
                    monitor="val_loss",
                    mode="min"
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode="min"
                )
            ]
            
            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=config["max_epochs"],
                callbacks=callbacks,
                accelerator="gpu" if self.use_gpu else "cpu",
                devices=1,
                strategy="ddp"
            )
            
            # Train model
            trainer.fit(model, train_loader, val_loader)
            
            # Log metrics
            metrics = trainer.callback_metrics
            mlflow.log_metrics(metrics)
            
            # Save model
            model.save_to_mlflow(config["model_path"])
    
    def train(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start distributed training."""
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker=self.resources_per_worker
        )
        
        trainer = TorchTrainer(
            train_loop_per_worker=self.train_func,
            train_loop_config=config,
            scaling_config=scaling_config
        )
        
        result = trainer.fit()
        return result 