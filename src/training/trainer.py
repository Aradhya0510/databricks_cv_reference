import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.mlflow import MLflowLoggerCallback
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

class UnifiedTrainer:
    """Unified trainer for all computer vision tasks using Ray on Databricks."""
    
    def __init__(self, task: str, model_class=None, data_module_class=None, config_path: str = None, distributed: bool = False, model_config_class=None, config: dict = None, model=None, data_module=None):
        """Initialize the trainer.
        
        Args:
            task: Task type ('classification', 'detection', or 'segmentation')
            model_class: Model class to use (optional if model is provided)
            data_module_class: Data module class to use (optional if data_module is provided)
            config_path: Path to the configuration file (optional if config is provided)
            distributed: Whether to use distributed training with Ray (default: False)
            model_config_class: Optional config class for the model (default: None)
            config: Optional pre-loaded configuration (default: None)
            model: Optional pre-initialized model (default: None)
            data_module: Optional pre-initialized data module (default: None)
        """
        self.task = task
        self.model_class = model_class
        self.model_config_class = model_config_class
        self.data_module_class = data_module_class
        self.config_path = config_path
        self.config = config if config is not None else self._load_config()
        self.model = model
        self.data_module = data_module
        self.trainer = None
        self.distributed = distributed
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_model(self):
        """Initialize the model if not already initialized."""
        if self.model is None:
            if self.model_config_class is not None:
                model_config = self.model_config_class(**self.config['model'])
                self.model = self.model_class(model_config)
            else:
                self.model = self.model_class(**self.config['model'])
    
    def _init_data_module(self):
        """Initialize the data module if not already initialized."""
        if self.data_module is None:
            self.data_module = self.data_module_class(**self.config['data'])
    
    def _init_callbacks(self):
        """Initialize training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['training']['checkpoint_dir'],
            filename=f"{self.task}_{self.config['model']['name']}_best",
            monitor=self.config['training']['monitor_metric'],
            mode=self.config['training']['monitor_mode'],
            save_top_k=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config['training']['monitor_metric'],
            mode=self.config['training']['monitor_mode'],
            patience=self.config['training']['early_stopping_patience']
        )
        callbacks.append(early_stopping)
        
        # Ray train report callback (only for distributed training)
        if self.distributed:
            ray_report_callback = RayTrainReportCallback()
            callbacks.append(ray_report_callback)
        
        return callbacks
    
    def _init_trainer(self):
        """Initialize the PyTorch Lightning trainer."""
        callbacks = self._init_callbacks()
        
        # Configure trainer based on training mode
        if self.distributed:
            # Distributed training with Ray
            self.trainer = pl.Trainer(
                max_epochs=self.config['training']['max_epochs'],
                accelerator="auto",
                devices="auto",
                strategy=RayDDPStrategy(),
                plugins=[RayLightningEnvironment()],
                callbacks=callbacks,
                log_every_n_steps=self.config['training']['log_every_n_steps']
            )
            # Validate trainer configuration
            self.trainer = prepare_trainer(self.trainer)
        else:
            # Local training with PyTorch Lightning
            # In notebook environment, use single GPU to avoid multiprocessing issues
            self.trainer = pl.Trainer(
                max_epochs=self.config['training']['max_epochs'],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,  # Use single GPU
                callbacks=callbacks,
                log_every_n_steps=self.config['training']['log_every_n_steps']
            )
    
    def train(self):
        """Train the model using either local or distributed training."""
        # Initialize components
        self._init_model()
        self._init_data_module()
        self._init_trainer()
        
        if self.distributed:
            # Set up Ray cluster for distributed training
            try:
                from ray.util.spark import setup_ray_cluster
                setup_ray_cluster(
                    num_worker_nodes=self.config['ray']['num_workers'],
                    num_cpus_per_node=self.config['ray']['resources_per_worker']['CPU'],
                    num_gpus_per_node=self.config['ray']['resources_per_worker']['GPU'] if self.config['ray']['use_gpu'] else 0
                )
            except ImportError:
                raise ImportError("Ray on Spark is not installed. Please install it using: pip install ray[spark]")
            
            # Configure Ray training
            scaling_config = ScalingConfig(
                num_workers=self.config['ray']['num_workers'],
                use_gpu=self.config['ray']['use_gpu'],
                resources_per_worker=self.config['ray']['resources_per_worker']
            )
            
            run_config = RunConfig(
                storage_path=self.config['training']['checkpoint_dir'],
                name=f"{self.task}_{self.config['model']['name']}",
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute=self.config['training']['monitor_metric'],
                    checkpoint_score_order=self.config['training']['monitor_mode']
                )
            )
            
            # Define training function
            def train_func():
                self._init_model()
                self._init_data_module()
                self._init_trainer()
                self.trainer.fit(self.model, datamodule=self.data_module)
            
            # Create Ray trainer
            ray_trainer = TorchTrainer(
                train_func,
                scaling_config=scaling_config,
                run_config=run_config
            )
            
            # Start training
            result = ray_trainer.fit()
        else:
            # Local training
            self.trainer.fit(self.model, datamodule=self.data_module)
            result = type('Result', (), {'metrics': self.trainer.callback_metrics})
        
        # Log results to MLflow
        with mlflow.start_run(run_name=f"{self.task}_training") as run:
            # Log parameters
            mlflow.log_params(self.config['model'])
            mlflow.log_params(self.config['training'])
            
            # Log metrics
            mlflow.log_metrics(result.metrics)
            
            # Log model
            mlflow.pytorch.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=f"{self.task}_{self.config['model']['name']}"
            )
            
            # Log checkpoint
            if self.distributed:
                checkpoint_path = result.checkpoint.path
            else:
                checkpoint_path = self.trainer.checkpoint_callback.best_model_path
            mlflow.log_artifact(checkpoint_path)
        
        return result
    
    def tune(self, search_space: dict, num_trials: int = 20):
        """Run hyperparameter tuning using Ray Tune.
        
        Args:
            search_space: Dictionary defining the hyperparameter search space
            num_trials: Number of trials to run
        """
        if not self.distributed:
            raise ValueError("Hyperparameter tuning requires distributed training mode")
        
        # Initialize Ray
        try:
            from ray.util.spark import setup_ray_cluster
            setup_ray_cluster(
                num_worker_nodes=self.config['ray']['num_workers'],
                num_cpus_per_node=self.config['ray']['resources_per_worker']['CPU'],
                num_gpus_per_node=self.config['ray']['resources_per_worker']['GPU'] if self.config['ray']['use_gpu'] else 0
            )
        except ImportError:
            raise ImportError("Ray on Spark is not installed. Please install it using: pip install ray[spark]")
        
        # Configure scheduler
        scheduler = ASHAScheduler(
            metric=self.config['training']['monitor_metric'],
            mode=self.config['training']['monitor_mode'],
            max_t=self.config['training']['max_epochs'],
            grace_period=10,
            reduction_factor=2
        )
        
        # Define training function for tuning
        def train_func(config):
            # Update model config with trial parameters
            trial_config = self.config.copy()
            for key, value in config.items():
                if key in trial_config:
                    trial_config[key].update(value)
                else:
                    trial_config[key] = value
            
            # Train model
            result = self.train()
            
            # Report metrics to Ray Tune
            train.report({
                'val_loss': result.metrics['val_loss'],
                'val_map': result.metrics['val_map'] if self.task == 'detection' else None,
                'val_iou': result.metrics['val_iou'] if self.task == 'segmentation' else None
            })
        
        # Configure MLflow logger
        mlflow_logger = MLflowLoggerCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            registry_uri=mlflow.get_registry_uri(),
            experiment_name=mlflow.active_run().info.experiment_id
        )
        
        # Run hyperparameter tuning
        from ray import tune
        analysis = tune.run(
            train_func,
            config=search_space,
            num_samples=num_trials,
            scheduler=scheduler,
            resources_per_trial={
                'cpu': self.config['ray']['resources_per_worker']['CPU'],
                'gpu': self.config['ray']['resources_per_worker']['GPU'] if self.config['ray']['use_gpu'] else 0
            },
            callbacks=[mlflow_logger],
            verbose=1
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial(
            metric=self.config['training']['monitor_metric'],
            mode=self.config['training']['monitor_mode']
        )
        
        return best_trial.config
    
    def get_metrics(self):
        """Get training metrics from MLflow."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(mlflow.active_run().info.run_id)
        return run.data.metrics 