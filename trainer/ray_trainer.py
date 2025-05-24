from typing import Dict, Any, Optional
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.air import RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from tasks.common.factory import make_module
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import sys
from ray.util.spark import setup_ray_cluster

class RayTrainer:
    """Distributed trainer using Ray."""
    
    def __init__(
        self,
        task: str,
        model_ckpt: str,
        num_workers: int = 4,
        use_gpu: bool = True,
        local_mode: bool = False,
        resources_per_worker: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Initialize Ray trainer.
        
        Args:
            task: One of "classification", "detection", "segmentation"
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            num_workers: Number of Ray workers
            use_gpu: Whether to use GPU
            local_mode: Whether to run in local mode (single process) or distributed mode
            resources_per_worker: Resources per worker
            **kwargs: Additional arguments passed to make_module
        """
        self.task = task
        self.model_ckpt = model_ckpt
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.local_mode = local_mode
        self.resources_per_worker = resources_per_worker or {"CPU": 1}
        if use_gpu:
            self.resources_per_worker["GPU"] = 1
        self.kwargs = kwargs
    
    def train_func(self, config: Dict[str, Any]):
        """Training function executed on each worker."""
        import torch
        
        # Debug GPU availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
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
            
            # Determine accelerator and devices based on actual GPU availability
            has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
            accelerator = "gpu" if (self.use_gpu and has_gpu) else "cpu"
            devices = 1
            
            # For Ray Train, we should NOT use distributed strategies in Lightning
            # Ray handles the distribution, Lightning should run on single device per worker
            if accelerator == "gpu" and self.num_workers > 1:
                # In distributed Ray training, each worker uses one GPU
                strategy = "auto"
            else:
                strategy = "auto"
                
            print(f"Using accelerator: {accelerator}, devices: {devices}, strategy: {strategy}")
            
            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=config["max_epochs"],
                callbacks=callbacks,
                accelerator=accelerator,
                devices=devices,
                strategy=strategy,
                enable_progress_bar=False,
                logger=False,
                # Explicitly disable distributed backends that conflict with Ray
                sync_batchnorm=False
            )
            
            # Train model
            trainer.fit(model, train_loader, val_loader)
            
            # Log metrics
            if hasattr(trainer, 'callback_metrics'):
                metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if hasattr(v, 'item')}
                mlflow.log_metrics(metrics)
            
            # Save model
            if hasattr(model, 'save_to_mlflow'):
                model.save_to_mlflow(config["model_path"])
            else:
                # Fallback: save model state dict
                mlflow.pytorch.log_model(model, config["model_path"])
    
    def train(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start training in either local or distributed mode."""
        # Clean up any existing Ray Train sessions
        try:
            from ray.train._internal.session import get_session
            session = get_session()
            if session is not None and hasattr(session, 'finish'):
                session.finish()
        except:
            pass  # Ignore if no session exists
        
        # Shutdown and reinitialize Ray to ensure clean state
        if ray.is_initialized():
            ray.shutdown()
        
        if self.local_mode:
            return self._train_local(config)
        else:
            return self._train_distributed(config)
    
    def _train_local(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run training in local mode (single process)."""
        import torch
        
        # Debug GPU availability before Ray initialization
        print(f"System CUDA available: {torch.cuda.is_available()}")
        print(f"System CUDA device count: {torch.cuda.device_count()}")
        
        # Initialize Ray in local mode with explicit resource specification
        if not ray.is_initialized():
            # For local mode, explicitly specify resources
            if self.use_gpu and torch.cuda.is_available():
                ray.init(
                    local_mode=True, 
                    ignore_reinit_error=True,
                    num_gpus=torch.cuda.device_count(),  # Use actual GPU count
                    num_cpus=2
                )
            else:
                ray.init(
                    local_mode=True, 
                    ignore_reinit_error=True,
                    num_cpus=2
                )
        
        # Check Ray's view of resources
        print(f"Ray cluster resources: {ray.cluster_resources()}")
        print(f"Ray available resources: {ray.available_resources()}")
        
        # Adjust use_gpu based on actual availability
        has_ray_gpus = ray.cluster_resources().get('GPU', 0) > 0
        actual_use_gpu = self.use_gpu and torch.cuda.is_available() and has_ray_gpus
        print(f"Will use GPU: {actual_use_gpu} (requested: {self.use_gpu}, cuda_available: {torch.cuda.is_available()}, ray_gpus: {has_ray_gpus})")
        
        # Build Ray Train configs
        resources_per_worker = {"CPU": 1}
        if actual_use_gpu:
            resources_per_worker["GPU"] = 1
            
        scaling_config = ScalingConfig(
            num_workers=1,  # Single worker for local mode
            use_gpu=actual_use_gpu,
            resources_per_worker=resources_per_worker
        )

        run_config = RunConfig(
            callbacks=[MLflowLoggerCallback(
                tracking_uri="databricks",
                experiment_name=config["experiment_name"]
            )]
        )

        # Update config to reflect actual GPU usage
        config_copy = config.copy()
        config_copy["use_gpu"] = actual_use_gpu

        # Launch TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=self.train_func,
            train_loop_config=config_copy,
            scaling_config=scaling_config,
            run_config=run_config
        )
        
        try:
            result = trainer.fit()
            return result
        finally:
            # Clean up
            try:
                trainer.shutdown()
            except:
                pass
    
    def _train_distributed(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run training in distributed mode using Ray on Spark."""
        # Set environment variables for distributed training
        if self.use_gpu:
            # Set NCCL to use primary network interface only when using GPUs
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
        
        # Initialize Ray cluster on Spark
        if not ray.is_initialized():
            head_addr, remote_client_addr = setup_ray_cluster(
                max_worker_nodes=self.num_workers,
                num_cpus_worker_node=int(self.resources_per_worker.get("CPU", 1)),
                num_gpus_worker_node=int(self.resources_per_worker.get("GPU", 0)) if self.use_gpu else 0,
            )
            ray.init(address=remote_client_addr)

        # Build Ray Train configs
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker=self.resources_per_worker
        )

        run_config = RunConfig(
            callbacks=[MLflowLoggerCallback(
                tracking_uri="databricks",
                experiment_name=config["experiment_name"]
            )]
        )

        # Launch TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=self.train_func,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=run_config
        )
        return trainer.fit()

def cleanup_ray_train():
    """Clean up Ray Train sessions and Ray cluster."""
    print("Cleaning up Ray Train sessions...")
    
    # Try to clean up any existing Ray Train sessions
    try:
        from ray.train._internal.session import get_session, _shutdown
        session = get_session()
        if session is not None:
            print("Found existing Ray Train session, cleaning up...")
            if hasattr(session, 'finish'):
                session.finish()
            _shutdown()
            print("Ray Train session cleaned up")
    except Exception as e:
        print(f"Error cleaning up Ray Train session: {e}")
    
    # Shutdown Ray if initialized
    if ray.is_initialized():
        print("Shutting down Ray...")
        ray.shutdown()
        print("Ray shutdown complete")
    
    print("Cleanup complete") 