from ray import train, tune
import pytorch_lightning as pl
from typing import Dict, Any
from tasks.common.factory import make_module, get_task_config

def lightning_entry(config: Dict[str, Any]) -> None:
    """Ray-compatible entry point for training any vision task.
    
    Args:
        config: Dictionary containing:
            - task: One of "classification", "detection", "segmentation"
            - model_ckpt: Path to model checkpoint or HuggingFace model ID
            - lr: Learning rate
            - epochs: Number of training epochs
            - batch_size: Batch size
            - num_workers: Number of dataloader workers
            - use_gpu: Whether to use GPU
    """
    # Get task-specific default config
    task_config = get_task_config(config["task"])
    
    # Create module using factory
    module = make_module(
        task=config["task"],
        model_ckpt=config["model_ckpt"],
        **task_config
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu" if config.get("use_gpu", True) else "cpu",
        devices=train.torch.get_device_count() if config.get("use_gpu", True) else None,
        max_epochs=config["epochs"],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min"
            )
        ]
    )
    
    # Train model
    trainer.fit(module)
    
    # Log final metrics
    metrics = trainer.callback_metrics
    if train.is_session_enabled():
        train.report(metrics)

def run_hyperparameter_search(
    task: str,
    model_ckpt: str,
    search_space: Dict[str, Any],
    num_samples: int = 10,
    use_gpu: bool = True
) -> None:
    """Run hyperparameter search using Ray Tune.
    
    Args:
        task: One of "classification", "detection", "segmentation"
        model_ckpt: Path to model checkpoint or HuggingFace model ID
        search_space: Dictionary of hyperparameters to search
        num_samples: Number of trials to run
        use_gpu: Whether to use GPU
    """
    # Define search space
    config = {
        "task": task,
        "model_ckpt": model_ckpt,
        "use_gpu": use_gpu,
        **search_space
    }
    
    # Run hyperparameter search
    tuner = tune.Tuner(
        tune.with_parameters(lightning_entry),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples
        ),
        param_space=config
    )
    
    results = tuner.fit()
    print("Best trial config:", results.get_best_result().config)
    print("Best trial final validation loss:", results.get_best_result().metrics["val_loss"]) 