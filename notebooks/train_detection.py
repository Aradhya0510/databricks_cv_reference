# Databricks notebook source
# COMMAND ----------

# Install dependencies
!pip install -q pytorch_lightning ray[default] ray[tune] mlflow albumentations

# COMMAND ----------

import os
import sys
import mlflow
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from ray.train import ScalingConfig
from ray.air import RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.util.spark import setup_ray_cluster
import ray

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from trainer.ray_trainer import RayTrainer, cleanup_ray_train
from tasks.detection.data import DetectionDataModule
from tasks.detection.detr_module import DetrModule
from tasks.detection.lightning_module import DetectionConfig

# COMMAND ----------

# Configure training parameters
config = {
    "experiment_name": "/Users/aradhya.chouhan/experiments/detection",
    "run_name": "detection_local_gpu",
    "model_ckpt": "facebook/detr-resnet-50",  # Using DETR model
    "batch_size": 8,
    "num_workers": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "max_epochs": 10,
    "checkpoint_dir": "checkpoints",
    "model_path": "models/detection",
    "use_gpu": True,
    "local_mode": True,  # Enable local mode
    "resources_per_worker": {
        "CPU": 1,
        "GPU": 1
    }
}

# COMMAND ----------

# Initialize MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(config["experiment_name"])

# Define transforms
train_transform = A.Compose([
    A.RandomResizedCrop(800, 800),
    A.HorizontalFlip(),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(800, 800),
    A.Normalize(),
    ToTensorV2()
])

# Create data module
data_module = DetectionDataModule(
    train_image_dir="data/coco/train2017",
    train_annotation_file="data/coco/annotations/instances_train2017.json",
    val_image_dir="data/coco/val2017",
    val_annotation_file="data/coco/annotations/instances_val2017.json",
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    train_transform=train_transform,
    val_transform=val_transform
)

# Prepare data
data_module.setup()

# Update config with data loaders
config["train_loader"] = data_module.train_dataloader()
config["val_loader"] = data_module.val_dataloader()

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")

# Initialize Ray trainer with proper GPU configuration
trainer = RayTrainer(
    task="detection",
    model_ckpt=config["model_ckpt"],
    num_workers=1,  # Single worker for local mode
    use_gpu=torch.cuda.is_available(),  # Only use GPU if available
    local_mode=True,  # Force local mode
    resources_per_worker={
        "CPU": 1,
        "GPU": 1 if torch.cuda.is_available() else 0
    }
)

# Clean up any existing Ray sessions
cleanup_ray_train()

try:
    # Start training
    print("Starting local GPU training...")
    
    # Configure Ray for local training
    if not ray.is_initialized():
        ray.init(
            local_mode=True,
            ignore_reinit_error=True,
            num_gpus=1 if torch.cuda.is_available() else 0,
            num_cpus=1
        )
    
    # Set environment variables for GPU training
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    
    result = trainer.train(config)
    print("Training completed successfully!")
    print(f"Best validation loss: {result.metrics.get('val_loss', 'N/A')}")
    
except Exception as e:
    print(f"Training failed with error: {e}")
    raise
finally:
    # Clean up Ray resources
    cleanup_ray_train()

# Optional: Run hyperparameter search using Ray Tune
"""
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Define search space
search_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "weight_decay": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([4, 8, 16])
}

# Define scheduler
scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=config["max_epochs"],
    grace_period=1,
    reduction_factor=2
)

# Run hyperparameter search
analysis = tune.run(
    trainer.train,
    config=search_space,
    num_samples=10,
    scheduler=scheduler,
    resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0}
)

print("Best hyperparameters found were: ", analysis.best_config)
""" 

# COMMAND ----------

# Direct PyTorch Lightning Training (Bypassing Ray)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from tasks.detection.lightning_module import DetectionModule, DetectionConfig

# Create data module
data_module = DetectionDataModule(
    train_image_dir="data/coco/train2017",
    train_annotation_file="data/coco/annotations/instances_train2017.json",
    val_image_dir="data/coco/val2017",
    val_annotation_file="data/coco/annotations/instances_val2017.json",
    batch_size=8,
    num_workers=4,
    train_transform=train_transform,
    val_transform=val_transform
)

# Initialize MLflow logger
mlflow_logger = MLFlowLogger(
    experiment_name=config["experiment_name"],
    run_name="detection_lightning_direct"
)

# Setup callbacks
callbacks = [
    ModelCheckpoint(
        dirpath=config["checkpoint_dir"],
        filename="detection-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
]

# Create detection config
detection_config = DetectionConfig(
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    confidence_threshold=0.5,
    nms_threshold=0.5,
    max_detections=100
)

# Initialize trainer
trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=mlflow_logger,
    callbacks=callbacks,
    precision="16-mixed" if torch.cuda.is_available() else "32",
    gradient_clip_val=1.0
)

# Create and train model
model = DetectionModule(
    model_ckpt=config["model_ckpt"],
    config=detection_config
)

print("Starting direct PyTorch Lightning training...")
trainer.fit(model, data_module)

# Save the final model
final_model_path = os.path.join(config["model_path"], "final_model.pt")
trainer.save_checkpoint(final_model_path)
print(f"Model saved to {final_model_path}")

# Print best validation loss
best_val_loss = min(callbacks[0].best_model_score.item() if callbacks[0].best_model_score else float('inf'),
                   callbacks[1].best_score.item() if callbacks[1].best_score else float('inf'))
print(f"Best validation loss: {best_val_loss}")

# Create model
model = DetrModule(
    model_ckpt="facebook/detr-resnet-50",
    config=DetectionConfig(
        learning_rate=1e-4,
        weight_decay=1e-4,
        confidence_threshold=0.5,
        nms_threshold=0.5,
        max_detections=100
    )
)

# COMMAND ---------- 