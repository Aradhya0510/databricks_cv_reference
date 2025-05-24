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

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from trainer.ray_trainer import RayTrainer, cleanup_ray_train
from tasks.detection.data import DetectionDataModule

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

# Initialize Ray trainer
trainer = RayTrainer(
    task="detection",
    model_ckpt=config["model_ckpt"],
    num_workers=1,  # Single worker for local mode
    use_gpu=config["use_gpu"],
    local_mode=config["local_mode"],
    resources_per_worker=config["resources_per_worker"]
)

# Clean up any existing Ray sessions
cleanup_ray_train()

try:
    # Start training
    print("Starting local GPU training...")
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
    resources_per_trial={"cpu": 1, "gpu": 1}
)

print("Best hyperparameters found were: ", analysis.best_config)
""" 