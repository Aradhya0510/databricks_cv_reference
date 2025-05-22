# Databricks notebook source
# COMMAND ----------

# Install dependencies
!pip install -r ../requirements.txt

# COMMAND ----------

import os
import mlflow
import torch
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tasks.segmentation.datamodule import SegmentationDataModule
from trainer.ray_trainer import RayTrainer

# COMMAND ----------

# Configure training parameters
config = {
    "task": "segmentation",
    "model_ckpt": "nvidia/mit-b0",
    "num_labels": 19,  # Cityscapes has 19 classes
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 8,  # Smaller batch size for segmentation
    "num_workers": 4,
    "epochs": 10,
    "use_gpu": True
}

# COMMAND ----------

# Initialize MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/cv_segmentation")

# COMMAND ----------

# Define transforms
train_transform = A.Compose([
    A.RandomResizedCrop(512, 512),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

# COMMAND ----------

# Initialize data module
data_module = SegmentationDataModule(
    train_image_dir="/dbfs/path/to/train/images",
    train_mask_dir="/dbfs/path/to/train/masks",
    val_image_dir="/dbfs/path/to/val/images",
    val_mask_dir="/dbfs/path/to/val/masks",
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    train_transform=train_transform,
    val_transform=val_transform
)

# COMMAND ----------

# Initialize Ray trainer
trainer = RayTrainer(
    task=config["task"],
    model_ckpt=config["model_ckpt"],
    num_workers=config["num_workers"],
    use_gpu=config["use_gpu"],
    num_labels=config["num_labels"],
    lr=config["lr"],
    weight_decay=config["weight_decay"]
)

# COMMAND ----------

# Configure training
training_config = {
    "experiment_name": mlflow.active_run().info.experiment_name,
    "run_name": "segmentation_training",
    "max_epochs": config["epochs"],
    "checkpoint_dir": "/dbfs/path/to/checkpoints",
    "model_path": "/dbfs/path/to/model",
    "train_loader": data_module.train_dataloader(),
    "val_loader": data_module.val_dataloader()
}

# COMMAND ----------

# Start training
result = trainer.train(training_config)

# COMMAND ----------

# Optional: Run hyperparameter search
if False:  # Set to True to run hyperparameter search
    from ray import tune
    
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([4, 8, 16])
    }
    
    # Initialize trainer with search space
    trainer = RayTrainer(
        task=config["task"],
        model_ckpt=config["model_ckpt"],
        num_workers=config["num_workers"],
        use_gpu=config["use_gpu"],
        num_labels=config["num_labels"]
    )
    
    # Run hyperparameter search
    analysis = trainer.train(
        training_config,
        tune_config={
            "num_samples": 10,
            "search_space": search_space
        }
    )
    
    print("Best configuration:", analysis.best_config) 