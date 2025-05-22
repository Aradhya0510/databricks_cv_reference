# Databricks notebook source
# COMMAND ----------

# Install dependencies
!pip install -r ../requirements.txt

# COMMAND ----------

import os
import mlflow
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tasks.common.multitask import MultiTaskModule, MultiTaskConfig
from tasks.classification.datamodule import ClassificationDataModule
from tasks.segmentation.datamodule import SegmentationDataModule
from orchestration.task_runner import run_hyperparameter_search

# COMMAND ----------

# Configure training parameters
config = {
    "tasks": ["classification", "segmentation"],
    "model_checkpoints": {
        "classification": "microsoft/resnet-50",
        "segmentation": "nvidia/segformer-b0-finetuned-ade-512-512"
    },
    "task_weights": {
        "classification": 1.0,
        "segmentation": 1.0
    },
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 16,
    "num_workers": 4,
    "epochs": 10,
    "use_gpu": True,
    "ignore_index": 255
}

# COMMAND ----------

# Initialize MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/cv_multitask")

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

# Initialize data modules
classification_data = ClassificationDataModule(
    train_image_dir="/dbfs/path/to/classification/train/images",
    train_annotation_file="/dbfs/path/to/classification/train/annotations.json",
    val_image_dir="/dbfs/path/to/classification/val/images",
    val_annotation_file="/dbfs/path/to/classification/val/annotations.json",
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    train_transform=train_transform,
    val_transform=val_transform
)

segmentation_data = SegmentationDataModule(
    train_image_dir="/dbfs/path/to/segmentation/train/images",
    train_annotation_file="/dbfs/path/to/segmentation/train/annotations.json",
    val_image_dir="/dbfs/path/to/segmentation/val/images",
    val_annotation_file="/dbfs/path/to/segmentation/val/annotations.json",
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    train_transform=train_transform,
    val_transform=val_transform,
    ignore_index=config["ignore_index"]
)

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name="multitask_training") as run:
    # Log parameters
    mlflow.log_params(config)
    
    # Initialize multi-task configuration
    multitask_config = MultiTaskConfig(
        tasks=config["tasks"],
        model_checkpoints=config["model_checkpoints"],
        task_weights=config["task_weights"]
    )
    
    # Initialize model
    model = MultiTaskModule(
        config=multitask_config,
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="/dbfs/path/to/checkpoints",
            filename="multitask-{epoch:02d}-{val_loss:.2f}",
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
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu" if config["use_gpu"] else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=pl.loggers.MLFlowLogger(
            experiment_name=mlflow.active_run().info.experiment_name,
            run_id=mlflow.active_run().info.run_id
        )
    )
    
    # Train model
    trainer.fit(model, [classification_data, segmentation_data])
    
    # Log final metrics
    mlflow.log_metrics(trainer.callback_metrics)
    
    # Save model
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="multitask_model"
    )

# COMMAND ----------

# Optional: Run hyperparameter search
if False:  # Set to True to run hyperparameter search
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "task_weights": {
            "classification": tune.uniform(0.5, 1.5),
            "segmentation": tune.uniform(0.5, 1.5)
        }
    }
    
    best_config = run_hyperparameter_search(
        task="multitask",
        model_ckpt=config["model_checkpoints"],
        search_space=search_space,
        num_samples=10,
        use_gpu=config["use_gpu"]
    )
    
    print("Best configuration:", best_config) 