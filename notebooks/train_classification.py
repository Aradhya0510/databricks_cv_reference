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

from tasks.classification.datamodule import ClassificationDataModule
from tasks.classification.lightning_module import ClassificationModule
from orchestration.task_runner import run_hyperparameter_search

# COMMAND ----------

# Configure training parameters
config = {
    "task": "classification",
    "model_ckpt": "microsoft/resnet-50",
    "num_labels": 1000,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 10,
    "use_gpu": True
}

# COMMAND ----------

# Initialize MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/cv_classification")

# COMMAND ----------

# Define transforms
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# COMMAND ----------

# Initialize data module
data_module = ClassificationDataModule(
    train_image_dir="/dbfs/path/to/train/images",
    train_annotation_file="/dbfs/path/to/train/annotations.json",
    val_image_dir="/dbfs/path/to/val/images",
    val_annotation_file="/dbfs/path/to/val/annotations.json",
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    train_transform=train_transform,
    val_transform=val_transform
)

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name="classification_training") as run:
    # Log parameters
    mlflow.log_params(config)
    
    # Initialize model
    model = ClassificationModule(
        model_ckpt=config["model_ckpt"],
        num_labels=config["num_labels"],
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="/dbfs/path/to/checkpoints",
            filename="classification-{epoch:02d}-{val_loss:.2f}",
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
    trainer.fit(model, data_module)
    
    # Log final metrics
    mlflow.log_metrics(trainer.callback_metrics)
    
    # Save model
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="classification_model"
    )

# COMMAND ----------

# Optional: Run hyperparameter search
if False:  # Set to True to run hyperparameter search
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64])
    }
    
    best_config = run_hyperparameter_search(
        task="classification",
        model_ckpt=config["model_ckpt"],
        search_space=search_space,
        num_samples=10,
        use_gpu=config["use_gpu"]
    )
    
    print("Best configuration:", best_config) 