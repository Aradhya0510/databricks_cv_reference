# Databricks notebook source
# COMMAND ----------

# MAGIC %run ../setup/install_dependencies

# COMMAND ----------

import os
import mlflow
from typing import Dict, Any
from tasks.common.factory import make_module, get_task_config
from tasks.common.multitask import MultiTaskModule, MultiTaskConfig
from orchestration.task_runner import run_hyperparameter_search

# COMMAND ----------

# DBTITLE 1,Configure Training Parameters
# Task configuration
task = "classification"  # or "detection", "segmentation"
model_ckpt = "microsoft/resnet-50"  # or any other HuggingFace model ID

# Training configuration
config = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "scheduler_params": {
        "T_max": 100
    },
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 10,
    "use_gpu": True
}

# COMMAND ----------

# DBTITLE 1,Initialize MLflow
mlflow.set_tracking_uri("databricks")
experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/vision/{task}"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1,Load and Prepare Data
# Load your dataset here
# Example for classification:
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder("/dbfs/path/to/train", transform=transform)
val_dataset = ImageFolder("/dbfs/path/to/val", transform=transform)

# COMMAND ----------

# DBTITLE 1,Create Data Module
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class VisionDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

data_module = VisionDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"]
)

# COMMAND ----------

# DBTITLE 1,Create and Train Model
with mlflow.start_run(run_name=f"{task}_{model_ckpt.split('/')[-1]}"):
    # Log parameters
    mlflow.log_params(config)
    
    # Get task-specific config
    task_config = get_task_config(task)
    config.update(task_config)
    
    # Create module
    module = make_module(
        task=task,
        model_ckpt=model_ckpt,
        **config
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu" if config["use_gpu"] else "cpu",
        devices=1,
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
    trainer.fit(module, data_module)
    
    # Log final metrics
    metrics = trainer.callback_metrics
    mlflow.log_metrics(metrics)
    
    # Save model
    mlflow.pytorch.log_model(module, "model")

# COMMAND ----------

# DBTITLE 1,Optional: Run Hyperparameter Search
# Define search space
search_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "weight_decay": tune.loguniform(1e-6, 1e-4),
    "batch_size": tune.choice([16, 32, 64])
}

# Run hyperparameter search
run_hyperparameter_search(
    task=task,
    model_ckpt=model_ckpt,
    search_space=search_space,
    num_samples=10,
    use_gpu=config["use_gpu"]
) 