# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training
# MAGIC 
# MAGIC This notebook demonstrates how to train the DETR model using both local and distributed training options.
# MAGIC 
# MAGIC ## Steps:
# MAGIC 1. Initialize the DETR model and data module
# MAGIC 2. Set up the unified trainer for local or distributed training
# MAGIC 3. Train the model with MLflow tracking
# MAGIC 4. Monitor training progress
# MAGIC 5. Save the trained model

# COMMAND ----------

import os
import yaml
import mlflow
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from src.training.trainer import UnifiedTrainer
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Choose training mode
TRAINING_MODE = "distributed"  # or "local"

# Load appropriate configuration
config_path = f"/dbfs/FileStore/configs/detection_{TRAINING_MODE}.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Trainer

# COMMAND ----------

# Initialize trainer with appropriate mode
trainer = UnifiedTrainer(
    task="detection",
    model_class=DetectionModel,
    data_module_class=DetectionDataModule,
    config_path=config_path,
    distributed=(TRAINING_MODE == "distributed")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name=f"detr_training_{TRAINING_MODE}") as run:
    # Log parameters
    mlflow.log_params(config["model"])
    mlflow.log_params(config["training"])
    if TRAINING_MODE == "distributed":
        mlflow.log_params(config["ray"])
    
    # Train model
    result = trainer.train()
    
    # Log metrics
    mlflow.log_metrics(result.metrics)
    
    # Log model
    mlflow.pytorch.log_model(
        trainer.model,
        artifact_path="model",
        registered_model_name=f"detr_coco_detection_{TRAINING_MODE}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor Training Progress

# COMMAND ----------

# Get training metrics
metrics = trainer.get_metrics()

# Plot training and validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(metrics["train_loss"], label="Training Loss")
plt.plot(metrics["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Plot validation mAP
plt.subplot(1, 2, 2)
plt.plot(metrics["val_map"], label="Validation mAP")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("Validation mAP")
plt.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Model Checkpoint

# COMMAND ----------

# Save model checkpoint
checkpoint_dir = "/dbfs/FileStore/models"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = f"{checkpoint_dir}/detr_coco_detection_{TRAINING_MODE}.ckpt"

if TRAINING_MODE == "distributed":
    # For distributed training, the checkpoint is already saved by Ray
    checkpoint_path = result.checkpoint.path
else:
    # For local training, save the best model
    torch.save(trainer.model.state_dict(), checkpoint_path)

print(f"Model checkpoint saved to: {checkpoint_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC The model has been trained and saved. You can now:
# MAGIC 
# MAGIC 1. Proceed to hyperparameter tuning to optimize model performance
# MAGIC 2. Evaluate the model on the test set
# MAGIC 3. Register and deploy the model for inference 