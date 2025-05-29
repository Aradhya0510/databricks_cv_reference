# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperparameter Tuning
# MAGIC 
# MAGIC This notebook demonstrates how to perform hyperparameter tuning for the DETR model using Ray Tune.
# MAGIC 
# MAGIC ## Steps:
# MAGIC 1. Define the hyperparameter search space
# MAGIC 2. Configure Ray Tune for distributed optimization
# MAGIC 3. Run hyperparameter tuning with MLflow tracking
# MAGIC 4. Analyze and visualize tuning results
# MAGIC 5. Select the best hyperparameters

# COMMAND ----------

import os
import yaml
import mlflow
import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.trainer import UnifiedTrainer
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Base Configuration

# COMMAND ----------

# Load distributed configuration as base
config_path = "/dbfs/FileStore/configs/detection_distributed.yaml"
with open(config_path, "r") as f:
    base_config = yaml.safe_load(f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Search Space

# COMMAND ----------

# Define hyperparameter search space
search_space = {
    "model": {
        "confidence_threshold": tune.uniform(0.5, 0.9),
        "iou_threshold": tune.uniform(0.3, 0.7),
        "max_detections": tune.choice([50, 100, 150])
    },
    "training": {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "warmup_epochs": tune.choice([3, 5, 7]),
        "gradient_clip_val": tune.uniform(0.1, 0.5)
    },
    "data": {
        "batch_size": tune.choice([16, 32, 64]),
        "image_size": tune.choice([[800, 800], [1024, 1024]])
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Trainer

# COMMAND ----------

# Initialize trainer in distributed mode
trainer = UnifiedTrainer(
    task="detection",
    model_class=DetectionModel,
    data_module_class=DetectionDataModule,
    config_path=config_path,
    distributed=True  # Hyperparameter tuning requires distributed mode
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Hyperparameter Tuning

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name="detr_hparam_tuning") as run:
    # Run hyperparameter tuning
    best_config = trainer.tune(
        search_space=search_space,
        num_trials=20
    )
    
    # Log best configuration
    mlflow.log_params(best_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Results

# COMMAND ----------

# Get tuning results
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("/Users/aradhya.chouhan/experiments/detr_coco_detection")
runs = client.search_runs(experiment.experiment_id)

# Convert results to DataFrame
results = []
for run in runs:
    if "trial" in run.data.tags:
        results.append({
            "trial": run.data.tags["trial"],
            "val_map": run.data.metrics.get("val_map", 0),
            "val_loss": run.data.metrics.get("val_loss", 0),
            **run.data.params
        })
df = pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Results

# COMMAND ----------

# Plot hyperparameter importance
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df.corr()["val_map"].sort_values(ascending=False).reset_index(),
    x="val_map",
    y="index"
)
plt.title("Hyperparameter Importance")
plt.xlabel("Correlation with Validation mAP")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Plot learning rate vs validation mAP
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="learning_rate",
    y="val_map",
    hue="batch_size",
    size="warmup_epochs"
)
plt.xscale("log")
plt.title("Learning Rate vs Validation mAP")
plt.xlabel("Learning Rate")
plt.ylabel("Validation mAP")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Best Configuration

# COMMAND ----------

# Update base configuration with best parameters
best_config_full = base_config.copy()
for key, value in best_config.items():
    if key in best_config_full:
        best_config_full[key].update(value)
    else:
        best_config_full[key] = value

# Save best configuration
best_config_path = "/dbfs/FileStore/configs/detection_best.yaml"
with open(best_config_path, "w") as f:
    yaml.dump(best_config_full, f)

print(f"Best configuration saved to: {best_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC The hyperparameter tuning is complete. You can now:
# MAGIC 
# MAGIC 1. Use the best configuration to train the final model
# MAGIC 2. Evaluate the model on the test set
# MAGIC 3. Register and deploy the model for inference 