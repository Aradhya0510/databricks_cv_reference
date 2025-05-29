# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Setup and Configuration
# MAGIC 
# MAGIC This notebook sets up the environment and configuration for training a DETR model on the COCO 2017 dataset.
# MAGIC 
# MAGIC ## Steps:
# MAGIC 1. Install required packages
# MAGIC 2. Configure MLflow experiment
# MAGIC 3. Set up Unity Catalog access
# MAGIC 4. Create training configuration for both local and distributed training

# COMMAND ----------

# MAGIC %pip install -r /Workspace/Repos/aradhya.chouhan/Databricks_CV_ref/requirements.txt
# MAGIC %pip install "ray[spark]"

# COMMAND ----------

import os
import yaml
import mlflow
from pathlib import Path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure MLflow Experiment

# COMMAND ----------

# Set up MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/aradhya.chouhan/experiments/detr_coco_detection")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Unity Catalog Access

# COMMAND ----------

# Define Unity Catalog paths
CATALOG_NAME = "catalog"
SCHEMA_NAME = "schema"
VOLUME_NAME = "volume"

# Construct data paths
DATA_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"
ANNOTATION_FILE = f"{DATA_PATH}/annotations.json"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Training Configuration

# COMMAND ----------

# Create configuration directory
os.makedirs("/dbfs/FileStore/configs", exist_ok=True)

# COMMAND ----------

# Create configuration for local training
local_config = {
    "model": {
        "name": "facebook/detr-resnet-50",
        "num_classes": 80,
        "confidence_threshold": 0.7,
        "iou_threshold": 0.5,
        "max_detections": 100
    },
    "data": {
        "train_path": f"{DATA_PATH}/train",
        "val_path": f"{DATA_PATH}/val",
        "test_path": f"{DATA_PATH}/test",
        "annotation_file": ANNOTATION_FILE,
        "batch_size": 32,
        "num_workers": 4,
        "image_size": [800, 800]
    },
    "training": {
        "max_epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "gradient_clip_val": 0.1,
        "early_stopping_patience": 10,
        "monitor_metric": "val_map",
        "monitor_mode": "max",
        "checkpoint_dir": "/dbfs/FileStore/checkpoints",
        "log_every_n_steps": 50
    }
}

# Save local configuration
with open("/dbfs/FileStore/configs/detection_local.yaml", "w") as f:
    yaml.dump(local_config, f)

# COMMAND ----------

# Create configuration for distributed training
distributed_config = local_config.copy()
distributed_config["ray"] = {
    "num_workers": 4,
    "use_gpu": True,
    "resources_per_worker": {
        "CPU": 4,
        "GPU": 1
    }
}

# Save distributed configuration
with open("/dbfs/FileStore/configs/detection_distributed.yaml", "w") as f:
    yaml.dump(distributed_config, f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Configuration

# COMMAND ----------

# Load and print configurations
print("Local Configuration:")
with open("/dbfs/FileStore/configs/detection_local.yaml", "r") as f:
    print(yaml.dump(yaml.safe_load(f), default_flow_style=False))

print("\nDistributed Configuration:")
with open("/dbfs/FileStore/configs/detection_distributed.yaml", "r") as f:
    print(yaml.dump(yaml.safe_load(f), default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC The configuration files have been created and saved. You can now:
# MAGIC 
# MAGIC 1. Use `detection_local.yaml` for local multi-GPU training
# MAGIC 2. Use `detection_distributed.yaml` for distributed training on the Spark cluster
# MAGIC 
# MAGIC Proceed to the next notebook to prepare the data for training. 