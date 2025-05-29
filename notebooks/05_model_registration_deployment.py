# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Registration and Deployment
# MAGIC 
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Register the trained DETR model in Unity Catalog
# MAGIC 2. Create a model serving endpoint
# MAGIC 3. Deploy the model for real-time inference
# MAGIC 4. Test the deployed endpoint
# MAGIC 5. Set up model versioning and staging

# COMMAND ----------

import os
import yaml
import mlflow
import torch
import numpy as np
import pandas as pd
import requests
import json
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration and Model

# COMMAND ----------

# Load best configuration
config_path = "/dbfs/FileStore/configs/detection_best.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load model from MLflow
model_uri = f"models:/detr_coco_detection/Production"
model = mlflow.pytorch.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model in Unity Catalog

# COMMAND ----------

# Set up Unity Catalog paths
CATALOG_NAME = "hive_metastore"
SCHEMA_NAME = "cv_models"
MODEL_NAME = "detr_coco_detection"

# Register model in Unity Catalog
with mlflow.start_run(run_name="model_registration") as run:
    # Log model to Unity Catalog
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
    )
    
    # Log model configuration
    mlflow.log_dict(config, "config.yaml")
    
    # Log evaluation metrics
    metrics = mlflow.get_run(run.info.run_id).data.metrics
    mlflow.log_metrics(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint

# COMMAND ----------

# Define endpoint configuration
endpoint_config = {
    "name": "detr-coco-endpoint",
    "config": {
        "served_models": [{
            "name": "detr-coco-model",
            "model_name": f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}",
            "model_version": "1",
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
    }
}

# Create endpoint
endpoint = mlflow.deployments.create_endpoint(
    name=endpoint_config["name"],
    config=endpoint_config["config"]
)

print(f"Endpoint created: {endpoint.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Endpoint with Sample Images

# COMMAND ----------

def prepare_image(image_path):
    """Prepare image for inference."""
    image = Image.open(image_path).convert('RGB')
    # Resize image
    image = image.resize((config["data"]["image_size"], config["data"]["image_size"]))
    # Convert to numpy array
    image = np.array(image)
    # Normalize
    image = image / 255.0
    image = (image - np.array(config["data"]["mean"])) / np.array(config["data"]["std"])
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def visualize_predictions(image_path, predictions):
    """Visualize model predictions."""
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Draw predictions
    for pred in predictions:
        box = pred["bbox"]
        score = pred["score"]
        label = pred["label"]
        
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "r-")
        plt.text(x1, y1, f"{label}: {score:.2f}", 
                color="white", bbox=dict(facecolor="red", alpha=0.5))
    
    plt.axis("off")
    plt.show()

# Test endpoint with sample images
test_images = [
    "/dbfs/FileStore/test_images/test1.jpg",
    "/dbfs/FileStore/test_images/test2.jpg",
    "/dbfs/FileStore/test_images/test3.jpg"
]

for image_path in test_images:
    # Prepare image
    image = prepare_image(image_path)
    
    # Make prediction
    response = mlflow.deployments.predict(
        endpoint_name=endpoint_config["name"],
        inputs={"instances": image.tolist()}
    )
    
    # Visualize predictions
    visualize_predictions(image_path, response["predictions"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up Model Versioning

# COMMAND ----------

# Get latest model version
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(MODEL_NAME)[0]

# Update model version stage
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version.version,
    stage="Production"
)

print(f"Model version {latest_version.version} moved to Production stage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Model registration and deployment completed:
# MAGIC - Model registered in Unity Catalog
# MAGIC - Serving endpoint created
# MAGIC - Endpoint tested with sample images
# MAGIC - Model versioning set up
# MAGIC 
# MAGIC Proceed to the next notebook to set up model monitoring. 