# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation
# MAGIC 
# MAGIC This notebook demonstrates how to prepare COCO format data for model training, using the COCO handler for format conversion and model-specific processors for data preprocessing.
# MAGIC 
# MAGIC ## Steps:
# MAGIC 1. Load and validate COCO format data
# MAGIC 2. Set up model-specific image processing
# MAGIC 3. Prepare data for training
# MAGIC 4. Save processed data

# COMMAND ----------

import os
import json
import yaml
import mlflow
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
import numpy as np
from pathlib import Path
from transformers import DetrImageProcessor, DetrImageProcessorFast

from src.tasks.detection.data import DetectionDataModule, DetectionDataConfig
from src.utils.coco_handler import COCOHandler

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load configuration
config_path = "/dbfs/FileStore/configs/detection.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize COCO Handler

# COMMAND ----------

# Initialize COCO handler
coco_handler = COCOHandler(
    annotation_file=config["data"]["annotation_file"]
)

# Print dataset statistics
print(f"Number of images: {len(coco_handler.coco_data['images'])}")
print(f"Number of annotations: {len(coco_handler.coco_data['annotations'])}")
print(f"Number of categories: {len(coco_handler.coco_data['categories'])}")
print("\nCategories:")
for cat in coco_handler.coco_data["categories"]:
    print(f"- {cat['name']} (id: {cat['id']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Model-Specific Processing

# COMMAND ----------

# Initialize DETR image processor
processor = DetrImageProcessorFast.from_pretrained(
    config["model"]["name"],
    size=config["data"]["image_size"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for Training

# COMMAND ----------

def prepare_sample(image_id: int, image_dir: str) -> Dict:
    """Prepare a single sample for training."""
    # Get image info and annotations
    image_info = coco_handler.get_image_info(image_id)
    annotations = coco_handler.get_annotations(image_id)
    
    # Load image
    image_path = Path(image_dir) / image_info["file_name"]
    image = Image.open(image_path).convert("RGB")
    
    # Prepare target
    target = coco_handler.prepare_target(annotations)
    
    # Process image and target
    inputs = processor(
        images=image,
        annotations=target,
        return_tensors="pt"
    )
    
    # Add image ID
    inputs["image_id"] = image_id
    
    return inputs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Sample

# COMMAND ----------

def visualize_sample(image_id: int, image_dir: str):
    """Visualize a sample with annotations."""
    # Get image info and annotations
    image_info = coco_handler.get_image_info(image_id)
    annotations = coco_handler.get_annotations(image_id)
    
    # Load image
    image_path = Path(image_dir) / image_info["file_name"]
    image = Image.open(image_path).convert("RGB")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Plot annotations
    for ann in annotations:
        bbox = ann["bbox"]  # [x, y, width, height]
        category = coco_handler.get_category_name(ann["category_id"])
        
        # Create rectangle
        rect = plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            fill=False,
            edgecolor="red",
            linewidth=2
        )
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(
            bbox[0],
            bbox[1] - 5,
            category,
            color="red",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7)
        )
    
    plt.axis("off")
    plt.title(f"Image ID: {image_id}")
    plt.show()

# Visualize a sample
sample_image_id = coco_handler.coco_data["images"][0]["id"]
visualize_sample(sample_image_id, config["data"]["train_path"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Processed Data

# COMMAND ----------

# Create directories for processed data
processed_dir = "/dbfs/FileStore/processed_data"
os.makedirs(processed_dir, exist_ok=True)

# Save processor configuration
processor.save_pretrained(f"{processed_dir}/processor_config")

# Save COCO handler state
torch.save(coco_handler, f"{processed_dir}/coco_handler.pt")

print(f"Processed data saved to: {processed_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC The data has been prepared and saved. You can now:
# MAGIC 
# MAGIC 1. Use the processed data for model training
# MAGIC 2. Use the COCO handler for evaluation and prediction
# MAGIC 3. Visualize model predictions 