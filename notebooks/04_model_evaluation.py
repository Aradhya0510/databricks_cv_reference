# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation and Prediction
# MAGIC 
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Load a trained model and COCO handler
# MAGIC 2. Make predictions on test images
# MAGIC 3. Convert predictions to COCO format
# MAGIC 4. Evaluate model performance
# MAGIC 5. Visualize predictions

# COMMAND ----------

import os
import yaml
import torch
import mlflow
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessorFast

from src.utils.coco_handler import COCOHandler
from src.tasks.detection.model import DetectionModel

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration and Model

# COMMAND ----------

# Load configuration
config_path = "/dbfs/FileStore/configs/detection.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load COCO handler
coco_handler = torch.load("/dbfs/FileStore/processed_data/coco_handler.pt")

# Load processor
processor = DetrImageProcessorFast.from_pretrained(
    "/dbfs/FileStore/processed_data/processor_config"
)

# Load model from MLflow
model_uri = f"models:/{config['model']['name']}/Production"
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Predictions

# COMMAND ----------

def predict_image(image_path: str) -> Dict:
    """Make predictions on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process outputs
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=config["model"]["confidence_threshold"]
    )[0]
    
    return {
        "boxes": results["boxes"].cpu().numpy(),
        "scores": results["scores"].cpu().numpy(),
        "labels": results["labels"].cpu().numpy()
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Predictions to COCO Format

# COMMAND ----------

def process_test_set(test_dir: str) -> List[Dict]:
    """Process all test images and convert predictions to COCO format."""
    all_predictions = []
    
    # Get test image IDs
    test_image_ids = [img["id"] for img in coco_handler.coco_data["images"]]
    
    for image_id in test_image_ids:
        # Get image info
        image_info = coco_handler.get_image_info(image_id)
        image_path = Path(test_dir) / image_info["file_name"]
        
        # Make prediction
        results = predict_image(str(image_path))
        
        # Convert to COCO format
        coco_predictions = coco_handler.convert_to_coco_format(
            image_id=image_id,
            boxes=results["boxes"],
            scores=results["scores"],
            labels=results["labels"],
            confidence_threshold=config["model"]["confidence_threshold"]
        )
        
        all_predictions.extend(coco_predictions)
    
    return all_predictions

# Process test set
test_predictions = process_test_set(config["data"]["test_path"])

# Save predictions
predictions_file = "/dbfs/FileStore/predictions/test_predictions.json"
coco_handler.save_predictions(test_predictions, predictions_file)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Model Performance

# COMMAND ----------

# Initialize COCO ground truth and predictions
coco_gt = COCO(config["data"]["annotation_file"])
coco_dt = coco_gt.loadRes(predictions_file)

# Initialize COCO evaluator
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Predictions

# COMMAND ----------

def visualize_prediction(image_id: int, image_dir: str):
    """Visualize model predictions on an image."""
    # Get image info
    image_info = coco_handler.get_image_info(image_id)
    image_path = Path(image_dir) / image_info["file_name"]
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Make prediction
    results = predict_image(str(image_path))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Plot predictions
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score > config["model"]["confidence_threshold"]:
            # Convert box to [x, y, width, height]
            bbox = coco_handler.convert_bbox_to_xywh(box)
            category = coco_handler.get_category_name(label)
            
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
            
            # Add label with score
            plt.text(
                bbox[0],
                bbox[1] - 5,
                f"{category}: {score:.2f}",
                color="red",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.7)
            )
    
    plt.axis("off")
    plt.title(f"Image ID: {image_id}")
    plt.show()

# Visualize predictions on a few test images
for image_id in np.random.choice(
    [img["id"] for img in coco_handler.coco_data["images"]],
    size=3,
    replace=False
):
    visualize_prediction(image_id, config["data"]["test_path"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC The model has been evaluated and predictions have been generated. You can now:
# MAGIC 
# MAGIC 1. Analyze model performance across different categories
# MAGIC 2. Identify common failure cases
# MAGIC 3. Fine-tune the model based on evaluation results
# MAGIC 4. Deploy the model for inference 