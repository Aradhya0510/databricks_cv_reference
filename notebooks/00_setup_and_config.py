# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Setup and Configuration
# MAGIC 
# MAGIC This notebook sets up the environment and loads configurations for the computer vision tasks.
# MAGIC 
# MAGIC ## Configuration Guide
# MAGIC 
# MAGIC The project uses YAML configuration files to manage settings for different tasks. Here's how to configure your task:
# MAGIC 
# MAGIC ### 1. Task Configuration
# MAGIC 
# MAGIC Create a YAML file in the `configs` directory with the following structure:
# MAGIC 
# MAGIC ```yaml
# MAGIC # Model Configuration
# MAGIC model:
# MAGIC   model_name: "nvidia/mit-b0"  # Hugging Face model name
# MAGIC   num_classes: 19              # Number of classes
# MAGIC   pretrained: true            # Use pretrained weights
# MAGIC   learning_rate: 1e-4         # Initial learning rate
# MAGIC   weight_decay: 1e-4          # Weight decay for regularization
# MAGIC   scheduler: "cosine"         # Learning rate scheduler
# MAGIC   epochs: 100                 # Number of training epochs
# MAGIC   class_names: []             # Optional: List of class names
# MAGIC 
# MAGIC # Training Configuration
# MAGIC training:
# MAGIC   batch_size: 32              # Batch size for training
# MAGIC   num_workers: 4              # Number of data loading workers
# MAGIC   gradient_clip_val: 1.0      # Gradient clipping value
# MAGIC   early_stopping_patience: 10  # Early stopping patience
# MAGIC   checkpoint_dir: "checkpoints" # Directory for model checkpoints
# MAGIC 
# MAGIC # Data Configuration
# MAGIC data:
# MAGIC   train_path: "train"         # Path to training data
# MAGIC   val_path: "val"             # Path to validation data
# MAGIC   test_path: "test"           # Path to test data
# MAGIC   image_size: [512, 512]      # Input image size
# MAGIC   augment: true               # Use data augmentation
# MAGIC ```
# MAGIC 
# MAGIC ### 2. Task-Specific Settings
# MAGIC 
# MAGIC #### Detection Task
# MAGIC ```yaml
# MAGIC model:
# MAGIC   model_name: "facebook/detr-resnet-50"
# MAGIC   num_classes: 80  # COCO classes
# MAGIC   confidence_threshold: 0.5
# MAGIC   iou_threshold: 0.5
# MAGIC   max_detections: 100
# MAGIC ```
# MAGIC 
# MAGIC #### Classification Task
# MAGIC ```yaml
# MAGIC model:
# MAGIC   model_name: "microsoft/resnet-50"
# MAGIC   num_classes: 1000  # ImageNet classes
# MAGIC   dropout: 0.2
# MAGIC   mixup_alpha: 0.2
# MAGIC ```
# MAGIC 
# MAGIC #### Segmentation Task
# MAGIC ```yaml
# MAGIC model:
# MAGIC   model_name: "nvidia/mit-b0"
# MAGIC   num_classes: 19  # Cityscapes classes
# MAGIC   segmentation_type: "semantic"  # or "instance" or "panoptic"
# MAGIC   aux_loss_weight: 0.4
# MAGIC   mask_threshold: 0.5
# MAGIC ```
# MAGIC 
# MAGIC ### 3. Unity Catalog Volumes
# MAGIC 
# MAGIC The project uses Unity Catalog volumes for storing:
# MAGIC - Configuration files
# MAGIC - Model checkpoints
# MAGIC - Training logs
# MAGIC - Evaluation results
# MAGIC 
# MAGIC These are mounted at the following locations:
# MAGIC - Configs: `/Volumes/main/cv_ref/configs`
# MAGIC - Checkpoints: `/Volumes/main/cv_ref/checkpoints`
# MAGIC - Logs: `/Volumes/main/cv_ref/logs`
# MAGIC - Results: `/Volumes/main/cv_ref/results`

# COMMAND ----------

# MAGIC %pip install -r /Workspace/Repos/Databricks_CV_ref/requirements.txt

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import yaml

# Add the project root to Python path
project_root = "/Workspace/Repos/Databricks_CV_ref"
sys.path.append(project_root)

from src.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    get_default_config,
    save_config
)
from src.utils.logging import setup_logger, get_metric_logger

# COMMAND ----------

# DBTITLE 1,Initialize Logging
logger = setup_logger(
    name="cv_pipeline",
    log_file="/Volumes/main/cv_ref/logs/setup.log"
)

# COMMAND ----------

# DBTITLE 1,Load or Create Configuration
def setup_config(task: str, config_path: str = None):
    """Setup configuration for the specified task."""
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
    else:
        logger.info(f"Creating default configuration for {task}")
        config = get_default_config(task)
        
        # Save default config
        if config_path:
            save_config(config, config_path)
            logger.info(f"Saved default configuration to {config_path}")
    
    return config

# COMMAND ----------

# DBTITLE 1,Setup MLflow
def setup_mlflow(experiment_name: str):
    """Setup MLflow experiment."""
    mlflow_logger = get_metric_logger(experiment_name)
    logger.info(f"Initialized MLflow experiment: {experiment_name}")
    return mlflow_logger

# COMMAND ----------

# DBTITLE 1,Main Setup Function
def setup_pipeline(task: str, experiment_name: str):
    """Main setup function for the pipeline."""
    # Create necessary directories in Unity Catalog volumes
    os.makedirs("/Volumes/main/cv_ref/logs", exist_ok=True)
    os.makedirs("/Volumes/main/cv_ref/configs", exist_ok=True)
    os.makedirs("/Volumes/main/cv_ref/checkpoints", exist_ok=True)
    os.makedirs("/Volumes/main/cv_ref/results", exist_ok=True)
    
    # Setup configuration
    config_path = f"/Volumes/main/cv_ref/configs/{task}_config.yaml"
    config = setup_config(task, config_path)
    
    # Setup MLflow
    mlflow_logger = setup_mlflow(experiment_name)
    
    # Log configuration
    mlflow_logger.log_params(config)
    
    return config, mlflow_logger

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Setup detection pipeline
task = "detection"
experiment_name = "detection_pipeline"

config, mlflow_logger = setup_pipeline(task, experiment_name)

# Display configuration
print("Configuration:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review and modify the configuration as needed
# MAGIC 2. Proceed to data preparation notebook
# MAGIC 3. Check MLflow experiment setup 