# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training
# MAGIC 
# MAGIC This notebook handles model initialization and training for computer vision tasks.
# MAGIC 
# MAGIC ## Unity Catalog Setup
# MAGIC 
# MAGIC The notebook uses the following Unity Catalog volume structure:
# MAGIC ```
# MAGIC /Volumes/cv_ref/datasets/coco_mini/
# MAGIC ├── configs/
# MAGIC │   └── {task}_config.yaml
# MAGIC ├── checkpoints/
# MAGIC │   └── {task}_model/
# MAGIC ├── logs/
# MAGIC │   └── training.log
# MAGIC └── results/
# MAGIC     └── {task}_test_results.yaml
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import mlflow
import torch
import yaml

# Add the project root to Python path
project_root = "/Workspace/Repos/Databricks_CV_ref"
sys.path.append(project_root)

from src.models.detection import DetectionModel
from src.models.classification import ClassificationModel
from src.models.segmentation import SegmentationModel
from src.training.trainer import UnifiedTrainer
from src.utils.logging import setup_logger, get_metric_logger

# COMMAND ----------

# DBTITLE 1,Initialize Logging
# Get the Unity Catalog volume path from environment or use default
volume_path = os.getenv("UNITY_CATALOG_VOLUME", "/Volumes/cv_ref/datasets/coco_mini")
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="model_training",
    log_file=f"{log_dir}/training.log"
)

# COMMAND ----------

# DBTITLE 1,Load Configuration
def load_task_config(task: str):
    """Load task-specific configuration."""
    config_path = f"{volume_path}/configs/{task}_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# COMMAND ----------

# DBTITLE 1,Get Model Class
def get_model_class(task: str):
    """Get the appropriate model class based on task."""
    model_classes = {
        'detection': DetectionModel,
        'classification': ClassificationModel,
        'segmentation': SegmentationModel
    }
    
    if task not in model_classes:
        raise ValueError(f"Unsupported task: {task}")
    
    return model_classes[task]

# COMMAND ----------

# DBTITLE 1,Initialize Model
def initialize_model(task: str, config: dict):
    """Initialize model for the specified task."""
    model_class = get_model_class(task)
    
    model = model_class(
        model_name=config['model']['model_name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        learning_rate=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay'],
        scheduler=config['model']['scheduler']
    )
    
    return model

# COMMAND ----------

# DBTITLE 1,Setup Trainer
def setup_trainer(task: str, model, config: dict, mlflow_logger):
    """Setup trainer for the specified task."""
    trainer = UnifiedTrainer(
        task=task,
        model=model,
        config=config,
        mlflow_logger=mlflow_logger
    )
    
    return trainer

# COMMAND ----------

# DBTITLE 1,Training Function
def train_model(
    task: str,
    train_loader,
    val_loader,
    test_loader=None,
    experiment_name: str = None
):
    """Main function to train the model."""
    # Load configuration
    config = load_task_config(task)
    
    # Initialize MLflow logger
    if experiment_name:
        mlflow_logger = get_metric_logger(experiment_name)
    else:
        mlflow_logger = None
    
    # Initialize model
    model = initialize_model(task, config)
    
    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    
    # Setup trainer
    trainer = setup_trainer(task, model, config, mlflow_logger)
    
    # Train model
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['model']['epochs']
    )
    
    # Test model if test loader is available
    if test_loader:
        test_results = trainer.test(test_loader)
        
        # Save test results
        results_path = f"{volume_path}/results/{task}_test_results.yaml"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            yaml.dump(test_results, f)
        
        logger.info(f"Test results: {test_results}")
    
    return trainer

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Train detection model
task = "detection"
experiment_name = "detection_training"

# Initialize MLflow logger
mlflow_logger = get_metric_logger(experiment_name)

# Prepare data loaders (from previous notebook)
from notebook_01_data_preparation import prepare_data
train_loader, val_loader, test_loader = prepare_data(task)

# Train model
trainer = train_model(
    task=task,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    experiment_name=experiment_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review training metrics in MLflow
# MAGIC 2. Check model performance on test set
# MAGIC 3. Proceed to hyperparameter tuning notebook 