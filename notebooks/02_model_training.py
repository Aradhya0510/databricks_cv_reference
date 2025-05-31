# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training
# MAGIC 
# MAGIC This notebook handles model initialization and training for computer vision tasks.
# MAGIC 
# MAGIC ## Model Configuration Guide
# MAGIC 
# MAGIC ### 1. Model Architecture
# MAGIC 
# MAGIC The project supports various model architectures from Hugging Face. Configure your model in the YAML config:
# MAGIC 
# MAGIC ```yaml
# MAGIC model:
# MAGIC   # Model architecture
# MAGIC   model_name: "nvidia/mit-b0"  # Hugging Face model name
# MAGIC   num_classes: 19              # Number of classes
# MAGIC   pretrained: true            # Use pretrained weights
# MAGIC   
# MAGIC   # Training hyperparameters
# MAGIC   learning_rate: 1e-4         # Initial learning rate
# MAGIC   weight_decay: 1e-4          # Weight decay for regularization
# MAGIC   scheduler: "cosine"         # Learning rate scheduler
# MAGIC   epochs: 100                 # Number of training epochs
# MAGIC   
# MAGIC   # Task-specific settings
# MAGIC   task_type: "detection"      # or "classification" or "segmentation"
# MAGIC   segmentation_type: "semantic"  # for segmentation tasks
# MAGIC ```
# MAGIC 
# MAGIC ### 2. Training Configuration
# MAGIC 
# MAGIC Configure training settings in the YAML config:
# MAGIC 
# MAGIC ```yaml
# MAGIC training:
# MAGIC   # Batch and optimization settings
# MAGIC   batch_size: 32              # Batch size for training
# MAGIC   gradient_clip_val: 1.0      # Gradient clipping value
# MAGIC   early_stopping_patience: 10  # Early stopping patience
# MAGIC   
# MAGIC   # Checkpoint settings
# MAGIC   checkpoint_dir: "/Volumes/main/cv_ref/checkpoints"  # Checkpoint directory
# MAGIC   save_top_k: 3               # Number of best models to keep
# MAGIC   monitor: "val_loss"         # Metric to monitor
# MAGIC   mode: "min"                 # Monitor mode (min/max)
# MAGIC   
# MAGIC   # Logging settings
# MAGIC   log_every_n_steps: 50       # Log frequency
# MAGIC   log_metrics: true           # Log metrics to MLflow
# MAGIC   log_artifacts: true         # Log artifacts to MLflow
# MAGIC ```
# MAGIC 
# MAGIC ### 3. Available Models
# MAGIC 
# MAGIC #### Detection Models
# MAGIC - `facebook/detr-resnet-50`: DETR with ResNet-50 backbone
# MAGIC - `facebook/detr-resnet-101`: DETR with ResNet-101 backbone
# MAGIC - `microsoft/conditional-detr-resnet-50`: Conditional DETR
# MAGIC 
# MAGIC #### Classification Models
# MAGIC - `microsoft/resnet-50`: ResNet-50
# MAGIC - `microsoft/resnet-101`: ResNet-101
# MAGIC - `microsoft/vit-base-patch16-224`: Vision Transformer
# MAGIC 
# MAGIC #### Segmentation Models
# MAGIC - `nvidia/mit-b0`: SegFormer-B0
# MAGIC - `nvidia/mit-b1`: SegFormer-B1
# MAGIC - `facebook/mask2former-swin-base-coco`: Mask2Former

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

from src.utils.logging import setup_logger, get_metric_logger
from src.training.trainer import UnifiedTrainer
from src.tasks.detection.model import DetectionModel
from src.tasks.classification.model import ClassificationModel
from src.tasks.segmentation.model import SegmentationModel

# COMMAND ----------

# DBTITLE 1,Initialize Logging
logger = setup_logger(
    name="model_training",
    log_file="/Volumes/main/cv_ref/logs/training.log"
)

# COMMAND ----------

# DBTITLE 1,Load Configuration
def load_task_config(task: str):
    """Load configuration for the specified task."""
    config_path = f"/Volumes/main/cv_ref/configs/{task}_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# COMMAND ----------

# DBTITLE 1,Get Model Class
def get_model_class(task: str):
    """Get the appropriate model class for the task."""
    model_classes = {
        'detection': DetectionModel,
        'classification': ClassificationModel,
        'segmentation': SegmentationModel
    }
    return model_classes[task]

# COMMAND ----------

# DBTITLE 1,Initialize Model
def initialize_model(task: str, config: dict):
    """Initialize model for the specified task."""
    model_class = get_model_class(task)
    
    # Create model instance
    model = model_class(
        model_name=config['model']['model_name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        learning_rate=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay'],
        scheduler=config['model']['scheduler'],
        epochs=config['model']['epochs'],
        class_names=config['model'].get('class_names')
    )
    
    return model

# COMMAND ----------

# DBTITLE 1,Setup Trainer
def setup_trainer(task: str, config: dict, mlflow_logger):
    """Setup trainer for the specified task."""
    trainer = UnifiedTrainer(
        task=task,
        model_class=get_model_class(task),
        config=config,
        mlflow_logger=mlflow_logger,
        checkpoint_dir=config['training']['checkpoint_dir']
    )
    
    return trainer

# COMMAND ----------

# DBTITLE 1,Training Function
def train_model(
    task: str,
    config: dict,
    data_loaders: dict,
    mlflow_logger
):
    """Train model for the specified task."""
    # Initialize model
    model = initialize_model(task, config)
    
    # Setup trainer
    trainer = setup_trainer(task, config, mlflow_logger)
    
    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloader=data_loaders['train_loader'],
        val_dataloader=data_loaders['val_loader']
    )
    
    # Test model if test loader is available
    if data_loaders.get('test_loader'):
        test_results = trainer.test(
            model=model,
            test_dataloader=data_loaders['test_loader']
        )
        logger.info(f"Test results: {test_results}")
        
        # Save test results
        results_path = f"/Volumes/main/cv_ref/results/{task}_test_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(test_results, f)
    
    return model, trainer

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Train detection model
task = "detection"
config = load_task_config(task)

# Get MLflow logger
mlflow_logger = get_metric_logger(f"{task}_training")

# Load data (from previous notebook)
data_loaders = {
    'train_loader': train_loader,
    'val_loader': val_loader,
    'test_loader': test_loader
}

# Train model
model, trainer = train_model(
    task=task,
    config=config,
    data_loaders=data_loaders,
    mlflow_logger=mlflow_logger
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review training metrics
# MAGIC 2. Check model performance
# MAGIC 3. Proceed to hyperparameter tuning notebook 