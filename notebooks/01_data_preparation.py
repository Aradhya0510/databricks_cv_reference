# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation
# MAGIC 
# MAGIC This notebook handles data preparation for computer vision tasks.
# MAGIC 
# MAGIC ## Unity Catalog Setup
# MAGIC 
# MAGIC The notebook expects data to be organized in the following structure within your Unity Catalog volume:
# MAGIC ```
# MAGIC /Volumes/cv_ref/datasets/coco_mini/
# MAGIC ├── data/
# MAGIC │   ├── train/
# MAGIC │   │   ├── images/
# MAGIC │   │   └── annotations.json
# MAGIC │   ├── val/
# MAGIC │   │   ├── images/
# MAGIC │   │   └── annotations.json
# MAGIC │   └── test/
# MAGIC │       ├── images/
# MAGIC │       └── annotations.json
# MAGIC ├── configs/
# MAGIC │   └── {task}_config.yaml
# MAGIC └── logs/
# MAGIC     └── data_prep.log
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import mlflow
import torch
from torch.utils.data import DataLoader
import yaml

# Add the project root to Python path
project_root = "/Workspace/Repos/Databricks_CV_ref"
sys.path.append(project_root)

from src.data.coco_handler import COCOHandler
from src.data.transforms import get_transforms
from src.utils.logging import setup_logger

# COMMAND ----------

# DBTITLE 1,Initialize Logging
# Get the Unity Catalog volume path from environment or use default
volume_path = os.getenv("UNITY_CATALOG_VOLUME", "/Volumes/cv_ref/datasets/coco_mini")
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="data_preparation",
    log_file=f"{log_dir}/data_prep.log"
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

# DBTITLE 1,Setup Data Handler
def setup_data_handler(config):
    """Initialize COCO handler and get data transformations."""
    handler = COCOHandler(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data']['test_path']
    )
    
    transforms = get_transforms(
        image_size=config['data']['image_size'],
        augment=config['data']['augment']
    )
    
    return handler, transforms

# COMMAND ----------

# DBTITLE 1,Prepare Dataset
def prepare_dataset(handler, transforms, config):
    """Create dataset instances for training, validation, and testing."""
    train_dataset = handler.get_dataset(
        split='train',
        transform=transforms['train']
    )
    
    val_dataset = handler.get_dataset(
        split='val',
        transform=transforms['val']
    )
    
    test_dataset = None
    if config['data']['test_path']:
        test_dataset = handler.get_dataset(
            split='test',
            transform=transforms['test']
        )
    
    return train_dataset, val_dataset, test_dataset

# COMMAND ----------

# DBTITLE 1,Create DataLoaders
def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    """Create DataLoader instances for the datasets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
    
    return train_loader, val_loader, test_loader

# COMMAND ----------

# DBTITLE 1,Main Data Preparation Function
def prepare_data(task: str):
    """Main function to prepare data for training."""
    # Load configuration
    config = load_task_config(task)
    
    # Setup data handler
    handler, transforms = setup_data_handler(config)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        handler, transforms, config
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # Log dataset statistics
    stats = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset) if test_dataset else 0,
        'num_classes': handler.num_classes,
        'class_names': handler.class_names
    }
    
    # Save statistics
    stats_path = f"{volume_path}/results/{task}_dataset_stats.yaml"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f)
    
    logger.info(f"Dataset statistics: {stats}")
    
    return train_loader, val_loader, test_loader

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Prepare data for detection task
task = "detection"

train_loader, val_loader, test_loader = prepare_data(task)

# Display a sample batch
sample_batch = next(iter(train_loader))
print("Sample batch keys:", sample_batch.keys())
print("Image shape:", sample_batch['image'].shape)
print("Target shape:", sample_batch['target'].shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review the prepared data
# MAGIC 2. Check data statistics and distributions
# MAGIC 3. Proceed to model training notebook 