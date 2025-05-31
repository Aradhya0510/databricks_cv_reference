# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation
# MAGIC 
# MAGIC This notebook handles dataset loading and preprocessing for computer vision tasks.
# MAGIC 
# MAGIC ## Data Configuration Guide
# MAGIC 
# MAGIC ### 1. Dataset Structure
# MAGIC 
# MAGIC The project expects datasets in COCO format, organized as follows:
# MAGIC 
# MAGIC ```
# MAGIC /Volumes/main/cv_ref/datasets/
# MAGIC ├── train/
# MAGIC │   ├── images/
# MAGIC │   └── annotations.json
# MAGIC ├── val/
# MAGIC │   ├── images/
# MAGIC │   └── annotations.json
# MAGIC └── test/
# MAGIC     ├── images/
# MAGIC     └── annotations.json
# MAGIC ```
# MAGIC 
# MAGIC ### 2. Data Configuration
# MAGIC 
# MAGIC Configure your dataset in the task's YAML config file:
# MAGIC 
# MAGIC ```yaml
# MAGIC data:
# MAGIC   # Dataset paths in Unity Catalog volumes
# MAGIC   train_path: "/Volumes/main/cv_ref/datasets/train"
# MAGIC   val_path: "/Volumes/main/cv_ref/datasets/val"
# MAGIC   test_path: "/Volumes/main/cv_ref/datasets/test"
# MAGIC   
# MAGIC   # Data processing settings
# MAGIC   image_size: [512, 512]      # Input image size
# MAGIC   augment: true               # Use data augmentation
# MAGIC   num_workers: 4              # Number of data loading workers
# MAGIC   pin_memory: true            # Use pinned memory for faster data transfer
# MAGIC   
# MAGIC   # Task-specific settings
# MAGIC   task_type: "detection"      # or "classification" or "segmentation"
# MAGIC   segmentation_type: "semantic"  # for segmentation tasks
# MAGIC ```
# MAGIC 
# MAGIC ### 3. Data Augmentation
# MAGIC 
# MAGIC The project supports various augmentation strategies. Configure them in the config:
# MAGIC 
# MAGIC ```yaml
# MAGIC data:
# MAGIC   augmentations:
# MAGIC     horizontal_flip: true
# MAGIC     vertical_flip: false
# MAGIC     rotation: 15              # Max rotation in degrees
# MAGIC     color_jitter:
# MAGIC       brightness: 0.2
# MAGIC       contrast: 0.2
# MAGIC       saturation: 0.2
# MAGIC       hue: 0.1
# MAGIC     random_crop: true
# MAGIC     random_resize: [0.8, 1.2]  # Scale range
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import mlflow
import yaml

# Add the project root to Python path
project_root = "/Workspace/Repos/Databricks_CV_ref"
sys.path.append(project_root)

from src.utils.logging import setup_logger, get_metric_logger
from src.data.coco_handler import COCOHandler
from src.data.dataset import BaseDataset
from src.data.transforms import get_transforms

# COMMAND ----------

# DBTITLE 1,Initialize Logging
logger = setup_logger(
    name="data_preparation",
    log_file="/Volumes/main/cv_ref/logs/data_prep.log"
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

# DBTITLE 1,Setup Data Handler
def setup_data_handler(task: str, config: dict):
    """Setup data handler for the specified task."""
    data_config = config['data']
    
    # Initialize COCO handler
    coco_handler = COCOHandler(
        train_path=data_config['train_path'],
        val_path=data_config['val_path'],
        test_path=data_config.get('test_path'),
        image_size=data_config['image_size']
    )
    
    # Get transforms
    transforms = get_transforms(
        task=task,
        image_size=data_config['image_size'],
        augment=data_config['augment'],
        augmentations=data_config.get('augmentations', {})
    )
    
    return coco_handler, transforms

# COMMAND ----------

# DBTITLE 1,Prepare Dataset
def prepare_dataset(task: str, coco_handler: COCOHandler, transforms: dict):
    """Prepare dataset for the specified task."""
    # Create dataset instances
    train_dataset = BaseDataset(
        task=task,
        data_handler=coco_handler,
        split='train',
        transform=transforms['train']
    )
    
    val_dataset = BaseDataset(
        task=task,
        data_handler=coco_handler,
        split='val',
        transform=transforms['val']
    )
    
    test_dataset = None
    if coco_handler.test_path:
        test_dataset = BaseDataset(
            task=task,
            data_handler=coco_handler,
            split='test',
            transform=transforms['test']
        )
    
    return train_dataset, val_dataset, test_dataset

# COMMAND ----------

# DBTITLE 1,Create DataLoaders
def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    config: dict
):
    """Create DataLoaders for training, validation, and testing."""
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data'].get('pin_memory', True)
        )
    
    return train_loader, val_loader, test_loader

# COMMAND ----------

# DBTITLE 1,Main Data Preparation Function
def prepare_data(task: str, config: dict):
    """Main function to prepare data for training."""
    # Setup data handler
    coco_handler, transforms = setup_data_handler(task, config)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        task, coco_handler, transforms
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # Log dataset statistics
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Save dataset statistics
    stats = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset) if test_dataset else 0,
        'num_classes': config['model']['num_classes'],
        'image_size': config['data']['image_size']
    }
    
    stats_path = f"/Volumes/main/cv_ref/results/{task}_dataset_stats.yaml"
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'coco_handler': coco_handler
    }

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Prepare detection data
task = "detection"
config = load_task_config(task)

data_loaders = prepare_data(task, config)

# Display sample batch
sample_batch = next(iter(data_loaders['train_loader']))
print("Sample batch keys:", sample_batch.keys())
print("Sample batch shapes:")
for key, value in sample_batch.items():
    if hasattr(value, 'shape'):
        print(f"{key}: {value.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review the prepared data
# MAGIC 2. Check data statistics and distributions
# MAGIC 3. Proceed to model training notebook 