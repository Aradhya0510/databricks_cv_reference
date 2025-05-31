import os
import yaml
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any

def tuple_constructor(loader, node):
    """Custom constructor for YAML tuples."""
    value = loader.construct_sequence(node)
    return tuple(value)

# Register the tuple constructor
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, yaml.SafeLoader)

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str
    num_classes: int
    pretrained: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    epochs: int = 100
    class_names: Optional[List[str]] = None
    task_type: str = "detection"
    segmentation_type: Optional[str] = None

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    num_workers: int = 4
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    checkpoint_dir: str = "/Volumes/main/cv_ref/checkpoints"
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"
    log_every_n_steps: int = 50
    log_metrics: bool = True
    log_artifacts: bool = True

@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str
    val_path: str
    test_path: Optional[str] = None
    image_size: Union[List[int], tuple] = (512, 512)
    augment: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    augmentations: Optional[Dict[str, Any]] = None

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_default_config(task: str) -> Dict[str, Any]:
    """Get default configuration for the specified task."""
    # Base configuration
    config = {
        'model': asdict(ModelConfig(
            model_name="nvidia/mit-b0",
            num_classes=19,
            task_type=task
        )),
        'training': asdict(TrainingConfig()),
        'data': asdict(DataConfig(
            train_path="/Volumes/main/cv_ref/datasets/train",
            val_path="/Volumes/main/cv_ref/datasets/val"
        ))
    }
    
    # Task-specific configurations
    if task == "detection":
        config['model'].update({
            'model_name': "facebook/detr-resnet-50",
            'num_classes': 80,
            'confidence_threshold': 0.5,
            'iou_threshold': 0.5,
            'max_detections': 100
        })
    elif task == "classification":
        config['model'].update({
            'model_name': "microsoft/resnet-50",
            'num_classes': 1000,
            'dropout': 0.2,
            'mixup_alpha': 0.2
        })
    elif task == "segmentation":
        config['model'].update({
            'model_name': "nvidia/mit-b0",
            'num_classes': 19,
            'segmentation_type': "semantic",
            'aux_loss_weight': 0.4,
            'mask_threshold': 0.5
        })
    
    return config 