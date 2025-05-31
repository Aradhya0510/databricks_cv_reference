from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    model_name: str
    num_classes: int
    pretrained: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    epochs: int = 100
    batch_size: int = 32
    class_names: Optional[list] = None
    model_kwargs: Dict[str, Any] = None

@dataclass
class TrainingConfig:
    task: str
    distributed: bool = False
    num_workers: int = 4
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"

@dataclass
class DataConfig:
    train_path: str
    val_path: str
    test_path: Optional[str] = None
    image_size: tuple = (224, 224)
    augment: bool = True
    cache: bool = True
    num_workers: int = 4

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_default_config(task: str) -> Dict[str, Any]:
    """Get default configuration for a specific task."""
    base_config = {
        "model": {
            "model_name": "facebook/detr-resnet-50" if task == "detection" else 
                         "google/vit-base-patch16-224" if task == "classification" else
                         "nvidia/mit-b0",
            "num_classes": 80 if task == "detection" else 
                          1000 if task == "classification" else
                          19,
            "pretrained": True,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "epochs": 100,
            "batch_size": 32
        },
        "training": {
            "task": task,
            "distributed": False,
            "num_workers": 4,
            "accelerator": "gpu",
            "devices": 1,
            "precision": "16-mixed",
            "gradient_clip_val": 1.0,
            "early_stopping_patience": 10,
            "checkpoint_monitor": "val_loss",
            "checkpoint_mode": "min"
        },
        "data": {
            "train_path": f"data/{task}/train",
            "val_path": f"data/{task}/val",
            "test_path": f"data/{task}/test",
            "image_size": (224, 224),
            "augment": True,
            "cache": True,
            "num_workers": 4
        }
    }
    return base_config 