from importlib import import_module
from typing import Dict, Any, Type
from .base_module import BaseVisionModule

TASK_MAP = {
    "classification": "tasks.classification.lightning_module:ClassificationModule",
    "detection": "tasks.detection.lightning_module:DetectionModule",
    "segmentation": "tasks.segmentation.lightning_module:SegmentationModule",
}

def make_module(task: str, model_ckpt: str, **kwargs) -> BaseVisionModule:
    """Create a task-specific module using the factory pattern.
    
    Args:
        task: One of "classification", "detection", "segmentation"
        model_ckpt: Path to model checkpoint or HuggingFace model ID
        **kwargs: Additional arguments passed to the module constructor
        
    Returns:
        An instance of the appropriate task-specific module
        
    Raises:
        ValueError: If task is not supported
    """
    if task not in TASK_MAP:
        raise ValueError(f"Unknown task {task}. Supported tasks: {list(TASK_MAP.keys())}")
        
    module_path, cls_name = TASK_MAP[task].split(":")
    cls = getattr(import_module(module_path), cls_name)
    return cls(model_ckpt=model_ckpt, **kwargs)

def get_task_config(task: str) -> Dict[str, Any]:
    """Get default configuration for a specific task.
    
    Args:
        task: One of "classification", "detection", "segmentation"
        
    Returns:
        Dictionary of default configuration parameters
        
    Raises:
        ValueError: If task is not supported
    """
    if task not in TASK_MAP:
        raise ValueError(f"Unknown task {task}. Supported tasks: {list(TASK_MAP.keys())}")
        
    config_path = f"tasks.{task}.config"
    try:
        config_module = import_module(config_path)
        return config_module.DEFAULT_CONFIG
    except (ImportError, AttributeError):
        return {} 