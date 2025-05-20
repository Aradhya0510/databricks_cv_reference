"""
Models package for object detection training.
"""

from models.classification import ClassificationModel
from models.config import get_model_config, get_train_config

__all__ = [
    'ClassificationModel',
    'get_model_config',
    'get_train_config'
] 