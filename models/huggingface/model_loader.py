from typing import Optional, Dict, Any
from transformers import AutoModelForImageClassification, AutoModelForObjectDetection, AutoModelForSemanticSegmentation
from ..base import BaseModel
from ...schemas.model import ModelConfig, ModelType
import torch.nn as nn

class HuggingFaceModelLoader:
    """Loader for HuggingFace computer vision models."""
    
    MODEL_MAPPING = {
        ModelType.CLASSIFICATION: AutoModelForImageClassification,
        ModelType.DETECTION: AutoModelForObjectDetection,
        ModelType.SEGMENTATION: AutoModelForSemanticSegmentation
    }
    
    @classmethod
    def load_model(cls, config: ModelConfig) -> nn.Module:
        """Load a HuggingFace model based on configuration."""
        model_class = cls.MODEL_MAPPING.get(config.model_type)
        if not model_class:
            raise ValueError(f"Unsupported model type: {config.model_type}")
            
        model = model_class.from_pretrained(
            config.model_name,
            num_labels=config.num_classes if config.model_type == ModelType.CLASSIFICATION else None,
            pretrained=config.pretrained
        )
        
        return model
    
    @classmethod
    def get_available_models(cls, model_type: ModelType) -> Dict[str, Any]:
        """Get available models for a specific task."""
        # This would typically query the HuggingFace API
        # For now, return a static list
        return {
            ModelType.CLASSIFICATION: [
                "google/vit-base-patch16-224",
                "microsoft/resnet-50",
                "facebook/convnext-base"
            ],
            ModelType.DETECTION: [
                "facebook/detr-resnet-50",
                "microsoft/table-transformer-detection"
            ],
            ModelType.SEGMENTATION: [
                "nvidia/segformer-b0-finetuned-ade-512-512",
                "facebook/mask2former-swin-base-coco-instance"
            ]
        } 