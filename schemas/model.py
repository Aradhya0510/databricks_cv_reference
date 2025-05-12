from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"

class ModelConfig(BaseModel):
    """Schema for model configuration."""
    model_name: str = Field(..., description="Name of the HuggingFace model")
    model_type: ModelType = Field(..., description="Type of model")
    pretrained: bool = Field(True, description="Whether to use pretrained weights")
    num_classes: Optional[int] = Field(None, description="Number of classes for classification")
    image_size: int = Field(224, description="Input image size")
    channels: int = Field(3, description="Number of input channels")
    
    class Config:
        use_enum_values = True

class ModelOutput(BaseModel):
    """Schema for model output."""
    predictions: Any = Field(..., description="Model predictions")
    logits: Optional[Any] = Field(None, description="Raw logits")
    features: Optional[Any] = Field(None, description="Feature maps")
    
    class Config:
        arbitrary_types_allowed = True 