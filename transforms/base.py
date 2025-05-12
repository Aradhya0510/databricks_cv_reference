from typing import Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pydantic import BaseModel

class TransformConfig(BaseModel):
    """Base configuration for transforms."""
    image_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    p: float = 1.0

class BaseTransform:
    """Base class for image transformations."""
    
    def __init__(self, config: TransformConfig):
        self.config = config
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """Build the transform pipeline."""
        raise NotImplementedError
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Apply transforms to the input data."""
        return self.transform(**kwargs) 