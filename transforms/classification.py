from typing import Dict, Any
import albumentations as A
from .base import BaseTransform, TransformConfig

class ClassificationTransformConfig(TransformConfig):
    """Configuration for classification transforms."""
    use_augmentation: bool = True
    rotation_limit: int = 30
    horizontal_flip: bool = True
    vertical_flip: bool = False
    brightness_contrast: bool = True

class ClassificationTransform(BaseTransform):
    """Transforms for classification tasks."""
    
    def _build_transform(self) -> A.Compose:
        transforms = []
        
        if self.config.use_augmentation:
            transforms.extend([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=self.config.rotation_limit, p=0.5),
                A.HorizontalFlip(p=0.5 if self.config.horizontal_flip else 0),
                A.VerticalFlip(p=0.5 if self.config.vertical_flip else 0),
                A.RandomBrightnessContrast(
                    p=0.5 if self.config.brightness_contrast else 0
                ),
            ])
        
        transforms.extend([
            A.Resize(self.config.image_size, self.config.image_size),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2()
        ])
        
        return A.Compose(transforms) 