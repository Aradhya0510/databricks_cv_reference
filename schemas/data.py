from typing import List, Optional, Union
from pydantic import BaseModel, Field
import numpy as np

class ImageData(BaseModel):
    """Schema for image data."""
    image: np.ndarray = Field(..., description="Image array in format (H, W, C)")
    image_id: str = Field(..., description="Unique identifier for the image")
    width: int = Field(..., gt=0, description="Image width")
    height: int = Field(..., gt=0, description="Image height")
    
    class Config:
        arbitrary_types_allowed = True

class Annotation(BaseModel):
    """Schema for annotation data."""
    bbox: List[float] = Field(..., min_items=4, max_items=4, description="Bounding box [x, y, w, h]")
    category_id: int = Field(..., ge=0, description="Category ID")
    segmentation: Optional[List[List[float]]] = Field(None, description="Segmentation mask")
    area: float = Field(..., gt=0, description="Area of the annotation")
    iscrowd: bool = Field(False, description="Whether the annotation is a crowd")

class DatasetItem(BaseModel):
    """Schema for a complete dataset item."""
    image: ImageData
    annotations: List[Annotation]
    
    class Config:
        arbitrary_types_allowed = True

class BatchData(BaseModel):
    """Schema for batched data."""
    images: np.ndarray = Field(..., description="Batch of images")
    targets: Union[List[Annotation], np.ndarray] = Field(..., description="Batch of targets")
    
    class Config:
        arbitrary_types_allowed = True 