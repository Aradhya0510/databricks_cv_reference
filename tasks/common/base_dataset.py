import json
import os
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pycocotools.coco import COCO

class BaseVisionDataset(Dataset):
    """Base dataset class for vision tasks using COCO format."""
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        label_map: Optional[Dict[int, str]] = None
    ):
        """Initialize base vision dataset.
        
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO format annotation file
            transform: Optional Albumentations transform pipeline
            label_map: Optional mapping from category IDs to names
        """
        self.image_dir = image_dir
        self.transform = transform or A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Initialize COCO API
        self.coco = COCO(annotation_file)
        
        # Get all image IDs that have annotations
        self.image_ids = list(sorted(self.coco.getImgIds()))
        
        # Create category mapping
        self.label_map = label_map or {
            cat['id']: cat['name']
            for cat in self.coco.loadCats(self.coco.getCatIds())
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    def load_image(self, image_id: int) -> np.ndarray:
        """Load and preprocess image.
        
        Args:
            image_id: ID of the image to load
            
        Returns:
            Preprocessed image as numpy array
        """
        # Get image info from COCO
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def get_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        """Get annotations for an image.
        
        Args:
            image_id: ID of the image
            
        Returns:
            List of annotations for the image
        """
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

    def get_image_info(self, image_id: int) -> Dict[str, Any]:
        """Get image information.
        
        Args:
            image_id: ID of the image
            
        Returns:
            Dictionary containing image information
        """
        return self.coco.loadImgs(image_id)[0]

    def get_category_info(self, category_id: int) -> Dict[str, Any]:
        """Get category information.
        
        Args:
            category_id: ID of the category
            
        Returns:
            Dictionary containing category information
        """
        return self.coco.loadCats(category_id)[0]

    def get_category_ids(self) -> List[int]:
        """Get all category IDs.
        
        Returns:
            List of category IDs
        """
        return self.coco.getCatIds()

    def get_image_ids_by_category(self, category_id: int) -> List[int]:
        """Get image IDs containing a specific category.
        
        Args:
            category_id: ID of the category
            
        Returns:
            List of image IDs containing the category
        """
        return self.coco.getImgIds(catIds=category_id) 