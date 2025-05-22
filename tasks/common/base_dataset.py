import json
import os
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

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
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image_id to annotations mapping
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            if ann['image_id'] not in self.image_annotations:
                self.image_annotations[ann['image_id']] = []
            self.image_annotations[ann['image_id']].append(ann)
        
        # Create image_id to filename mapping
        self.image_id_to_file = {
            img['id']: img['file_name']
            for img in self.coco_data['images']
        }
        
        # Create category mapping
        self.label_map = label_map or {
            cat['id']: cat['name']
            for cat in self.coco_data['categories']
        }
        
        # Get unique image IDs
        self.image_ids = list(self.image_annotations.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def load_image(self, image_id: int) -> np.ndarray:
        """Load and preprocess image.
        
        Args:
            image_id: ID of the image to load
            
        Returns:
            Preprocessed image as numpy array
        """
        image_path = os.path.join(self.image_dir, self.image_id_to_file[image_id])
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def get_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        """Get annotations for an image.
        
        Args:
            image_id: ID of the image
            
        Returns:
            List of annotations for the image
        """
        return self.image_annotations[image_id] 