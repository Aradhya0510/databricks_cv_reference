import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor
import numpy as np

from ..common.base_dataset import BaseVisionDataset
from ..common.base_datamodule import BaseVisionDataModule

class DetectionDataset(BaseVisionDataset):
    """Dataset for object detection tasks using COCO format."""
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        image_processor: Optional[AutoImageProcessor] = None,
        min_area: float = 0.0,
        max_area: float = float('inf'),
        min_visibility: float = 0.0,
        exclude_crowd: bool = True
    ):
        """Initialize detection dataset.
        
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO format annotation file
            transform: Optional Albumentations transform pipeline
            image_processor: Optional HuggingFace image processor
            min_area: Minimum area of bounding boxes to include
            max_area: Maximum area of bounding boxes to include
            min_visibility: Minimum visibility of objects to include
            exclude_crowd: Whether to exclude crowd annotations
        """
        super().__init__(image_dir, annotation_file)
        self.transform = transform or A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        self.image_processor = image_processor
        self.min_area = min_area
        self.max_area = max_area
        self.min_visibility = min_visibility
        self.exclude_crowd = exclude_crowd

    def _filter_annotations(self, anns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter annotations based on criteria.
        
        Args:
            anns: List of annotations to filter
            
        Returns:
            Filtered list of annotations
        """
        filtered_anns = []
        for ann in anns:
            # Skip crowd annotations if exclude_crowd is True
            if self.exclude_crowd and ann.get('iscrowd', 0) == 1:
                continue
                
            # Skip annotations with low visibility
            if ann.get('visibility', 1.0) < self.min_visibility:
                continue
                
            # Skip annotations outside area range
            if not (self.min_area <= ann['area'] <= self.max_area):
                continue
                
            filtered_anns.append(ann)
        return filtered_anns

    def _convert_bbox_to_xyxy(self, bbox: List[float]) -> List[float]:
        """Convert COCO bbox format [x,y,w,h] to [x1,y1,x2,y2].
        
        Args:
            bbox: Bounding box in COCO format [x,y,w,h]
            
        Returns:
            Bounding box in [x1,y1,x2,y2] format
        """
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - pixel_values: Image tensor
                - boxes: Bounding box tensor [N, 4]
                - labels: Category ID tensor [N]
                - area: Area tensor [N]
                - iscrowd: Crowd flag tensor [N]
                - image_id: Original image ID
                - image_size: Original image size (height, width)
        """
        # Get image id and load image using base class functionality
        img_id = self.image_ids[idx]
        image = self.load_image(img_id)
        
        # Get image info for size
        img_info = self.get_image_info(img_id)
        image_size = (img_info['height'], img_info['width'])
        
        # Get and filter annotations using base class functionality
        anns = self.get_annotations(img_id)
        anns = self._filter_annotations(anns)
        
        if not anns:
            # If no valid annotations, return empty tensors
            return {
                "pixel_values": torch.tensor(image),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.long),
                "image_id": torch.tensor(img_id),
                "image_size": torch.tensor(image_size)
            }
        
        # Prepare target dictionary
        target = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': []
        }
        
        # Process annotations
        for ann in anns:
            # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
            bbox = self._convert_bbox_to_xyxy(ann['bbox'])
            target['boxes'].append(bbox)
            target['labels'].append(ann['category_id'])
            target['area'].append(ann['area'])
            target['iscrowd'].append(ann.get('iscrowd', 0))
        
        # Convert lists to tensors
        target = {
            'boxes': torch.tensor(target['boxes'], dtype=torch.float32),
            'labels': torch.tensor(target['labels'], dtype=torch.long),
            'area': torch.tensor(target['area'], dtype=torch.float32),
            'iscrowd': torch.tensor(target['iscrowd'], dtype=torch.long),
            'image_id': torch.tensor(img_id),
            'image_size': torch.tensor(image_size)
        }
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, **target)
            image = transformed["image"]
            target = {k: v for k, v in transformed.items() if k != "image"}
        
        # Apply HuggingFace processor if provided
        if self.image_processor:
            processed = self.image_processor(
                image,
                return_tensors="pt",
                do_resize=True,
                do_pad=True
            )
            image = processed['pixel_values'].squeeze(0)
            
        return {"pixel_values": image, **target}

class DetectionDataModule(BaseVisionDataModule):
    """Data module for object detection tasks."""
    
    def __init__(
        self,
        train_image_dir: str,
        train_annotation_file: str,
        val_image_dir: str,
        val_annotation_file: str,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None,
        image_processor: Optional[AutoImageProcessor] = None,
        min_area: float = 0.0,
        max_area: float = float('inf'),
        min_visibility: float = 0.0,
        exclude_crowd: bool = True
    ):
        """Initialize detection data module.
        
        Args:
            train_image_dir: Directory containing training images
            train_annotation_file: Path to training COCO annotation file
            val_image_dir: Directory containing validation images
            val_annotation_file: Path to validation COCO annotation file
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_transform: Optional training transforms
            val_transform: Optional validation transforms
            image_processor: Optional HuggingFace image processor
            min_area: Minimum area of bounding boxes to include
            max_area: Maximum area of bounding boxes to include
            min_visibility: Minimum visibility of objects to include
            exclude_crowd: Whether to exclude crowd annotations
        """
        super().__init__(train_image_dir, train_annotation_file, val_image_dir, val_annotation_file, batch_size, num_workers)
        self.image_processor = image_processor
        self.min_area = min_area
        self.max_area = max_area
        self.min_visibility = min_visibility
        self.exclude_crowd = exclude_crowd
        
        # Default transforms
        self.train_transform = train_transform or A.Compose([
            A.RandomResizedCrop(800, 800),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2()
        ])
        
        self.val_transform = val_transform or A.Compose([
            A.Resize(800, 800),
            A.Normalize(),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = DetectionDataset(
                self.train_image_dir,
                self.train_annotation_file,
                transform=self.train_transform,
                image_processor=self.image_processor,
                min_area=self.min_area,
                max_area=self.max_area,
                min_visibility=self.min_visibility,
                exclude_crowd=self.exclude_crowd
            )
            self.val_dataset = DetectionDataset(
                self.val_image_dir,
                self.val_annotation_file,
                transform=self.val_transform,
                image_processor=self.image_processor,
                min_area=self.min_area,
                max_area=self.max_area,
                min_visibility=self.min_visibility,
                exclude_crowd=self.exclude_crowd
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for detection batches.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Dictionary of batched tensors
        """
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        targets = {
            k: [x[k] for x in batch]  # Don't stack boxes as they may have different sizes
            for k in batch[0].keys()
            if k != "pixel_values"
        }
        return {"pixel_values": pixel_values, **targets} 