import json
import os
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pycocotools import mask as coco_mask

from ..common.base_dataset import BaseVisionDataset
from ..common.base_datamodule import BaseVisionDataModule

class SegmentationDataset(BaseVisionDataset):
    """Dataset for semantic segmentation using COCO format."""
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        label_map: Optional[Dict[int, str]] = None,
        ignore_index: int = 255
    ):
        """Initialize segmentation dataset.
        
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO format annotation file
            transform: Optional Albumentations transform pipeline
            label_map: Optional mapping from category IDs to names
            ignore_index: Index to use for ignored regions
        """
        super().__init__(image_dir, annotation_file, transform, label_map)
        self.ignore_index = ignore_index

    def _create_semantic_mask(
        self,
        image_size: Tuple[int, int],
        annotations: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Create semantic segmentation mask from COCO annotations.
        
        Args:
            image_size: (height, width) of the image
            annotations: List of COCO annotations for the image
            
        Returns:
            Semantic segmentation mask
        """
        mask = np.zeros(image_size, dtype=np.uint8)
        
        for ann in annotations:
            category_id = ann['category_id']
            
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    for polygon in ann['segmentation']:
                        polygon = np.array(polygon).reshape(-1, 2)
                        rr, cc = polygon_to_mask(polygon, image_size)
                        mask[rr, cc] = category_id
                else:
                    # RLE format
                    rle = ann['segmentation']
                    binary_mask = coco_mask.decode(rle)
                    mask[binary_mask > 0] = category_id
        
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - pixel_values: Image tensor
                - labels: Segmentation mask tensor
        """
        image_id = self.image_ids[idx]
        image = self.load_image(image_id)
        
        # Create semantic mask
        mask = self._create_semantic_mask(
            image.shape[:2],
            self.get_annotations(image_id)
        )
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'pixel_values': image,
            'labels': torch.tensor(mask, dtype=torch.long)
        }

class SegmentationDataModule(BaseVisionDataModule):
    """Data module for semantic segmentation tasks."""
    
    def __init__(
        self,
        train_image_dir: str,
        train_annotation_file: str,
        val_image_dir: str,
        val_annotation_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None,
        label_map: Optional[Dict[int, str]] = None,
        ignore_index: int = 255
    ):
        """Initialize segmentation data module.
        
        Args:
            train_image_dir: Directory containing training images
            train_annotation_file: Path to training COCO annotation file
            val_image_dir: Directory containing validation images
            val_annotation_file: Path to validation COCO annotation file
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_transform: Optional training transforms
            val_transform: Optional validation transforms
            label_map: Optional mapping from category IDs to names
            ignore_index: Index to use for ignored regions
        """
        super().__init__(
            train_image_dir,
            train_annotation_file,
            val_image_dir,
            val_annotation_file,
            batch_size,
            num_workers,
            train_transform,
            val_transform,
            label_map
        )
        self.ignore_index = ignore_index

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = SegmentationDataset(
                self.train_image_dir,
                self.train_annotation_file,
                transform=self.train_transform,
                label_map=self.label_map,
                ignore_index=self.ignore_index
            )
            self.val_dataset = SegmentationDataset(
                self.val_image_dir,
                self.val_annotation_file,
                transform=self.val_transform,
                label_map=self.label_map,
                ignore_index=self.ignore_index
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        ) 