import json
import os
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from ..common.base_dataset import BaseVisionDataset
from ..common.base_datamodule import BaseVisionDataModule

class ClassificationDataset(BaseVisionDataset):
    """Dataset for image classification using COCO format."""
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        label_map: Optional[Dict[int, str]] = None
    ):
        """Initialize classification dataset.
        
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO format annotation file
            transform: Optional Albumentations transform pipeline
            label_map: Optional mapping from category IDs to names
        """
        super().__init__(image_dir, annotation_file)
        self.transform = transform or A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Create category mapping
        self.label_map = label_map or {
            cat['id']: cat['name']
            for cat in self.coco_data['categories']
        }
        
        # Get unique image IDs
        self.image_ids = list(self.image_annotations.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - pixel_values: Image tensor
                - labels: Category ID tensor
        """
        image_id = self.image_ids[idx]
        image = self.load_image(image_id)
        
        # Get category ID (using the first annotation's category)
        category_id = self.get_annotations(image_id)[0]['category_id']
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'pixel_values': image,
            'labels': torch.tensor(category_id, dtype=torch.long)
        }

class ClassificationDataModule(BaseVisionDataModule):
    """Data module for image classification tasks."""
    
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
        label_map: Optional[Dict[int, str]] = None
    ):
        """Initialize classification data module.
        
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
        """
        super().__init__(train_image_dir, train_annotation_file, val_image_dir, val_annotation_file, batch_size, num_workers)
        self.label_map = label_map
        
        # Default transforms
        self.train_transform = train_transform or A.Compose([
            A.RandomResizedCrop(size=(224, 224)),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.Normalize(),
            ToTensorV2()
        ])
        
        self.val_transform = val_transform or A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = ClassificationDataset(
                self.train_image_dir,
                self.train_annotation_file,
                transform=self.train_transform,
                label_map=self.label_map
            )
            self.val_dataset = ClassificationDataset(
                self.val_image_dir,
                self.val_annotation_file,
                transform=self.val_transform,
                label_map=self.label_map
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