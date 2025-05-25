import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor
import numpy as np
import os
from PIL import Image

from ..common.base_dataset import BaseVisionDataset
from ..common.base_datamodule import BaseVisionDataModule

class SemanticSegmentationDataset(BaseVisionDataset):
    """Dataset for semantic segmentation tasks."""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[A.Compose] = None,
        image_processor: Optional[AutoImageProcessor] = None,
        num_classes: int = 21,
        ignore_index: int = 255
    ):
        """Initialize semantic segmentation dataset.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing segmentation masks
            transform: Optional Albumentations transform pipeline
            image_processor: Optional HuggingFace image processor
            num_classes: Number of classes in the dataset
            ignore_index: Index to ignore in the mask
        """
        super().__init__(image_dir)
        self.mask_dir = mask_dir
        self.transform = transform or A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        self.image_processor = image_processor
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Get list of image and mask files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))])
        
        # Verify that we have matching image and mask files
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must match"
        for img, mask in zip(self.image_files, self.mask_files):
            assert os.path.splitext(img)[0] == os.path.splitext(mask)[0], f"Mismatched image and mask: {img} vs {mask}"

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - pixel_values: Image tensor
                - labels: Segmentation mask tensor
        """
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        
        # Process image
        if self.image_processor:
            # Process image with HuggingFace processor
            processed = self.image_processor(
                image,
                return_tensors="pt",
                do_resize=True,
                do_pad=True
            )
            image = processed['pixel_values'].squeeze(0)
            
            # Resize mask to match processed image size
            mask = Image.fromarray(mask)
            mask = mask.resize(
                (processed['pixel_values'].shape[-1], processed['pixel_values'].shape[-2]),
                Image.NEAREST
            )
            mask = np.array(mask)
        else:
            # Apply transforms
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()
        
        return {
            "pixel_values": image,
            "labels": mask
        }

class SemanticSegmentationDataModule(BaseVisionDataModule):
    """Data module for semantic segmentation tasks."""
    
    def __init__(
        self,
        train_image_dir: str,
        train_mask_dir: str,
        val_image_dir: str,
        val_mask_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None,
        image_processor: Optional[AutoImageProcessor] = None,
        num_classes: int = 21,
        ignore_index: int = 255
    ):
        """Initialize semantic segmentation data module.
        
        Args:
            train_image_dir: Directory containing training images
            train_mask_dir: Directory containing training masks
            val_image_dir: Directory containing validation images
            val_mask_dir: Directory containing validation masks
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_transform: Optional training transforms
            val_transform: Optional validation transforms
            image_processor: Optional HuggingFace image processor
            num_classes: Number of classes in the dataset
            ignore_index: Index to ignore in the mask
        """
        super().__init__(train_image_dir, val_image_dir, batch_size, num_workers)
        self.train_mask_dir = train_mask_dir
        self.val_mask_dir = val_mask_dir
        self.image_processor = image_processor
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Default transforms
        self.train_transform = train_transform or A.Compose([
            A.RandomResizedCrop(size=(512, 512)),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.Normalize(),
            ToTensorV2()
        ])
        
        self.val_transform = val_transform or A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = SemanticSegmentationDataset(
                self.train_image_dir,
                self.train_mask_dir,
                transform=self.train_transform,
                image_processor=self.image_processor,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index
            )
            self.val_dataset = SemanticSegmentationDataset(
                self.val_image_dir,
                self.val_mask_dir,
                transform=self.val_transform,
                image_processor=self.image_processor,
                num_classes=self.num_classes,
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