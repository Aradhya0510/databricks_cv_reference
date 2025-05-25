from typing import Dict, Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BaseVisionDataModule(pl.LightningDataModule):
    """Base data module for vision tasks."""
    
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
        """Initialize base vision data module.
        
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
        super().__init__()
        self.train_image_dir = train_image_dir
        self.train_annotation_file = train_annotation_file
        self.val_image_dir = val_image_dir
        self.val_annotation_file = val_annotation_file
        self.batch_size = batch_size
        self.num_workers = num_workers
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