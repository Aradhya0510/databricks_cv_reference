import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DetectionDataset(Dataset):
    """Dataset for object detection tasks."""
    
    def __init__(self, images: torch.Tensor, targets: Dict[str, torch.Tensor], transform: Optional[A.Compose] = None):
        """Initialize detection dataset.
        
        Args:
            images: Tensor of images
            targets: Dictionary of target tensors (boxes, labels, etc.)
            transform: Optional Albumentations transform pipeline
        """
        self.images = images
        self.targets = targets
        self.transform = transform or A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx]
        target = {k: v[idx] for k, v in self.targets.items()}
        
        if self.transform:
            transformed = self.transform(image=image, **target)
            image = transformed["image"]
            target = {k: v for k, v in transformed.items() if k != "image"}
            
        return {"pixel_values": image, **target}

class DetectionDataModule(pl.LightningDataModule):
    """Data module for object detection tasks."""
    
    def __init__(
        self,
        train_images: torch.Tensor,
        train_targets: Dict[str, torch.Tensor],
        val_images: torch.Tensor,
        val_targets: Dict[str, torch.Tensor],
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None
    ):
        """Initialize detection data module.
        
        Args:
            train_images: Training images tensor
            train_targets: Training targets dictionary
            val_images: Validation images tensor
            val_targets: Validation targets dictionary
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_transform: Optional training transforms
            val_transform: Optional validation transforms
        """
        super().__init__()
        self.train_images = train_images
        self.train_targets = train_targets
        self.val_images = val_images
        self.val_targets = val_targets
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Default transforms
        self.train_transform = train_transform or A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2()
        ])
        
        self.val_transform = val_transform or A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = DetectionDataset(
                self.train_images,
                self.train_targets,
                transform=self.train_transform
            )
            self.val_dataset = DetectionDataset(
                self.val_images,
                self.val_targets,
                transform=self.val_transform
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

    def _collate_fn(self, batch: list) -> Dict[str, torch.Tensor]:
        """Custom collate function for detection batches.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Dictionary of batched tensors
        """
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        targets = {
            k: torch.stack([x[k] for x in batch])
            for k in batch[0].keys()
            if k != "pixel_values"
        }
        return {"pixel_values": pixel_values, **targets} 