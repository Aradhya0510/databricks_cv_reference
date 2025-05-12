from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

class COCODataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        annotations: List[Dict],
        transform: Optional[A.Compose] = None,
        task: str = 'detection'
    ):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
        self.task = task
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)
        
        target = self.annotations[idx]
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=target['bboxes'])
            image = transformed['image']
            target['bboxes'] = transformed['bboxes']
            
        return image, target

def get_transforms(mode: str = 'train') -> A.Compose:
    """Get data augmentation transforms."""
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader with optimal settings for Databricks."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    ) 