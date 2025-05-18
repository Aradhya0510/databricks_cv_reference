from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import io

class COCODataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        annotations: List[Dict],
        transform: Optional[A.Compose] = None,
        task: str = 'detection',
        use_binary: bool = False
    ):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
        self.task = task
        self.use_binary = use_binary
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Load image
        if self.use_binary:
            # Load from binary data
            image = Image.open(io.BytesIO(self.image_paths[idx])).convert('RGB')
        else:
            # Load from file path
            image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)
        
        # Get annotations for this image
        anns = self.annotations[idx]
        
        # Prepare target dictionary
        target = {
            'image_id': anns['image_id'],
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': []
        }
        
        # Process annotations
        for ann in anns['annotations']:
            target['boxes'].append(ann['bbox'])
            target['labels'].append(ann['category_id'])
            target['area'].append(ann['area'])
            target['iscrowd'].append(ann['iscrowd'])
        
        # Convert lists to numpy arrays for albumentations
        target['boxes'] = np.array(target['boxes'], dtype=np.float32)
        target['labels'] = np.array(target['labels'], dtype=np.int64)
        target['area'] = np.array(target['area'], dtype=np.float32)
        target['iscrowd'] = np.array(target['iscrowd'], dtype=np.int64)
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=target['boxes'])
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        else:
            # Convert to tensors if no transform
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
            target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
            target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.int64)
            
        return image, target

def get_transforms(mode: str = 'train') -> A.Compose:
    """Get data augmentation transforms."""
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(size=(512, 512)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

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