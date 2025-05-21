from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path

class COCODataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        task: str = 'detection'
    ):
        """
        Args:
            image_dir: Directory with all the images
            annotation_file: Path to the COCO format annotation file
            transform: Optional transform to be applied on images and boxes
            task: 'detection' or 'segmentation'
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.task = task
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
            
        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # Create image id to image info mapping
        self.img_to_info = {img['id']: img for img in self.coco_data['images']}
        
        # Get all image ids
        self.ids = list(self.img_to_info.keys())

        # Filter out images without annotations
        self.ids = [img_id for img_id in self.img_to_info.keys() 
                    if img_id in self.img_to_anns and len(self.img_to_anns[img_id]) > 0]
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Get image id
        img_id = self.ids[idx]
        
        # Get image info
        img_info = self.img_to_info[img_id]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Prepare target dictionary
        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': []
        }
        
        # Process annotations
        for ann in anns:
            target['boxes'].append(ann['bbox'])  # [x, y, width, height]
            target['labels'].append(ann['category_id'])
            target['area'].append(ann['area'])
            target['iscrowd'].append(ann['iscrowd'])
        
        # Convert lists to numpy arrays
        target['boxes'] = np.array(target['boxes'], dtype=np.float32)
        target['labels'] = np.array(target['labels'], dtype=np.int64)
        target['area'] = np.array(target['area'], dtype=np.float32)
        target['iscrowd'] = np.array(target['iscrowd'], dtype=np.int64)
        
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=target['boxes'],
                labels=target['labels']
            )
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
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
            A.RandomResizedCrop(size=(800, 800), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(size=(800, 800)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

def coco_collate_fn(batch):
    """
    Custom collate function for COCO dataset that properly batches images
    while keeping targets as a list of dictionaries.
    
    Args:
        batch: List of tuples (image, target)
        
    Returns:
        images: Tensor of shape [batch_size, channels, height, width]
        targets: List of dictionaries containing the annotations
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images into a single tensor
    images = torch.stack(images)
    
    return images, targets

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
        persistent_workers=True,
        collate_fn=coco_collate_fn
        # collate_fn=lambda x: tuple(zip(*x))  # Add collate_fn to handle the tuple returns
    ) 