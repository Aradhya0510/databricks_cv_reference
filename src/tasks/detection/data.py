from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from transformers import AutoFeatureExtractor
from PIL import Image

@dataclass
class DetectionDataConfig:
    """Configuration for detection data module."""
    data_path: str
    annotation_file: str
    image_size: int = 640
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 8
    num_workers: int = 4
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: int = 30
    brightness_contrast: float = 0.2
    hue_saturation: float = 0.2
    model_name: Optional[str] = None

class COCODetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[Union[A.Compose, Any]] = None,
        image_size: int = 640,
        model_name: Optional[str] = None
    ):
        self.root_dir = Path(root_dir)
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.image_size = image_size
        self.model_name = model_name
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Initialize feature extractor if using Hugging Face model
        if model_name:
            with torch.device('cpu'):
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    device_map=None,
                    torch_dtype=torch.float32
                )
        
        # Load class names and create category to index mapping
        self.class_names = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.cat_to_idx = {cat["id"]: idx for idx, cat in enumerate(self.coco.loadCats(self.coco.getCatIds()))}
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Prepare boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            # Convert to [x1, y1, x2, y2] format
            boxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ])
            # Convert category ID to zero-based index
            labels.append(self.cat_to_idx[ann['category_id']])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'class_labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        # Apply transforms
        if self.model_name:
            # Convert OpenCV image to PIL Image
            image = Image.fromarray(image)
            
            # Use Hugging Face feature extractor with consistent size
            inputs = self.feature_extractor(
                image,
                return_tensors="pt",
                do_resize=True,
                size={"height": self.image_size, "width": self.image_size},
                do_normalize=True
            )
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "labels": target
            }
        else:
            # Use Albumentations transforms
            if self.transform:
                transformed = self.transform(image=image, bboxes=boxes, labels=labels)
                image = transformed['image']
                target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                target['class_labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            
            return {
                "pixel_values": image,
                "labels": target
            }

class DetectionDataModule(pl.LightningDataModule):
    def __init__(self, config: Union[Dict[str, Any], DetectionDataConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = DetectionDataConfig(**config)
        self.config = config
        
        # Initialize feature extractor if using Hugging Face model
        if config.model_name:
            with torch.device('cpu'):
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    config.model_name,
                    device_map=None,
                    torch_dtype=torch.float32
                )
            self.transform = None
        else:
            # Define transforms
            self.train_transform = A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.HorizontalFlip(p=0.5 if config.horizontal_flip else 0),
                A.VerticalFlip(p=0.5 if config.vertical_flip else 0),
                A.Rotate(limit=config.rotation, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config.brightness_contrast,
                    contrast_limit=config.brightness_contrast,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=config.hue_saturation,
                    sat_shift_limit=config.hue_saturation,
                    val_shift_limit=config.hue_saturation,
                    p=0.5
                ),
                A.Normalize(
                    mean=config.mean,
                    std=config.std
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            
            self.val_transform = A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.Normalize(
                    mean=config.mean,
                    std=config.std
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = COCODetectionDataset(
                root_dir=self.config.data_path,
                annotation_file=self.config.annotation_file,
                transform=self.train_transform if not self.config.model_name else None,
                image_size=self.config.image_size,
                model_name=self.config.model_name
            )
            
            self.val_dataset = COCODetectionDataset(
                root_dir=self.config.data_path,
                annotation_file=self.config.annotation_file,
                transform=self.val_transform if not self.config.model_name else None,
                image_size=self.config.image_size,
                model_name=self.config.model_name
            )
        
        if stage == 'test':
            self.test_dataset = COCODetectionDataset(
                root_dir=self.config.data_path,
                annotation_file=self.config.annotation_file,
                transform=self.val_transform if not self.config.model_name else None,
                image_size=self.config.image_size,
                model_name=self.config.model_name
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    @staticmethod
    def _collate_fn(batch):
        images = []
        targets = []
        for sample in batch:
            images.append(sample["pixel_values"])
            targets.append(sample["labels"])
        
        # Stack images
        images = torch.stack(images)
        
        # Collate targets
        collated_targets = {
            "boxes": torch.cat([t["boxes"] for t in targets]),
            "class_labels": torch.cat([t["class_labels"] for t in targets]),
            "image_id": torch.cat([t["image_id"] for t in targets])
        }
        
        return {
            "pixel_values": images,
            "labels": collated_targets
        }
    
    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        return self.train_dataset.class_names 