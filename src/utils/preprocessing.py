import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_detection_transforms(
    image_size: Tuple[int, int],
    is_training: bool = True
) -> A.Compose:
    """Get transforms for detection task.
    
    Args:
        image_size: Target image size (height, width)
        is_training: Whether to use training transforms
    
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    if is_training:
        return A.Compose([
            A.RandomResizedCrop(
                height=image_size[0],
                width=image_size[1],
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
    else:
        return A.Compose([
            A.Resize(
                height=image_size[0],
                width=image_size[1]
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))

def get_segmentation_transforms(
    image_size: Tuple[int, int],
    is_training: bool = True
) -> A.Compose:
    """Get transforms for segmentation task.
    
    Args:
        image_size: Target image size (height, width)
        is_training: Whether to use training transforms
    
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    if is_training:
        return A.Compose([
            A.RandomResizedCrop(
                height=image_size[0],
                width=image_size[1],
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(
                height=image_size[0],
                width=image_size[1]
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

def preprocess_detection_image(
    image: Union[str, np.ndarray],
    image_size: Tuple[int, int],
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Preprocess image for detection.
    
    Args:
        image: Input image path or array
        image_size: Target image size (height, width)
        device: Device to move tensor to
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get transforms
    transform = get_detection_transforms(image_size, is_training=False)
    
    # Apply transforms
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    return image_tensor

def preprocess_segmentation_image(
    image: Union[str, np.ndarray],
    image_size: Tuple[int, int],
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Preprocess image for segmentation.
    
    Args:
        image: Input image path or array
        image_size: Target image size (height, width)
        device: Device to move tensor to
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get transforms
    transform = get_segmentation_transforms(image_size, is_training=False)
    
    # Apply transforms
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    return image_tensor

def postprocess_detection_predictions(
    predictions: Dict[str, torch.Tensor],
    original_size: Tuple[int, int],
    score_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """Postprocess detection predictions.
    
    Args:
        predictions: Model predictions
        original_size: Original image size (height, width)
        score_threshold: Minimum score to keep
    
    Returns:
        Dict[str, np.ndarray]: Postprocessed predictions
    """
    # Get predictions
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    # Filter by score
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Scale boxes to original size
    h_scale = original_size[0] / predictions['boxes'].size(1)
    w_scale = original_size[1] / predictions['boxes'].size(2)
    
    boxes[:, [0, 2]] *= w_scale
    boxes[:, [1, 3]] *= h_scale
    
    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels
    }

def postprocess_segmentation_predictions(
    predictions: torch.Tensor,
    original_size: Tuple[int, int]
) -> np.ndarray:
    """Postprocess segmentation predictions.
    
    Args:
        predictions: Model predictions (B, C, H, W)
        original_size: Original image size (height, width)
    
    Returns:
        np.ndarray: Postprocessed segmentation mask
    """
    # Get predictions
    predictions = predictions.cpu().numpy()
    
    # Get class predictions
    masks = np.argmax(predictions, axis=1)
    
    # Resize to original size
    masks = np.array([
        cv2.resize(
            mask.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        for mask in masks
    ])
    
    return masks

def create_detection_target(
    boxes: List[List[float]],
    labels: List[int],
    image_size: Tuple[int, int]
) -> Dict[str, torch.Tensor]:
    """Create detection target dictionary.
    
    Args:
        boxes: List of bounding boxes in [x1, y1, x2, y2] format
        labels: List of class labels
        image_size: Image size (height, width)
    
    Returns:
        Dict[str, torch.Tensor]: Target dictionary
    """
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'image_size': torch.tensor(image_size, dtype=torch.long)
    }

def create_segmentation_target(
    mask: np.ndarray,
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """Create segmentation target tensor.
    
    Args:
        mask: Segmentation mask
        image_size: Target image size (height, width)
    
    Returns:
        torch.Tensor: Target tensor
    """
    # Resize mask
    mask = cv2.resize(
        mask.astype(np.uint8),
        (image_size[1], image_size[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    return torch.tensor(mask, dtype=torch.long) 