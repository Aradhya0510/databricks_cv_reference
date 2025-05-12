from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        
    def compute_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, float]:
        """Compute segmentation metrics."""
        metrics = {}
        
        # Convert predictions and ground truth to numpy arrays
        pred_masks = [p['mask'] for p in predictions]
        gt_masks = [g['mask'] for g in ground_truth]
        
        # Compute IoU for each class
        ious = self._compute_iou(pred_masks, gt_masks)
        metrics['mean_iou'] = np.mean(ious)
        
        # Compute pixel accuracy
        pixel_acc = self._compute_pixel_accuracy(pred_masks, gt_masks)
        metrics['pixel_accuracy'] = pixel_acc
        
        # Compute dice coefficient
        dice = self._compute_dice(pred_masks, gt_masks)
        metrics['dice_coefficient'] = dice
        
        # Compute SSIM
        ssim_score = self._compute_ssim(pred_masks, gt_masks)
        metrics['ssim'] = ssim_score
        
        return metrics
        
    def _compute_iou(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray]
    ) -> np.ndarray:
        """Compute IoU for each class."""
        ious = np.zeros(self.num_classes)
        
        for cls in range(self.num_classes):
            intersection = 0
            union = 0
            
            for pred, gt in zip(pred_masks, gt_masks):
                pred_cls = (pred == cls)
                gt_cls = (gt == cls)
                
                intersection += np.logical_and(pred_cls, gt_cls).sum()
                union += np.logical_or(pred_cls, gt_cls).sum()
                
            ious[cls] = intersection / (union + 1e-10)
            
        return ious
        
    def _compute_pixel_accuracy(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray]
    ) -> float:
        """Compute pixel-wise accuracy."""
        correct = 0
        total = 0
        
        for pred, gt in zip(pred_masks, gt_masks):
            correct += (pred == gt).sum()
            total += gt.size
            
        return correct / total
        
    def _compute_dice(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray]
    ) -> float:
        """Compute Dice coefficient."""
        dice = 0
        
        for pred, gt in zip(pred_masks, gt_masks):
            intersection = np.logical_and(pred, gt).sum()
            dice += (2 * intersection) / (pred.sum() + gt.sum() + 1e-10)
            
        return dice / len(pred_masks)
        
    def _compute_ssim(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray]
    ) -> float:
        """Compute Structural Similarity Index."""
        ssim_scores = []
        
        for pred, gt in zip(pred_masks, gt_masks):
            score = ssim(pred, gt, data_range=1.0)
            ssim_scores.append(score)
            
        return np.mean(ssim_scores)
        
    def plot_segmentation_results(
        self,
        images: List[np.ndarray],
        predictions: List[Dict],
        ground_truth: List[Dict],
        class_names: Optional[List[str]] = None,
        max_images: int = 4
    ) -> plt.Figure:
        """Plot segmentation results with overlays."""
        fig, axes = plt.subplots(max_images, 3, figsize=(15, 5*max_images))
        
        for idx, (img, pred, gt) in enumerate(zip(images, predictions, ground_truth)):
            if idx >= max_images:
                break
                
            # Original image
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')
            
            # Ground truth
            axes[idx, 1].imshow(img)
            axes[idx, 1].imshow(gt['mask'], alpha=0.5)
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            # Prediction
            axes[idx, 2].imshow(img)
            axes[idx, 2].imshow(pred['mask'], alpha=0.5)
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
            
        plt.tight_layout()
        return fig 