from tasks.common.base_module import MetricLogger
import torch
from typing import Dict, Any, List
import numpy as np

class SemanticSegmentationMetricLogger(MetricLogger):
    """Logger for semantic segmentation-specific metrics."""
    
    def _compute_iou(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute Intersection over Union (IoU)."""
        predictions = outputs['predictions'].cpu().numpy()
        targets = batch['masks'].cpu().numpy()
        
        intersection = np.logical_and(predictions, targets).sum()
        union = np.logical_or(predictions, targets).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_dice(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute Dice coefficient."""
        predictions = outputs['predictions'].cpu().numpy()
        targets = batch['masks'].cpu().numpy()
        
        intersection = np.logical_and(predictions, targets).sum()
        total = predictions.sum() + targets.sum()
        
        return 2 * intersection / total if total > 0 else 0.0
    
    def _compute_pixel_accuracy(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute pixel accuracy."""
        predictions = outputs['predictions'].cpu().numpy()
        targets = batch['masks'].cpu().numpy()
        
        return (predictions == targets).mean()
    
    def _compute_mean_iou(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute mean IoU across all classes."""
        predictions = outputs['predictions'].cpu().numpy()
        targets = batch['masks'].cpu().numpy()
        
        ious = []
        for class_id in range(self.num_classes):
            pred_mask = predictions == class_id
            target_mask = targets == class_id
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)
        
        return np.mean(ious) 