from tasks.common.base_module import MetricLogger
import torch
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ClassificationMetricLogger(MetricLogger):
    """Logger for classification-specific metrics."""
    
    def _compute_accuracy(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute accuracy score."""
        predictions = outputs['predictions'].cpu().numpy()
        targets = batch['labels'].cpu().numpy()
        return accuracy_score(targets, predictions)
    
    def _compute_f1(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute F1 score."""
        predictions = outputs['predictions'].cpu().numpy()
        targets = batch['labels'].cpu().numpy()
        return f1_score(targets, predictions, average='weighted')
    
    def _compute_precision(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute precision score."""
        predictions = outputs['predictions'].cpu().numpy()
        targets = batch['labels'].cpu().numpy()
        return precision_score(targets, predictions, average='weighted')
    
    def _compute_recall(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute recall score."""
        predictions = outputs['predictions'].cpu().numpy()
        targets = batch['labels'].cpu().numpy()
        return recall_score(targets, predictions, average='weighted') 