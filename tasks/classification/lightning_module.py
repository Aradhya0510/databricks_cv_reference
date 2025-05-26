from transformers import AutoModelForImageClassification
from tasks.common.base_module import BaseVisionModule, BaseConfig, ModelProcessor, MetricLogger
import torch
from typing import Dict, Any, List
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ClassificationConfig(BaseConfig):
    """Configuration specific to image classification."""
    num_labels: int = 1000
    dropout: float = 0.1
    label_smoothing: float = 0.0

class ClassificationProcessor(ModelProcessor):
    """Processor for classification model inputs and outputs."""
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by the model.
        
        Args:
            batch: Dictionary containing image tensors and labels
            
        Returns:
            Dictionary of model inputs
        """
        return {
            'pixel_values': batch['pixel_values'],
            'labels': batch['labels']
        }
    
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary containing loss and predictions
        """
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'predictions': outputs.logits.argmax(dim=1)
        }

class ClassificationMetricLogger(MetricLogger):
    """Logger for classification-specific metrics."""
    
    def _compute_accuracy(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute classification accuracy."""
        predictions = outputs['predictions']
        targets = batch['labels']
        
        correct = (predictions == targets).sum().float()
        total = targets.numel()
        
        return (correct / total).item()
    
    def _compute_top_k_accuracy(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], k: int = 5) -> float:
        """Compute top-k classification accuracy."""
        logits = outputs['logits']
        targets = batch['labels']
        
        # Get top-k predictions
        _, top_k_preds = torch.topk(logits, k, dim=1)
        
        # Check if target is in top-k predictions
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1
        
        return correct / len(targets)
    
    def _compute_f1_score(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute F1 score for multi-class classification."""
        predictions = outputs['predictions']
        targets = batch['labels']
        
        # Convert to one-hot encoding
        pred_one_hot = F.one_hot(predictions, num_classes=self.num_classes)
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        
        # Compute precision and recall for each class
        true_positives = (pred_one_hot & target_one_hot).sum(dim=0).float()
        false_positives = (pred_one_hot & ~target_one_hot).sum(dim=0).float()
        false_negatives = (~pred_one_hot & target_one_hot).sum(dim=0).float()
        
        # Compute F1 score for each class
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # Return mean F1 score
        return f1_scores.mean().item()

class ClassificationModule(BaseVisionModule):
    """Image classification module using HuggingFace models."""
    
    def __init__(
        self,
        model_ckpt: str,
        config: ClassificationConfig = None,
        processor: ClassificationProcessor = None,
        metric_logger: ClassificationMetricLogger = None
    ):
        """Initialize classification module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
            processor: Optional custom processor
            metric_logger: Optional custom metric logger
        """
        config = config or ClassificationConfig()
        processor = processor or ClassificationProcessor()
        metric_logger = metric_logger or ClassificationMetricLogger(
            metrics=['accuracy', 'top_k_accuracy', 'f1_score'],
            log_every_n_steps=config.log_every_n_steps
        )
        
        super().__init__(config, processor, metric_logger)
    
    def _init_model(self) -> torch.nn.Module:
        """Initialize the classification model."""
        return AutoModelForImageClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            dropout=self.config.dropout,
            label_smoothing=self.config.label_smoothing
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step implementation.
        
        Args:
            batch: Dictionary containing input tensors
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor
        """
        outputs = self(batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step implementation.
        
        Args:
            batch: Dictionary containing input tensors
            batch_idx: Index of the current batch
        """
        outputs = self(batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log classification metrics
        if "labels" in batch:
            predictions = outputs.logits
            self._log_classification_metrics(predictions, batch["labels"])

    def _log_classification_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Log classification-specific metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        """
        # Calculate accuracy
        pred_labels = predictions.argmax(dim=-1)
        accuracy = (pred_labels == targets).float().mean()
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)
        
        # Calculate top-5 accuracy
        top5_preds = predictions.topk(5, dim=-1)[1]
        top5_accuracy = torch.any(top5_preds == targets.unsqueeze(1), dim=1).float().mean()
        self.log("val_top5_accuracy", top5_accuracy, on_step=False, on_epoch=True) 