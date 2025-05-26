from transformers import AutoModelForSemanticSegmentation
from tasks.common.base_module import BaseVisionModule, BaseConfig, ModelProcessor, MetricLogger
import torch
from typing import Dict, Any, List
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SemanticSegmentationConfig(BaseConfig):
    """Configuration specific to semantic segmentation."""
    num_labels: int = 1000
    dropout: float = 0.1
    label_smoothing: float = 0.0

class SemanticSegmentationProcessor(ModelProcessor):
    """Processor for semantic segmentation model inputs and outputs."""
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by the model.
        
        Args:
            batch: Dictionary containing image tensors and masks
            
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

class SemanticSegmentationMetricLogger(MetricLogger):
    """Logger for semantic segmentation-specific metrics."""
    
    def _compute_iou(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute mean Intersection over Union."""
        predictions = outputs['predictions']
        targets = batch['labels']
        
        # Compute IoU for each class
        ious = []
        for cls in range(self.num_classes):
            pred_mask = (predictions == cls)
            target_mask = (targets == cls)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                ious.append((intersection / union).item())
        
        return sum(ious) / len(ious) if ious else 0.0
    
    def _compute_pixel_accuracy(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute pixel-wise accuracy."""
        predictions = outputs['predictions']
        targets = batch['labels']
        
        correct = (predictions == targets).sum().float()
        total = targets.numel()
        
        return (correct / total).item()
    
    def _compute_dice(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute mean Dice coefficient."""
        predictions = outputs['predictions']
        targets = batch['labels']
        
        # Compute Dice for each class
        dice_scores = []
        for cls in range(self.num_classes):
            pred_mask = (predictions == cls)
            target_mask = (targets == cls)
            
            intersection = (pred_mask & target_mask).sum().float()
            total = pred_mask.sum() + target_mask.sum()
            
            if total > 0:
                dice_scores.append((2 * intersection / total).item())
        
        return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0

class SemanticSegmentationModule(BaseVisionModule):
    """Semantic segmentation module using HuggingFace models."""
    
    def __init__(
        self,
        model_ckpt: str,
        config: SemanticSegmentationConfig = None,
        processor: SemanticSegmentationProcessor = None,
        metric_logger: SemanticSegmentationMetricLogger = None
    ):
        """Initialize semantic segmentation module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
            processor: Optional custom processor
            metric_logger: Optional custom metric logger
        """
        config = config or SemanticSegmentationConfig()
        processor = processor or SemanticSegmentationProcessor()
        metric_logger = metric_logger or SemanticSegmentationMetricLogger(
            metrics=['iou', 'pixel_accuracy', 'dice'],
            log_every_n_steps=config.log_every_n_steps
        )
        
        super().__init__(config, processor, metric_logger)
    
    def _init_model(self) -> torch.nn.Module:
        """Initialize the semantic segmentation model."""
        return AutoModelForSemanticSegmentation.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            dropout=self.config.dropout,
            label_smoothing=self.config.label_smoothing
        )

    def _prepare_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by the model.
        
        Args:
            batch: Dictionary containing image tensors and masks
            
        Returns:
            Dictionary of model inputs
        """
        # This method should be overridden by model-specific implementations
        raise NotImplementedError("Model-specific input preparation not implemented")

    def _process_model_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary containing loss and predictions
        """
        # This method should be overridden by model-specific implementations
        raise NotImplementedError("Model-specific output processing not implemented")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            batch: Dictionary containing image tensors and masks
            
        Returns:
            Dictionary containing model outputs
        """
        # Prepare inputs for the specific model
        model_inputs = self._prepare_model_inputs(batch)
        
        # Get model outputs
        model_outputs = self.model(**model_inputs)
        
        # Process outputs
        return self._process_model_outputs(model_outputs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step implementation.
        
        Args:
            batch: Dictionary containing image tensors and masks
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor
        """
        outputs = self(batch)
        loss = outputs['loss']
        
        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update metrics
        if 'logits' in outputs and 'labels' in batch:
            preds = outputs['logits'].argmax(dim=1)
            self.train_iou(preds, batch['labels'])
            self.train_dice(preds, batch['labels'])
            
            # Log metrics
            self.log("train_iou", self.train_iou, on_step=False, on_epoch=True)
            self.log("train_dice", self.train_dice, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step implementation.
        
        Args:
            batch: Dictionary containing image tensors and masks
            batch_idx: Index of the current batch
        """
        outputs = self(batch)
        loss = outputs['loss']
        
        # Log loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update metrics
        if 'logits' in outputs and 'labels' in batch:
            preds = outputs['logits'].argmax(dim=1)
            self.val_iou(preds, batch['labels'])
            self.val_dice(preds, batch['labels'])
            
            # Log metrics
            self.log("val_iou", self.val_iou, on_step=False, on_epoch=True)
            self.log("val_dice", self.val_dice, on_step=False, on_epoch=True)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step implementation.
        
        Args:
            batch: Dictionary containing image tensors
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing predictions
        """
        outputs = self(batch)
        return {
            'logits': outputs['logits'],
            'preds': outputs['logits'].argmax(dim=1)
        } 