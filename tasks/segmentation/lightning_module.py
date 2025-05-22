from transformers import AutoModelForSemanticSegmentation
from tasks.common.base_module import BaseVisionModule, BaseConfig
import torch
from typing import Dict, Any
import torch.nn.functional as F

class SegmentationConfig(BaseConfig):
    """Configuration specific to semantic segmentation."""
    num_labels: int = 21  # Default for Pascal VOC
    ignore_index: int = 255
    use_aux_loss: bool = False

class SegmentationModule(BaseVisionModule):
    """Semantic segmentation module using HuggingFace models."""
    
    def __init__(self, model_ckpt: str, config: SegmentationConfig = None):
        """Initialize segmentation module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
        """
        config = config or SegmentationConfig()
        super().__init__(config)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_ckpt,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True
        )
        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Dictionary containing model outputs
        """
        return self.model(**batch)

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
        
        # Log segmentation metrics
        if "labels" in batch:
            predictions = outputs.logits
            self._log_segmentation_metrics(predictions, batch["labels"])

    def _log_segmentation_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Log segmentation-specific metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        """
        # Calculate pixel accuracy
        pred_labels = predictions.argmax(dim=1)
        valid_mask = targets != self.config.ignore_index
        pixel_accuracy = (pred_labels[valid_mask] == targets[valid_mask]).float().mean()
        self.log("val_pixel_accuracy", pixel_accuracy, on_step=False, on_epoch=True)
        
        # Calculate mean IoU
        intersection = torch.zeros(self.config.num_labels, device=predictions.device)
        union = torch.zeros(self.config.num_labels, device=predictions.device)
        
        for label in range(self.config.num_labels):
            pred_mask = pred_labels == label
            target_mask = targets == label
            intersection[label] = (pred_mask & target_mask).sum()
            union[label] = (pred_mask | target_mask).sum()
        
        iou = intersection / (union + 1e-8)
        mean_iou = iou[union > 0].mean()
        self.log("val_mean_iou", mean_iou, on_step=False, on_epoch=True) 