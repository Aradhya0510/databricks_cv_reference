from transformers import AutoModelForObjectDetection
from tasks.common.base_module import BaseVisionModule, BaseConfig
import torch
from typing import Dict, Any

class DetectionConfig(BaseConfig):
    """Configuration specific to object detection."""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100

class DetectionModule(BaseVisionModule):
    """Object detection module using HuggingFace models."""
    
    def __init__(self, model_ckpt: str, config: DetectionConfig = None):
        """Initialize detection module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
        """
        config = config or DetectionConfig()
        super().__init__(config)
        self.model = AutoModelForObjectDetection.from_pretrained(model_ckpt)
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
        
        # Log detection metrics
        if "labels" in batch:
            predictions = outputs.logits
            self._log_detection_metrics(predictions, batch["labels"])

    def _log_detection_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Log detection-specific metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        """
        # Implement detection metrics (mAP, etc.)
        pass 