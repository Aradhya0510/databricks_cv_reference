from transformers import AutoModelForImageClassification
from tasks.common.base_module import BaseVisionModule, BaseConfig
import torch
from typing import Dict, Any
import torch.nn.functional as F

class ClassificationConfig(BaseConfig):
    """Configuration specific to image classification."""
    num_labels: int = 1000
    dropout: float = 0.1
    label_smoothing: float = 0.0

class ClassificationModule(BaseVisionModule):
    """Image classification module using HuggingFace models."""
    
    def __init__(self, model_ckpt: str, config: ClassificationConfig = None):
        """Initialize classification module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
        """
        config = config or ClassificationConfig()
        super().__init__(config)
        self.model = AutoModelForImageClassification.from_pretrained(
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