from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoModelForObjectDetection, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import DetrObjectDetectionOutput

@dataclass
class DetectionModelConfig:
    """Configuration for detection model."""
    model_name: str
    num_classes: int
    pretrained: bool = True
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_detections: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    epochs: int = 10
    class_names: Optional[List[str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None

class DetectionModel(pl.LightningModule):
    """Base detection model that can work with any Hugging Face object detection model."""
    
    def __init__(self, config: Union[Dict[str, Any], DetectionModelConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = DetectionModelConfig(**config)
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Initialize model
        self._init_model()
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_model(self) -> None:
        """Initialize the model architecture."""
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            **self.config.model_kwargs or {}
        )
        
        # Initialize model
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.config.model_name,
            config=model_config,
            **self.config.model_kwargs or {}
        )
        
        # Set model parameters
        self.model.config.confidence_threshold = self.config.confidence_threshold
        self.model.config.iou_threshold = self.config.iou_threshold
        self.model.config.max_detections = self.config.max_detections
    
    def _init_metrics(self) -> None:
        """Initialize metrics for training, validation, and testing."""
        self.train_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True
        )
        self.val_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True
        )
        self.test_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Union[torch.Tensor, DetrObjectDetectionOutput]:
        """Forward pass of the model.
        
        Args:
            pixel_values: Input images
            pixel_mask: Optional attention mask
            labels: Optional target labels
            
        Returns:
            Model outputs
        """
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
            labels=batch["labels"]
        )
        
        # Update metrics
        self.train_map.update(
            outputs.pred_boxes,
            outputs.pred_logits,
            batch["labels"]["boxes"],
            batch["labels"]["labels"]
        )
        
        # Log metrics
        self.log("train_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in outputs.loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True)
        
        return outputs.loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
            labels=batch["labels"]
        )
        
        # Update metrics
        self.val_map.update(
            outputs.pred_boxes,
            outputs.pred_logits,
            batch["labels"]["boxes"],
            batch["labels"]["labels"]
        )
        
        # Log metrics
        self.log("val_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in outputs.loss_dict.items():
            self.log(f"val_{k}", v, on_step=True, on_epoch=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
            labels=batch["labels"]
        )
        
        # Update metrics
        self.test_map.update(
            outputs.pred_boxes,
            outputs.pred_logits,
            batch["labels"]["boxes"],
            batch["labels"]["labels"]
        )
        
        # Log metrics
        self.log("test_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in outputs.loss_dict.items():
            self.log(f"test_{k}", v, on_step=True, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        """Calculate and log mAP metrics at the end of validation epoch."""
        map_metrics = self.val_map.compute()
        self.log("val_map", map_metrics["map"], prog_bar=True)
        self.log("val_map_50", map_metrics["map_50"], prog_bar=True)
        self.log("val_map_75", map_metrics["map_75"], prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
            return [optimizer], [scheduler]
        
        return optimizer
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int,
        **kwargs
    ) -> "DetectionModel":
        """Create a model from a pretrained checkpoint.
        
        Args:
            model_name: Name or path of the pretrained model
            num_classes: Number of output classes
            **kwargs: Additional arguments for model configuration
            
        Returns:
            Initialized model
        """
        config = DetectionModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
        return cls(config) 