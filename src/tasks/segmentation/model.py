from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Dice, JaccardIndex, Accuracy, Precision, Recall
from torchmetrics.detection import MeanAveragePrecision
from transformers import (
    AutoModelForSemanticSegmentation,
    AutoModelForInstanceSegmentation,
    AutoModelForPanopticSegmentation,
    AutoConfig,
    PreTrainedModel
)
from .adapters import get_output_adapter

@dataclass
class SegmentationModelConfig:
    """Configuration for segmentation model."""
    model_name: str
    num_classes: int
    segmentation_type: str = "semantic"  # "semantic", "instance", or "panoptic"
    pretrained: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    epochs: int = 10
    class_names: Optional[List[str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None

class SegmentationModel(pl.LightningModule):
    """Base segmentation model that can work with any Hugging Face segmentation model."""
    
    def __init__(self, config: Union[Dict[str, Any], SegmentationModelConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = SegmentationModelConfig(**config)
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Initialize model
        self._init_model()
        
        # Initialize metrics
        self._init_metrics()
        
        # Initialize output adapter
        self.output_adapter = get_output_adapter(
            config.model_name,
            segmentation_type=config.segmentation_type
        )
    
    def _init_model(self) -> None:
        """Initialize the model architecture."""
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_classes,
                **self.config.model_kwargs or {}
            )
            
            # Initialize model based on segmentation type
            if self.config.segmentation_type == "semantic":
                self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    self.config.model_name,
                    config=model_config,
                    ignore_mismatched_sizes=True,
                    **self.config.model_kwargs or {}
                )
            elif self.config.segmentation_type == "instance":
                self.model = AutoModelForInstanceSegmentation.from_pretrained(
                    self.config.model_name,
                    config=model_config,
                    ignore_mismatched_sizes=True,
                    **self.config.model_kwargs or {}
                )
            elif self.config.segmentation_type == "panoptic":
                self.model = AutoModelForPanopticSegmentation.from_pretrained(
                    self.config.model_name,
                    config=model_config,
                    ignore_mismatched_sizes=True,
                    **self.config.model_kwargs or {}
                )
            else:
                raise ValueError(f"Unknown segmentation type: {self.config.segmentation_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _init_metrics(self) -> None:
        """Initialize metrics for training, validation, and testing."""
        # Common metrics for all segmentation types
        self.train_dice = Dice(num_classes=self.config.num_classes)
        self.train_iou = JaccardIndex(task="multiclass", num_classes=self.config.num_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.train_precision = Precision(task="multiclass", num_classes=self.config.num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=self.config.num_classes)
        
        self.val_dice = Dice(num_classes=self.config.num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=self.config.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=self.config.num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=self.config.num_classes)
        
        self.test_dice = Dice(num_classes=self.config.num_classes)
        self.test_iou = JaccardIndex(task="multiclass", num_classes=self.config.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=self.config.num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=self.config.num_classes)
        
        # Instance and panoptic specific metrics
        if self.config.segmentation_type in ["instance", "panoptic"]:
            self.train_map = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="segm",
                class_metrics=True
            )
            self.val_map = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="segm",
                class_metrics=True
            )
            self.test_map = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="segm",
                class_metrics=True
            )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Forward pass of the model.
        
        Args:
            pixel_values: Input images
            labels: Optional target labels
            
        Returns:
            Dictionary containing model outputs including:
            - loss: Training loss if labels are provided
            - logits: Predicted class logits
            - loss_dict: Dictionary of individual loss components
        """
        # Validate input
        if pixel_values.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {pixel_values.dim()}D")
        
        # Adapt targets if needed
        if labels is not None:
            labels = self.output_adapter.adapt_targets(labels)
            
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels
        )
        return self.output_adapter.adapt_output(outputs)
    
    def _format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format model outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        return self.output_adapter.format_predictions(outputs)
    
    def _format_targets(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format batch targets for metric computation.
        
        Args:
            batch: Batch dictionary containing targets
            
        Returns:
            List of target dictionaries for each image
        """
        return self.output_adapter.format_targets(batch["labels"])
    
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
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        for pred, target in zip(preds, targets):
            # Common metrics
            if "masks" in pred and "masks" in target:
                self.train_dice.update(pred["masks"], target["masks"])
                self.train_iou.update(pred["masks"], target["masks"])
            if "labels" in pred and "labels" in target:
                self.train_accuracy.update(pred["labels"], target["labels"])
                self.train_precision.update(pred["labels"], target["labels"])
                self.train_recall.update(pred["labels"], target["labels"])
            
            # Instance and panoptic specific metrics
            if self.config.segmentation_type in ["instance", "panoptic"]:
                if all(k in pred for k in ["boxes", "masks", "scores", "labels"]):
                    self.train_map.update(
                        preds=[{
                            "boxes": pred["boxes"],
                            "masks": pred["masks"],
                            "scores": pred["scores"],
                            "labels": pred["labels"]
                        }],
                        target=[{
                            "boxes": target["boxes"],
                            "masks": target["masks"],
                            "labels": target["labels"]
                        }]
                    )
        
        # Log metrics
        self.log("train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        for k, v in outputs["loss_dict"].items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True)
        
        return outputs["loss"]
    
    def on_train_epoch_end(self) -> None:
        """Calculate and log metrics at the end of training epoch."""
        # Common metrics
        self.log("train_dice", self.train_dice.compute(), prog_bar=True)
        self.log("train_iou", self.train_iou.compute())
        self.log("train_accuracy", self.train_accuracy.compute())
        self.log("train_precision", self.train_precision.compute())
        self.log("train_recall", self.train_recall.compute())
        
        # Instance and panoptic specific metrics
        if self.config.segmentation_type in ["instance", "panoptic"]:
            self.log("train_map", self.train_map.compute(), prog_bar=True)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        for pred, target in zip(preds, targets):
            # Common metrics
            if "masks" in pred and "masks" in target:
                self.val_dice.update(pred["masks"], target["masks"])
                self.val_iou.update(pred["masks"], target["masks"])
            if "labels" in pred and "labels" in target:
                self.val_accuracy.update(pred["labels"], target["labels"])
                self.val_precision.update(pred["labels"], target["labels"])
                self.val_recall.update(pred["labels"], target["labels"])
            
            # Instance and panoptic specific metrics
            if self.config.segmentation_type in ["instance", "panoptic"]:
                if all(k in pred for k in ["boxes", "masks", "scores", "labels"]):
                    self.val_map.update(
                        preds=[{
                            "boxes": pred["boxes"],
                            "masks": pred["masks"],
                            "scores": pred["scores"],
                            "labels": pred["labels"]
                        }],
                        target=[{
                            "boxes": target["boxes"],
                            "masks": target["masks"],
                            "labels": target["labels"]
                        }]
                    )
        
        # Log metrics
        self.log("val_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """Calculate and log metrics at the end of validation epoch."""
        # Common metrics
        self.log("val_dice", self.val_dice.compute(), prog_bar=True)
        self.log("val_iou", self.val_iou.compute())
        self.log("val_accuracy", self.val_accuracy.compute())
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        
        # Instance and panoptic specific metrics
        if self.config.segmentation_type in ["instance", "panoptic"]:
            self.log("val_map", self.val_map.compute(), prog_bar=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        for pred, target in zip(preds, targets):
            # Common metrics
            if "masks" in pred and "masks" in target:
                self.test_dice.update(pred["masks"], target["masks"])
                self.test_iou.update(pred["masks"], target["masks"])
            if "labels" in pred and "labels" in target:
                self.test_accuracy.update(pred["labels"], target["labels"])
                self.test_precision.update(pred["labels"], target["labels"])
                self.test_recall.update(pred["labels"], target["labels"])
            
            # Instance and panoptic specific metrics
            if self.config.segmentation_type in ["instance", "panoptic"]:
                if all(k in pred for k in ["boxes", "masks", "scores", "labels"]):
                    self.test_map.update(
                        preds=[{
                            "boxes": pred["boxes"],
                            "masks": pred["masks"],
                            "scores": pred["scores"],
                            "labels": pred["labels"]
                        }],
                        target=[{
                            "boxes": target["boxes"],
                            "masks": target["masks"],
                            "labels": target["labels"]
                        }]
                    )
        
        # Log metrics
        self.log("test_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        for k, v in outputs["loss_dict"].items():
            self.log(f"test_{k}", v, on_step=True, on_epoch=True)
    
    def on_test_epoch_end(self) -> None:
        """Calculate and log metrics at the end of test epoch."""
        # Common metrics
        self.log("test_dice", self.test_dice.compute(), prog_bar=True)
        self.log("test_iou", self.test_iou.compute())
        self.log("test_accuracy", self.test_accuracy.compute())
        self.log("test_precision", self.test_precision.compute())
        self.log("test_recall", self.test_recall.compute())
        
        # Instance and panoptic specific metrics
        if self.config.segmentation_type in ["instance", "panoptic"]:
            self.log("test_map", self.test_map.compute(), prog_bar=True)
    
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
        segmentation_type: str = "semantic",
        **kwargs
    ) -> "SegmentationModel":
        """Create a model from a pretrained checkpoint.
        
        Args:
            model_name: Name or path of the pretrained model
            num_classes: Number of output classes
            segmentation_type: Type of segmentation ("semantic", "instance", or "panoptic")
            **kwargs: Additional arguments for model configuration
            
        Returns:
            Initialized model
        """
        config = SegmentationModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            segmentation_type=segmentation_type,
            **kwargs
        )
        return cls(config)
    
    def on_train_epoch_start(self) -> None:
        """Reset training metrics at the start of each epoch."""
        self.train_dice.reset()
        self.train_iou.reset()
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        if self.config.segmentation_type in ["instance", "panoptic"]:
            self.train_map.reset()
    
    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics at the start of each epoch."""
        self.val_dice.reset()
        self.val_iou.reset()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        if self.config.segmentation_type in ["instance", "panoptic"]:
            self.val_map.reset()
    
    def on_test_epoch_start(self) -> None:
        """Reset test metrics at the start of each epoch."""
        self.test_dice.reset()
        self.test_iou.reset()
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        if self.config.segmentation_type in ["instance", "panoptic"]:
            self.test_map.reset()
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional state to checkpoint.
        
        Args:
            checkpoint: Dictionary to save state to
        """
        checkpoint["model_config"] = self.config.__dict__
        checkpoint["class_names"] = self.config.class_names
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load additional state from checkpoint.
        
        Args:
            checkpoint: Dictionary to load state from
        """
        if "model_config" in checkpoint:
            self.config = SegmentationModelConfig(**checkpoint["model_config"])
        if "class_names" in checkpoint:
            self.config.class_names = checkpoint["class_names"] 