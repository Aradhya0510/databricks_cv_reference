from transformers import AutoModelForSemanticSegmentation
from tasks.common.base_module import BaseVisionModule, BaseConfig
import torch
from typing import Dict, Any, List
import numpy as np
from torchmetrics import JaccardIndex, Dice
import torch.nn.functional as F

class SemanticSegmentationConfig(BaseConfig):
    """Configuration specific to semantic segmentation."""
    num_classes: int = 21  # Default for Pascal VOC
    ignore_index: int = 255
    use_aux_loss: bool = False
    aux_loss_weight: float = 0.4

class SemanticSegmentationModule(BaseVisionModule):
    """Semantic segmentation module using HuggingFace models."""
    
    def __init__(self, model_ckpt: str, config: SemanticSegmentationConfig = None):
        """Initialize semantic segmentation module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
        """
        config = config or SemanticSegmentationConfig()
        super().__init__(config)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_ckpt)
        self.save_hyperparameters()
        
        # Initialize metrics
        self.train_iou = JaccardIndex(task='multiclass', num_classes=config.num_classes, ignore_index=config.ignore_index)
        self.val_iou = JaccardIndex(task='multiclass', num_classes=config.num_classes, ignore_index=config.ignore_index)
        self.train_dice = Dice(num_classes=config.num_classes, ignore_index=config.ignore_index)
        self.val_dice = Dice(num_classes=config.num_classes, ignore_index=config.ignore_index)

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