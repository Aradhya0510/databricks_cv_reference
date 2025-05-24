from transformers import AutoModelForObjectDetection
from tasks.common.base_module import BaseVisionModule, BaseConfig
import torch
from typing import Dict, Any, List
from pycocotools.coco import COCO
import numpy as np

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

    def _prepare_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by the model.
        
        Args:
            batch: Dictionary containing COCO format tensors
            
        Returns:
            Dictionary of model inputs
        """
        # This method should be overridden by model-specific implementations
        raise NotImplementedError("Model-specific input preparation not implemented")

    def _process_model_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process model outputs back to COCO format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary of COCO format outputs
        """
        # This method should be overridden by model-specific implementations
        raise NotImplementedError("Model-specific output processing not implemented")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            batch: Dictionary containing COCO format tensors
            
        Returns:
            Dictionary containing model outputs in COCO format
        """
        # Prepare inputs for the specific model
        model_inputs = self._prepare_model_inputs(batch)
        
        # Get model outputs
        model_outputs = self.model(**model_inputs)
        
        # Process outputs back to COCO format
        return self._process_model_outputs(model_outputs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step implementation.
        
        Args:
            batch: Dictionary containing COCO format tensors
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
            batch: Dictionary containing COCO format tensors
            batch_idx: Index of the current batch
        """
        outputs = self(batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log detection metrics
        if "labels" in batch:
            predictions = outputs.predictions
            self._log_detection_metrics(predictions, batch["labels"])

    def _log_detection_metrics(self, predictions: List[Dict], targets: List[Dict]) -> None:
        """Log detection-specific metrics using COCO evaluation.
        
        Args:
            predictions: List of predictions in COCO format
            targets: List of ground truth annotations in COCO format
        """
        # Use COCO evaluation metrics
        from pycocotools.cocoeval import COCOeval
        import json
        
        # Convert predictions and targets to COCO format
        coco_gt = self._to_coco_format(targets)
        coco_dt = self._to_coco_format(predictions)
        
        # Initialize COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Log metrics
        metrics = {
            'mAP': coco_eval.stats[0],
            'mAP_50': coco_eval.stats[1],
            'mAP_75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5]
        }
        self.log_dict(metrics, on_epoch=True)

    def _to_coco_format(self, annotations: List[Dict]) -> Dict:
        """Convert annotations to COCO format.
        
        Args:
            annotations: List of annotations
            
        Returns:
            Dictionary in COCO format
        """
        return {
            'images': [{'id': i} for i in range(len(annotations))],
            'annotations': [
                {
                    'id': i,
                    'image_id': i,
                    'category_id': int(ann['category_id']),
                    'bbox': ann['bbox'],
                    'area': float(ann['area']),
                    'iscrowd': int(ann.get('iscrowd', 0))
                }
                for i, ann in enumerate(annotations)
            ],
            'categories': [
                {'id': i, 'name': f'class_{i}'}
                for i in range(self.model.config.num_labels)
            ]
        } 