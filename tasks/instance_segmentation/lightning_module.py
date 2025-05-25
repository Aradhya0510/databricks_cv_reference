from transformers import AutoModelForInstanceSegmentation
from tasks.common.base_module import BaseVisionModule, BaseConfig
import torch
from typing import Dict, Any, List
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import tempfile
import os

class InstanceSegmentationConfig(BaseConfig):
    """Configuration specific to instance segmentation."""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100
    mask_threshold: float = 0.5

class InstanceSegmentationModule(BaseVisionModule):
    """Instance segmentation module using HuggingFace models."""
    
    def __init__(self, model_ckpt: str, config: InstanceSegmentationConfig = None):
        """Initialize instance segmentation module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
        """
        config = config or InstanceSegmentationConfig()
        super().__init__(config)
        self.model = AutoModelForInstanceSegmentation.from_pretrained(model_ckpt)
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
        loss = outputs['loss']
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step implementation.
        
        Args:
            batch: Dictionary containing COCO format tensors
            batch_idx: Index of the current batch
        """
        outputs = self(batch)
        loss = outputs['loss']
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log segmentation metrics
        if "labels" in batch:
            predictions = outputs['predictions']
            self._log_segmentation_metrics(predictions, batch["labels"])

    def _log_segmentation_metrics(self, predictions: List[Dict], targets: List[Dict]) -> None:
        """Log instance segmentation metrics using COCO evaluation.
        
        Args:
            predictions: List of predictions in COCO format
            targets: List of ground truth annotations in COCO format
        """
        # Convert predictions and targets to COCO format
        coco_gt = self._to_coco_format(targets)
        
        # Format predictions for COCO evaluation
        coco_dt = []
        for pred in predictions:
            if isinstance(pred, dict):
                # Handle single prediction
                coco_dt.append({
                    'image_id': pred['image_id'],
                    'category_id': pred['category_id'],
                    'bbox': pred['bbox'],
                    'score': pred['score'],
                    'segmentation': pred['segmentation']
                })
            else:
                # Handle list of predictions
                for p in pred:
                    coco_dt.append({
                        'image_id': p['image_id'],
                        'category_id': p['category_id'],
                        'bbox': p['bbox'],
                        'score': p['score'],
                        'segmentation': p['segmentation']
                    })
        
        # Create temporary files for COCO evaluation
        f_gt = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        f_dt = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        try:
            # Write ground truth and predictions to temporary files
            json.dump(coco_gt, f_gt)
            json.dump(coco_dt, f_dt)
            
            # Close files to ensure they are written
            f_gt.close()
            f_dt.close()
            
            # Load COCO objects
            coco_gt = COCO(f_gt.name)
            coco_dt = coco_gt.loadRes(f_dt.name)
            
            # Initialize COCO evaluation for both bbox and segmentation
            coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
            
            # Evaluate both metrics
            coco_eval_bbox.evaluate()
            coco_eval_bbox.accumulate()
            coco_eval_bbox.summarize()
            
            coco_eval_segm.evaluate()
            coco_eval_segm.accumulate()
            coco_eval_segm.summarize()
            
            # Log metrics
            metrics = {
                'bbox_mAP': coco_eval_bbox.stats[0],
                'bbox_mAP_50': coco_eval_bbox.stats[1],
                'bbox_mAP_75': coco_eval_bbox.stats[2],
                'segm_mAP': coco_eval_segm.stats[0],
                'segm_mAP_50': coco_eval_segm.stats[1],
                'segm_mAP_75': coco_eval_segm.stats[2]
            }
            self.log_dict(metrics, on_epoch=True)
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(f_gt.name)
                os.unlink(f_dt.name)
            except:
                pass

    def _to_coco_format(self, annotations: List[Dict]) -> Dict:
        """Convert annotations to COCO format.
        
        Args:
            annotations: List of annotations
            
        Returns:
            Dictionary in COCO format
        """
        # Handle both tensor and list inputs
        processed_annotations = []
        for i, ann in enumerate(annotations):
            # Skip empty tensors
            if isinstance(ann, torch.Tensor) and ann.numel() == 0:
                continue
                
            # Handle category_id
            if isinstance(ann, dict):
                category_id = ann.get('category_id')
                if isinstance(category_id, torch.Tensor):
                    if category_id.numel() == 0:
                        continue
                    if category_id.numel() == 1:
                        category_id = int(category_id.item())
                    else:
                        category_id = int(category_id[0].item())
                else:
                    category_id = int(category_id)
            elif isinstance(ann, torch.Tensor):
                if ann.numel() == 0:
                    continue
                if ann.numel() == 1:
                    category_id = int(ann.item())
                else:
                    category_id = int(ann[0].item())
            else:
                category_id = int(ann)
                
            # Handle bbox
            if isinstance(ann, dict):
                bbox = ann.get('bbox')
                if isinstance(bbox, torch.Tensor):
                    if bbox.numel() == 0:
                        continue
                    bbox = bbox.cpu().numpy().tolist()
            else:
                bbox = [0, 0, 0, 0]
                
            # Handle segmentation
            if isinstance(ann, dict):
                segmentation = ann.get('segmentation')
                if isinstance(segmentation, torch.Tensor):
                    if segmentation.numel() == 0:
                        continue
                    segmentation = segmentation.cpu().numpy().tolist()
                elif segmentation is None:
                    segmentation = []
            else:
                segmentation = []
                
            # Handle area
            if isinstance(ann, dict):
                area = ann.get('area')
                if isinstance(area, torch.Tensor):
                    if area.numel() == 0:
                        continue
                    if area.numel() == 1:
                        area = float(area.item())
                    else:
                        area = float(area[0].item())
                else:
                    area = float(area if area is not None else 0)
            else:
                area = 0.0
                
            # Handle iscrowd
            if isinstance(ann, dict):
                iscrowd = ann.get('iscrowd')
                if isinstance(iscrowd, torch.Tensor):
                    if iscrowd.numel() == 0:
                        continue
                    if iscrowd.numel() == 1:
                        iscrowd = int(iscrowd.item())
                    else:
                        iscrowd = int(iscrowd[0].item())
                else:
                    iscrowd = int(iscrowd if iscrowd is not None else 0)
            else:
                iscrowd = 0
                
            processed_annotations.append({
                'id': i,
                'image_id': i,
                'category_id': category_id,
                'bbox': bbox,
                'segmentation': segmentation,
                'area': area,
                'iscrowd': iscrowd
            })
        
        # If no valid annotations, return empty COCO format
        if not processed_annotations:
            return {
                'images': [],
                'annotations': [],
                'categories': [
                    {'id': i, 'name': f'class_{i}'}
                    for i in range(self.model.config.num_labels)
                ]
            }
        
        return {
            'images': [{'id': i} for i in range(len(processed_annotations))],
            'annotations': processed_annotations,
            'categories': [
                {'id': i, 'name': f'class_{i}'}
                for i in range(self.model.config.num_labels)
            ]
        } 