from transformers import AutoModelForObjectDetection
from tasks.common.base_module import BaseVisionModule, BaseConfig, ModelProcessor, MetricLogger
import torch
from typing import Dict, Any, List, Optional
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import tempfile
import os
from dataclasses import dataclass

@dataclass
class DetectionConfig(BaseConfig):
    """Configuration specific to object detection."""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100

class DetectionProcessor(ModelProcessor):
    """Base processor for detection model inputs and outputs.
    
    This class provides a template for model-specific implementations.
    Each model type should implement its own processor that inherits from this class.
    """
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by the model.
        
        Args:
            batch: Dictionary containing COCO format tensors
            
        Returns:
            Dictionary of model inputs
        """
        return {
            'pixel_values': batch['pixel_values'],
            'labels': batch['labels']
        }
    
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process model outputs back to COCO format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary of COCO format outputs
        """
        return {
            'loss': outputs.loss,
            'predictions': outputs.predictions
        }

class DetectionMetricLogger(MetricLogger):
    """Logger for detection-specific metrics."""
    
    def _compute_map(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute mean Average Precision."""
        predictions = outputs['predictions']
        targets = batch['labels']
        
        # Convert predictions and targets to COCO format
        coco_gt = self._to_coco_format(targets)
        coco_dt = self._format_predictions(predictions)
        
        # Evaluate using COCO API
        metrics = self._evaluate_coco(coco_gt, coco_dt)
        return metrics['mAP']
    
    def _compute_map_50(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute mean Average Precision at IoU=0.50."""
        predictions = outputs['predictions']
        targets = batch['labels']
        
        # Convert predictions and targets to COCO format
        coco_gt = self._to_coco_format(targets)
        coco_dt = self._format_predictions(predictions)
        
        # Evaluate using COCO API
        metrics = self._evaluate_coco(coco_gt, coco_dt)
        return metrics['mAP_50']
    
    def _format_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Format predictions for COCO evaluation."""
        coco_dt = []
        for pred in predictions:
            if isinstance(pred, dict):
                coco_dt.append({
                    'image_id': pred['image_id'],
                    'category_id': pred['category_id'],
                    'bbox': pred['bbox'],
                    'score': pred['score']
                })
            else:
                for p in pred:
                    coco_dt.append({
                        'image_id': p['image_id'],
                        'category_id': p['category_id'],
                        'bbox': p['bbox'],
                        'score': p['score']
                    })
        return coco_dt
    
    def _evaluate_coco(self, coco_gt: Dict, coco_dt: List[Dict]) -> Dict[str, float]:
        """Evaluate predictions using COCO API."""
        # Create temporary files
        f_gt = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        f_dt = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        try:
            # Write to temporary files
            json.dump(coco_gt, f_gt)
            json.dump(coco_dt, f_dt)
            f_gt.close()
            f_dt.close()
            
            # Load COCO objects
            coco_gt = COCO(f_gt.name)
            coco_dt = coco_gt.loadRes(f_dt.name)
            
            # Evaluate
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            return {
                'mAP': coco_eval.stats[0],
                'mAP_50': coco_eval.stats[1],
                'mAP_75': coco_eval.stats[2],
                'mAP_small': coco_eval.stats[3],
                'mAP_medium': coco_eval.stats[4],
                'mAP_large': coco_eval.stats[5]
            }
        finally:
            # Clean up
            try:
                os.unlink(f_gt.name)
                os.unlink(f_dt.name)
            except:
                pass
    
    def _to_coco_format(self, annotations: List[Dict]) -> Dict:
        """Convert annotations to COCO format."""
        processed_annotations = []
        for i, ann in enumerate(annotations):
            if isinstance(ann, torch.Tensor) and ann.numel() == 0:
                continue
            
            # Process category_id
            category_id = self._process_category_id(ann)
            if category_id is None:
                continue
            
            # Process bbox
            bbox = self._process_bbox(ann)
            if bbox is None:
                continue
            
            # Process area and iscrowd
            area = self._process_area(ann)
            iscrowd = self._process_iscrowd(ann)
            
            processed_annotations.append({
                'id': i,
                'image_id': i,
                'category_id': category_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': iscrowd
            })
        
        return {
            'images': [{'id': i} for i in range(len(processed_annotations))],
            'annotations': processed_annotations,
            'categories': [
                {'id': i, 'name': f'class_{i}'}
                for i in range(self.num_classes)
            ]
        }
    
    def _process_category_id(self, ann: Dict) -> Optional[int]:
        """Process category ID from annotation."""
        if isinstance(ann, dict):
            category_id = ann.get('category_id')
            if isinstance(category_id, torch.Tensor):
                if category_id.numel() == 0:
                    return None
                return int(category_id[0].item() if category_id.numel() > 1 else category_id.item())
            return int(category_id)
        elif isinstance(ann, torch.Tensor):
            if ann.numel() == 0:
                return None
            return int(ann[0].item() if ann.numel() > 1 else ann.item())
        return int(ann)
    
    def _process_bbox(self, ann: Dict) -> Optional[List[float]]:
        """Process bounding box from annotation."""
        if isinstance(ann, dict):
            bbox = ann.get('bbox')
            if isinstance(bbox, torch.Tensor):
                if bbox.numel() == 0:
                    return None
                return bbox.cpu().numpy().tolist()
        return [0, 0, 0, 0]
    
    def _process_area(self, ann: Dict) -> float:
        """Process area from annotation."""
        if isinstance(ann, dict):
            area = ann.get('area')
            if isinstance(area, torch.Tensor):
                if area.numel() == 0:
                    return 0.0
                return float(area[0].item() if area.numel() > 1 else area.item())
            return float(area if area is not None else 0)
        return 0.0
    
    def _process_iscrowd(self, ann: Dict) -> int:
        """Process iscrowd flag from annotation."""
        if isinstance(ann, dict):
            iscrowd = ann.get('iscrowd')
            if isinstance(iscrowd, torch.Tensor):
                if iscrowd.numel() == 0:
                    return 0
                return int(iscrowd[0].item() if iscrowd.numel() > 1 else iscrowd.item())
            return int(iscrowd if iscrowd is not None else 0)
        return 0

class DetectionModule(BaseVisionModule):
    """Object detection module using HuggingFace models."""
    
    def __init__(
        self,
        model_ckpt: str,
        config: DetectionConfig = None,
        processor: DetectionProcessor = None,
        metric_logger: DetectionMetricLogger = None
    ):
        """Initialize detection module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
            processor: Optional custom processor
            metric_logger: Optional custom metric logger
        """
        config = config or DetectionConfig()
        processor = processor or DetectionProcessor()
        metric_logger = metric_logger or DetectionMetricLogger(
            metrics=['map', 'map_50'],
            log_every_n_steps=config.log_every_n_steps
        )
        
        super().__init__(config, processor, metric_logger)
    
    def _init_model(self) -> torch.nn.Module:
        """Initialize the detection model."""
        return AutoModelForObjectDetection.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes
        ) 