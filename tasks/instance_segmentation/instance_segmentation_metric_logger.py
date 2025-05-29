from tasks.common.base_module import MetricLogger
import torch
from typing import Dict, Any, List
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
import os

class InstanceSegmentationMetricLogger(MetricLogger):
    """Logger for instance segmentation-specific metrics."""
    
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
    
    def _compute_mask_map(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute mask mAP."""
        predictions = outputs['predictions']
        targets = batch['labels']
        
        # Convert predictions and targets to COCO format
        coco_gt = self._to_coco_format(targets)
        coco_dt = self._format_predictions(predictions)
        
        # Evaluate using COCO API
        metrics = self._evaluate_coco(coco_gt, coco_dt, iou_type='segm')
        return metrics['mAP']
    
    def _format_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Format predictions for COCO evaluation."""
        coco_dt = []
        for pred in predictions:
            if isinstance(pred, dict):
                coco_dt.append({
                    'image_id': pred['image_id'],
                    'category_id': pred['category_id'],
                    'bbox': pred['bbox'],
                    'score': pred['score'],
                    'segmentation': pred['segmentation']
                })
            else:
                for p in pred:
                    coco_dt.append({
                        'image_id': p['image_id'],
                        'category_id': p['category_id'],
                        'bbox': p['bbox'],
                        'score': p['score'],
                        'segmentation': p['segmentation']
                    })
        return coco_dt
    
    def _evaluate_coco(self, coco_gt: Dict, coco_dt: List[Dict], iou_type: str = 'bbox') -> Dict[str, float]:
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
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
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
            
            # Process segmentation
            segmentation = self._process_segmentation(ann)
            
            # Process area and iscrowd
            area = self._process_area(ann)
            iscrowd = self._process_iscrowd(ann)
            
            processed_annotations.append({
                'id': i,
                'image_id': i,
                'category_id': category_id,
                'bbox': bbox,
                'segmentation': segmentation,
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
    
    def _process_category_id(self, ann: Dict) -> int:
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
    
    def _process_bbox(self, ann: Dict) -> List[float]:
        """Process bounding box from annotation."""
        if isinstance(ann, dict):
            bbox = ann.get('bbox')
            if isinstance(bbox, torch.Tensor):
                if bbox.numel() == 0:
                    return None
                return bbox.cpu().numpy().tolist()
        return [0, 0, 0, 0]
    
    def _process_segmentation(self, ann: Dict) -> List[List[float]]:
        """Process segmentation from annotation."""
        if isinstance(ann, dict):
            segmentation = ann.get('segmentation')
            if isinstance(segmentation, torch.Tensor):
                if segmentation.numel() == 0:
                    return []
                return segmentation.cpu().numpy().tolist()
            return segmentation if segmentation is not None else []
        return []
    
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