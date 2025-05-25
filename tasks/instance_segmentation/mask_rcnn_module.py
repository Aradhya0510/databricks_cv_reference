from tasks.instance_segmentation.lightning_module import InstanceSegmentationModule
import torch
from typing import Dict, Any, List
import numpy as np

class MaskRCNNModule(InstanceSegmentationModule):
    """Mask R-CNN instance segmentation module."""
    
    def __init__(self, model_ckpt: str, config: Any = None):
        """Initialize Mask R-CNN module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
        """
        super().__init__(model_ckpt, config)
        self.save_hyperparameters()

    def _prepare_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for Mask R-CNN model.
        
        Args:
            batch: Dictionary containing image tensors and annotations
            
        Returns:
            Dictionary of model inputs
        """
        # Mask R-CNN expects images and targets
        return {
            'images': batch['pixel_values'],
            'targets': [{
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
            } for boxes, labels, masks, image_id, area, iscrowd in zip(
                batch['boxes'],
                batch['labels'],
                batch['masks'],
                batch['image_id'],
                batch['area'],
                batch['iscrowd']
            )]
        }

    def _process_model_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process Mask R-CNN outputs to COCO format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary containing loss and predictions in COCO format
        """
        predictions = []
        
        # Process each image's predictions
        for i, pred in enumerate(outputs):
            if pred is None:
                continue
                
            # Get boxes, scores, labels, and masks
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            masks = pred['masks'].cpu().numpy()
            
            # Filter predictions based on confidence threshold
            keep = scores > self.config.confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]
            
            # Convert to COCO format
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                # Convert box from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                # Convert mask to RLE format
                mask = (mask > self.config.mask_threshold).astype(np.uint8)
                from pycocotools import mask as coco_mask
                rle = coco_mask.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                predictions.append({
                    'image_id': i,
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score),
                    'segmentation': rle
                })
        
        return {
            'loss': outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0),
            'predictions': predictions
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through Mask R-CNN model.
        
        Args:
            batch: Dictionary containing image tensors and annotations
            
        Returns:
            Dictionary containing model outputs
        """
        # Prepare inputs
        model_inputs = self._prepare_model_inputs(batch)
        
        # Get model outputs
        if self.training:
            # During training, we need to compute loss
            outputs = self.model(**model_inputs)
        else:
            # During inference, we only need predictions
            outputs = self.model(model_inputs['images'])
        
        # Process outputs
        return self._process_model_outputs(outputs) 