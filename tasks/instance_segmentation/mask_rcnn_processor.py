from tasks.instance_segmentation.instance_segmentation_processor import InstanceSegmentationProcessor
import torch
from typing import Dict, Any, List
from transformers import DetrImageProcessor

class MaskRCNNProcessor(InstanceSegmentationProcessor):
    """Processor for Mask R-CNN model inputs and outputs."""
    
    def __init__(self, config: Any):
        """Initialize Mask R-CNN processor.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(config)
        self.processor = DetrImageProcessor.from_pretrained(config.model_name)
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by Mask R-CNN.
        
        Args:
            batch: Dictionary containing COCO format tensors
            
        Returns:
            Dictionary of model inputs
        """
        # Process images using Mask R-CNN processor
        processed = self.processor(
            images=batch['pixel_values'],
            annotations=batch['labels'],
            return_tensors="pt"
        )
        
        return {
            'pixel_values': processed.pixel_values,
            'labels': processed.labels
        }
    
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process Mask R-CNN model outputs to COCO format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary of COCO format outputs
        """
        # Process raw outputs
        processed = self._postprocess_outputs(outputs)
        
        # Convert to COCO format
        predictions = self._convert_to_coco_format(processed)
        
        return {
            'loss': processed['loss'],
            'predictions': predictions
        }
    
    def _postprocess_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process raw model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs
        """
        predictions = []
        for pred in outputs.predictions:
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            masks = pred['masks']
            
            # Apply confidence threshold
            mask = scores > self.config.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            masks = masks[mask]
            
            # Apply NMS
            keep = self._nms(boxes, scores)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]
            
            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'masks': masks
            })
        
        return {
            'loss': outputs.loss,
            'predictions': predictions
        }
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply non-maximum suppression.
        
        Args:
            boxes: Bounding boxes
            scores: Confidence scores
            
        Returns:
            Indices of kept boxes
        """
        return torch.ops.torchvision.nms(
            boxes,
            scores,
            self.config.nms_threshold
        ) 