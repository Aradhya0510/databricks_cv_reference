from tasks.common.base_module import ModelProcessor
import torch
from typing import Dict, Any, List

class DetectionProcessor(ModelProcessor):
    """Base processor for object detection model inputs and outputs."""
    
    def __init__(self, config: Any):
        """Initialize detection processor.
        
        Args:
            config: Configuration object containing model parameters
        """
        # Initialize the parent class (ModelProcessor)
        ModelProcessor.__init__(self)
        self.config = config
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by the model.
        
        Args:
            batch: Dictionary containing COCO format tensors
            
        Returns:
            Dictionary of model inputs
        """
        # Normalize images to [0, 1] range
        pixel_values = self._preprocess_image(batch['pixel_values'])
        
        return {
            'pixel_values': pixel_values,
            'labels': batch['labels']
        }
    
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process model outputs to COCO format.
        
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
    
    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Image preprocessing.
        
        Args:
            image: Input image tensor
            
        Returns:
            Preprocessed image tensor
        """
        if image.max() > 1.0:
            image = image / 255.0
        return image
    
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
            
            # Apply confidence threshold
            mask = scores > self.config.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        
        return {
            'loss': outputs.loss,
            'predictions': predictions
        }
    
    def _convert_to_coco_format(self, processed: Dict[str, torch.Tensor]) -> List[Dict]:
        """Convert processed outputs to COCO format.
        
        Args:
            processed: Processed model outputs
            
        Returns:
            List of predictions in COCO format
        """
        predictions = []
        for pred in processed['predictions']:
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            
            for box, score, label in zip(boxes, scores, labels):
                predictions.append({
                    'image_id': 0,  # This should be set appropriately
                    'category_id': int(label.item()),
                    'bbox': box.cpu().numpy().tolist(),
                    'score': float(score.item())
                })
        
        return predictions 