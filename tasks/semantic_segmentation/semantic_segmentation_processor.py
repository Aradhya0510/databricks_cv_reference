from tasks.common.base_module import ModelProcessor
import torch
from typing import Dict, Any, List

class SemanticSegmentationProcessor(ModelProcessor):
    """Base processor for semantic segmentation model inputs and outputs."""
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by the model.
        
        Args:
            batch: Dictionary containing image tensors and masks
            
        Returns:
            Dictionary of model inputs
        """
        # Normalize images to [0, 1] range
        pixel_values = self._preprocess_image(batch['pixel_values'])
        
        return {
            'pixel_values': pixel_values,
            'labels': batch['masks']
        }
    
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary of processed outputs
        """
        # Process raw outputs
        processed = self._postprocess_outputs(outputs)
        
        return {
            'loss': processed['loss'],
            'logits': processed['logits'],
            'predictions': processed['predictions']
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
        # Get predictions from logits
        predictions = torch.argmax(outputs.logits, dim=1)
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'predictions': predictions
        } 