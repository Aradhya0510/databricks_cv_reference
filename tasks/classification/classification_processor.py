from tasks.common.base_module import ModelProcessor
import torch
from typing import Dict, Any, List

class ClassificationProcessor(ModelProcessor):
    """Base processor for classification model inputs and outputs."""
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by the model.
        
        Args:
            batch: Dictionary containing image tensors and labels
            
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
        """Process model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary of processed outputs
        """
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'predictions': torch.argmax(outputs.logits, dim=-1)
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