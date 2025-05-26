from tasks.classification.classification_processor import ClassificationProcessor
import torch
from typing import Dict, Any, List
from transformers import ViTImageProcessor

class ViTProcessor(ClassificationProcessor):
    """Processor for ViT model inputs and outputs."""
    
    def __init__(self, config: Any):
        """Initialize ViT processor.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(config)
        self.processor = ViTImageProcessor.from_pretrained(config.model_name)
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by ViT.
        
        Args:
            batch: Dictionary containing image tensors and labels
            
        Returns:
            Dictionary of model inputs
        """
        # Process images using ViT processor
        processed = self.processor(
            images=batch['pixel_values'],
            return_tensors="pt"
        )
        
        return {
            'pixel_values': processed.pixel_values,
            'labels': batch['labels']
        }
    
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process ViT model outputs.
        
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