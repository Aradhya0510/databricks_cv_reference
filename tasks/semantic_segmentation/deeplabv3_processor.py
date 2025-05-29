from tasks.semantic_segmentation.semantic_segmentation_processor import SemanticSegmentationProcessor
import torch
from typing import Dict, Any, List
from transformers import SegformerImageProcessor

class DeepLabV3Processor(SemanticSegmentationProcessor):
    """Processor for DeepLabV3 model inputs and outputs."""
    
    def __init__(self, config: Any):
        """Initialize DeepLabV3 processor.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(config)
        self.processor = SegformerImageProcessor.from_pretrained(config.model_name)
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by DeepLabV3.
        
        Args:
            batch: Dictionary containing image tensors and masks
            
        Returns:
            Dictionary of model inputs
        """
        # Process images using DeepLabV3 processor
        processed = self.processor(
            images=batch['pixel_values'],
            segmentation_maps=batch['masks'],
            return_tensors="pt"
        )
        
        return {
            'pixel_values': processed.pixel_values,
            'labels': processed.labels
        }
    
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process DeepLabV3 model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary of processed outputs
        """
        # Get predictions from logits
        predictions = torch.argmax(outputs.logits, dim=1)
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'predictions': predictions
        } 