from tasks.semantic_segmentation.lightning_module import SemanticSegmentationModule
import torch
from typing import Dict, Any, List
import torch.nn.functional as F

class DeepLabV3Module(SemanticSegmentationModule):
    """DeepLabV3 semantic segmentation module."""
    
    def __init__(self, model_ckpt: str, config: Any = None):
        """Initialize DeepLabV3 module.
        
        Args:
            model_ckpt: Path to model checkpoint or HuggingFace model ID
            config: Optional configuration overrides
        """
        super().__init__(model_ckpt, config)
        self.save_hyperparameters()

    def _prepare_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for DeepLabV3 model.
        
        Args:
            batch: Dictionary containing image tensors and masks
            
        Returns:
            Dictionary of model inputs
        """
        return {
            'pixel_values': batch['pixel_values'],
            'labels': batch['labels'] if 'labels' in batch else None
        }

    def _process_model_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process DeepLabV3 outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary containing loss and predictions
        """
        # Get logits and loss
        logits = outputs.logits
        
        # Resize logits to match input size if needed
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # During training, resize logits to match target size
            if 'labels' in outputs:
                target_size = outputs.labels.shape[-2:]
                logits = F.interpolate(
                    logits,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            loss = outputs.loss
        else:
            # During inference, no loss
            loss = torch.tensor(0.0)
        
        return {
            'loss': loss,
            'logits': logits
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through DeepLabV3 model.
        
        Args:
            batch: Dictionary containing image tensors and masks
            
        Returns:
            Dictionary containing model outputs
        """
        # Prepare inputs
        model_inputs = self._prepare_model_inputs(batch)
        
        # Get model outputs
        outputs = self.model(**model_inputs)
        
        # Process outputs
        return self._process_model_outputs(outputs) 