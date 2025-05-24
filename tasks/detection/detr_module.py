from transformers import AutoImageProcessor
from .lightning_module import DetectionModule
import torch
from typing import Dict, Any

class DetrModule(DetectionModule):
    """DETR-specific implementation of the detection module."""
    
    def __init__(self, model_ckpt: str, config: Any = None):
        """Initialize DETR module.
        
        Args:
            model_ckpt: Path to DETR checkpoint or HuggingFace model ID
            config: Optional configuration overrides
        """
        super().__init__(model_ckpt, config)
        self.image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

    def _prepare_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare COCO format inputs for DETR.
        
        Args:
            batch: Dictionary containing COCO format tensors
            
        Returns:
            Dictionary of DETR inputs
        """
        inputs = {
            "pixel_values": batch["pixel_values"],
            "pixel_mask": torch.ones_like(batch["pixel_values"][:, 0, :, :], dtype=torch.long)
        }
        
        # Add labels if present (for training)
        if "labels" in batch and "boxes" in batch:
            inputs["labels"] = [{
                "class_labels": labels,
                "boxes": boxes
            } for labels, boxes in zip(batch["labels"], batch["boxes"])]
        
        return inputs

    def _process_model_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process DETR outputs back to COCO format.
        
        Args:
            outputs: Raw DETR outputs
            
        Returns:
            Dictionary of COCO format outputs
        """
        # Convert DETR outputs to COCO format
        predictions = []
        for i, (logits, boxes) in enumerate(zip(outputs.logits, outputs.pred_boxes)):
            # Get class predictions
            probs = torch.softmax(logits, dim=-1)
            scores, labels = probs.max(dim=-1)
            
            # Filter by confidence
            mask = scores > self.config.confidence_threshold
            scores = scores[mask]
            labels = labels[mask]
            boxes = boxes[mask]
            
            # Convert to COCO format
            predictions.append({
                'image_id': i,
                'category_id': labels.cpu().numpy().tolist(),
                'bbox': boxes.cpu().numpy().tolist(),
                'score': scores.cpu().numpy().tolist(),
                'area': (boxes[:, 2] * boxes[:, 3]).cpu().numpy().tolist(),
                'iscrowd': [0] * len(labels)
            })
        
        return {'predictions': predictions} 