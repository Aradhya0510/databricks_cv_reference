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

    def _process_model_outputs(self, outputs):
        """Process DETR outputs to COCO format.
        
        Args:
            outputs: Raw DETR model outputs
            
        Returns:
            Dictionary containing loss and predictions in COCO format
        """
        # Get predictions
        pred_logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        
        # Convert to probabilities and get class predictions
        probs = pred_logits.softmax(-1)
        scores, labels = probs.max(-1)
        
        # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format
        boxes = pred_boxes
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Filter predictions based on confidence threshold
        mask = scores > self.config.confidence_threshold
        filtered_boxes = boxes_xyxy[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        
        # Convert to COCO format
        predictions = []
        for i in range(len(filtered_labels)):
            predictions.append({
                'image_id': 0,  # Assuming single image for now
                'category_id': int(filtered_labels[i].item()),
                'bbox': filtered_boxes[i].detach().cpu().numpy().tolist(),
                'score': float(filtered_scores[i].item()),
                'area': float((filtered_boxes[i, 2] - filtered_boxes[i, 0]) * 
                            (filtered_boxes[i, 3] - filtered_boxes[i, 1])),
                'iscrowd': 0
            })
        
        return {
            'loss': outputs.loss,  # Include the loss from DETR outputs
            'predictions': predictions
        } 