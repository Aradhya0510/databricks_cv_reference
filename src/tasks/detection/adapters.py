from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch

class OutputAdapter(ABC):
    """Base class for model output adapters."""
    
    @abstractmethod
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model outputs to standard format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Adapted outputs in standard format
        """
        pass
    
    @abstractmethod
    def adapt_targets(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt targets to model-specific format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in model-specific format
        """
        pass
    
    @abstractmethod
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format model outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        pass
    
    @abstractmethod
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        pass

class DETROutputAdapter(OutputAdapter):
    """Adapter for DETR model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt DETR outputs to standard format.
        
        Args:
            outputs: Raw DETR outputs
            
        Returns:
            Adapted outputs in standard format
        """
        # Convert center format boxes to corner format
        pred_boxes = outputs.pred_boxes
        if pred_boxes is not None:
            # Convert from [center_x, center_y, width, height] to [x0, y0, x1, y1]
            pred_boxes = self._center_to_corner(pred_boxes)
        
        return {
            "loss": outputs.loss,
            "pred_boxes": pred_boxes,
            "pred_logits": outputs.logits,
            "loss_dict": outputs.loss_dict
        }
    
    def adapt_targets(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt targets to the format expected by DETR.
        
        Args:
            targets: Dictionary containing:
                - boxes: List of bounding box tensors
                - class_labels: List of class label tensors
                - image_id: Tensor of image IDs
        
        Returns:
            List of target dictionaries in DETR format
        """
        adapted_targets = []
        
        # Get the number of images in the batch
        num_images = len(targets["boxes"])
        
        for i in range(num_images):
            # Get current image's boxes and labels
            boxes = targets["boxes"][i]
            class_labels = targets["class_labels"][i]
            image_id = targets["image_id"][i]
            
            # Handle empty boxes
            if len(boxes) == 0:
                # Create empty tensors with correct shape
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=boxes.device)
                class_labels = torch.zeros(0, dtype=torch.int64, device=class_labels.device)
            else:
                # Convert boxes to center format for DETR
                boxes = self._corner_to_center(boxes)
            
            # Create target dictionary for current image
            target = {
                "boxes": boxes,
                "class_labels": class_labels,  # Keep original key for DETR
                "labels": class_labels,  # Add labels key for compatibility
                "image_id": image_id
            }
            adapted_targets.append(target)
        
        return adapted_targets
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format DETR outputs for metric computation.
        
        Args:
            outputs: DETR outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        preds = []
        for i in range(len(outputs["pred_boxes"])):
            # Ensure tensors are on the same device
            boxes = outputs["pred_boxes"][i]
            logits = outputs["pred_logits"][i]
            
            # Handle empty predictions
            if len(boxes) == 0:
                preds.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=boxes.device),
                    "scores": torch.zeros(0, dtype=torch.float32, device=logits.device),
                    "labels": torch.zeros(0, dtype=torch.int64, device=logits.device)
                })
                continue
            
            # Convert boxes from center format to corner format for mAP
            boxes = self._center_to_corner(boxes)
            
            # Compute scores and labels
            scores = logits.softmax(dim=-1)[..., :-1].max(dim=-1)[0]
            labels = logits.softmax(dim=-1)[..., :-1].argmax(dim=-1)
            
            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })
        return preds
    
    def format_targets(self, targets: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Dictionary containing:
                - boxes: List of bounding box tensors
                - class_labels: List of class label tensors
                - image_id: Tensor of image IDs
        
        Returns:
            List of target dictionaries for each image
        """
        target_list = []
        
        # Get the number of images in the batch
        num_images = len(targets["boxes"])
        
        for i in range(num_images):
            # Get current image's boxes and labels
            boxes = targets["boxes"][i]
            class_labels = targets["class_labels"][i]
            
            # Handle empty boxes
            if len(boxes) == 0:
                target_list.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=boxes.device),
                    "labels": torch.zeros(0, dtype=torch.int64, device=class_labels.device)
                })
                continue
            
            # Convert boxes from center format to corner format for mAP
            boxes = self._center_to_corner(boxes)
            
            target_list.append({
                "boxes": boxes,
                "labels": class_labels
            })
        
        return target_list
    
    def _center_to_corner(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [center_x, center_y, width, height] to [x0, y0, x1, y1] format.
        
        Args:
            boxes: Boxes in center format
            
        Returns:
            Boxes in corner format
        """
        x_center, y_center, width, height = boxes.unbind(-1)
        x0 = x_center - width / 2
        y0 = y_center - height / 2
        x1 = x_center + width / 2
        y1 = y_center + height / 2
        return torch.stack([x0, y0, x1, y1], dim=-1)
    
    def _corner_to_center(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [x0, y0, x1, y1] to [center_x, center_y, width, height] format.
        
        Args:
            boxes: Boxes in corner format
            
        Returns:
            Boxes in center format
        """
        x0, y0, x1, y1 = boxes.unbind(-1)
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        width = x1 - x0
        height = y1 - y0
        return torch.stack([center_x, center_y, width, height], dim=-1)

class YOLOOutputAdapter(OutputAdapter):
    """Adapter for YOLO model outputs."""
    
    def adapt_output(self, outputs: Any) -> Dict[str, torch.Tensor]:
        """Adapt YOLO outputs to standard format.
        
        Args:
            outputs: YOLO model outputs
            
        Returns:
            Dictionary with standardized keys
        """
        # Implement YOLO-specific adaptation
        pass

def get_output_adapter(model_name: str) -> OutputAdapter:
    """Get the appropriate output adapter for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Appropriate output adapter
    """
    if "detr" in model_name.lower():
        return DETROutputAdapter()
    elif "yolo" in model_name.lower():
        return YOLOOutputAdapter()
    else:
        raise ValueError(f"No output adapter found for model: {model_name}") 