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
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        return {
            "loss": outputs.loss,
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.logits,
            "loss_dict": outputs.loss_dict
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to DETR format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in model-specific format
        """
        # Convert targets to list of dictionaries
        target_list = []
        current_image_id = targets["image_id"][0]
        current_boxes = []
        current_labels = []
        
        # Get the minimum length to avoid index out of bounds
        min_length = min(
            len(targets["boxes"]),
            len(targets["image_id"]),
            len(targets["class_labels"])
        )
        
        for i in range(min_length):
            if i > 0 and targets["image_id"][i] != current_image_id:
                # Add previous image's targets
                target_list.append({
                    "boxes": torch.stack(current_boxes),
                    "labels": torch.stack(current_labels),
                    "class_labels": torch.stack(current_labels)  # Keep both keys for compatibility
                })
                # Start new image
                current_image_id = targets["image_id"][i]
                current_boxes = []
                current_labels = []
            
            current_boxes.append(targets["boxes"][i])
            current_labels.append(targets["class_labels"][i])
        
        # Add last image's targets
        if current_boxes:
            target_list.append({
                "boxes": torch.stack(current_boxes),
                "labels": torch.stack(current_labels),
                "class_labels": torch.stack(current_labels)  # Keep both keys for compatibility
            })
        
        return target_list
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format DETR outputs for metric computation.
        
        Args:
            outputs: DETR outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        preds = []
        for i in range(len(outputs["pred_boxes"])):
            preds.append({
                "boxes": outputs["pred_boxes"][i],
                "scores": outputs["pred_logits"][i].softmax(dim=-1)[..., :-1].max(dim=-1)[0],
                "labels": outputs["pred_logits"][i].softmax(dim=-1)[..., :-1].argmax(dim=-1)
            })
        return preds
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        # Convert targets to list of dictionaries
        target_list = []
        current_image_id = targets["image_id"][0]
        current_boxes = []
        current_labels = []
        
        # Get the minimum length to avoid index out of bounds
        min_length = min(
            len(targets["boxes"]),
            len(targets["image_id"]),
            len(targets["class_labels"])
        )
        
        for i in range(min_length):
            if i > 0 and targets["image_id"][i] != current_image_id:
                # Add previous image's targets
                target_list.append({
                    "boxes": torch.stack(current_boxes),
                    "labels": torch.stack(current_labels),
                    "class_labels": torch.stack(current_labels)  # Keep both keys for compatibility
                })
                # Start new image
                current_image_id = targets["image_id"][i]
                current_boxes = []
                current_labels = []
            
            current_boxes.append(targets["boxes"][i])
            current_labels.append(targets["class_labels"][i])
        
        # Add last image's targets
        if current_boxes:
            target_list.append({
                "boxes": torch.stack(current_boxes),
                "labels": torch.stack(current_labels),
                "class_labels": torch.stack(current_labels)  # Keep both keys for compatibility
            })
        
        return target_list

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