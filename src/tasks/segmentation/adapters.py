from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn.functional as F

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

class SemanticSegmentationAdapter(OutputAdapter):
    """Adapter for semantic segmentation model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt semantic segmentation outputs to standard format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Adapted outputs in standard format
        """
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "loss_dict": outputs.loss_dict if hasattr(outputs, "loss_dict") else {}
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to semantic segmentation format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in semantic segmentation format
        """
        return {
            "masks": targets["semantic_masks"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format semantic segmentation outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        preds = []
        for i in range(len(outputs["logits"])):
            preds.append({
                "masks": outputs["logits"][i].softmax(dim=1),
                "labels": outputs["logits"][i].argmax(dim=1)
            })
        return preds
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        target_list = []
        for i in range(len(targets["semantic_masks"])):
            target_list.append({
                "masks": targets["semantic_masks"][i],
                "labels": targets["semantic_masks"][i].argmax(dim=0)
            })
        return target_list

class InstanceSegmentationAdapter(OutputAdapter):
    """Adapter for instance segmentation model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt instance segmentation outputs to standard format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Adapted outputs in standard format
        """
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "boxes": outputs.pred_boxes if hasattr(outputs, "pred_boxes") else None,
            "masks": outputs.pred_masks if hasattr(outputs, "pred_masks") else None,
            "loss_dict": outputs.loss_dict if hasattr(outputs, "loss_dict") else {}
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to instance segmentation format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in instance segmentation format
        """
        return {
            "boxes": targets["instance_boxes"],
            "masks": targets["instance_masks"],
            "labels": targets["instance_labels"],
            "instance_ids": targets["instance_ids"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format instance segmentation outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        preds = []
        for i in range(len(outputs["logits"])):
            pred = {
                "boxes": outputs["boxes"][i] if outputs["boxes"] is not None else None,
                "masks": outputs["masks"][i] if outputs["masks"] is not None else None,
                "scores": outputs["logits"][i].softmax(dim=-1),
                "labels": outputs["logits"][i].argmax(dim=-1)
            }
            preds.append(pred)
        return preds
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        target_list = []
        for i in range(len(targets["instance_masks"])):
            target_list.append({
                "boxes": targets["instance_boxes"][i],
                "masks": targets["instance_masks"][i],
                "labels": targets["instance_labels"][i],
                "instance_ids": targets["instance_ids"][i]
            })
        return target_list

class PanopticSegmentationAdapter(OutputAdapter):
    """Adapter for panoptic segmentation model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt panoptic segmentation outputs to standard format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Adapted outputs in standard format
        """
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "boxes": outputs.pred_boxes if hasattr(outputs, "pred_boxes") else None,
            "masks": outputs.pred_masks if hasattr(outputs, "pred_masks") else None,
            "loss_dict": outputs.loss_dict if hasattr(outputs, "loss_dict") else {}
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to panoptic segmentation format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in panoptic segmentation format
        """
        return {
            "semantic_masks": targets["semantic_masks"],
            "instance_masks": targets["instance_masks"],
            "instance_boxes": targets["instance_boxes"],
            "instance_labels": targets["instance_labels"],
            "instance_ids": targets["instance_ids"],
            "is_thing": targets["is_thing"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format panoptic segmentation outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        preds = []
        for i in range(len(outputs["logits"])):
            # Get semantic predictions
            semantic_logits = outputs["logits"][i]
            semantic_masks = semantic_logits.softmax(dim=1)
            semantic_labels = semantic_logits.argmax(dim=1)
            
            # Get instance predictions if available
            instance_boxes = outputs["boxes"][i] if outputs["boxes"] is not None else None
            instance_masks = outputs["masks"][i] if outputs["masks"] is not None else None
            
            preds.append({
                "semantic_masks": semantic_masks,
                "semantic_labels": semantic_labels,
                "instance_boxes": instance_boxes,
                "instance_masks": instance_masks
            })
        return preds
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        target_list = []
        for i in range(len(targets["semantic_masks"])):
            target_list.append({
                "semantic_masks": targets["semantic_masks"][i],
                "instance_masks": targets["instance_masks"][i],
                "instance_boxes": targets["instance_boxes"][i],
                "instance_labels": targets["instance_labels"][i],
                "instance_ids": targets["instance_ids"][i],
                "is_thing": targets["is_thing"][i]
            })
        return target_list

def get_output_adapter(model_name: str, segmentation_type: str = "semantic") -> OutputAdapter:
    """Get the appropriate output adapter for a model.
    
    Args:
        model_name: Name of the model
        segmentation_type: Type of segmentation ("semantic", "instance", or "panoptic")
        
    Returns:
        Appropriate output adapter
    """
    if segmentation_type == "semantic":
        return SemanticSegmentationAdapter()
    elif segmentation_type == "instance":
        return InstanceSegmentationAdapter()
    elif segmentation_type == "panoptic":
        return PanopticSegmentationAdapter()
    else:
        raise ValueError(f"Unknown segmentation type: {segmentation_type}") 