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

class ViTOutputAdapter(OutputAdapter):
    """Adapter for ViT model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt ViT outputs to standard format.
        
        Args:
            outputs: Raw ViT outputs
            
        Returns:
            Adapted outputs in standard format
        """
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "loss_dict": outputs.loss_dict if hasattr(outputs, "loss_dict") else {}
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to ViT format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in ViT format
        """
        return {
            "labels": targets["class_labels"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format ViT outputs for metric computation.
        
        Args:
            outputs: ViT outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        preds = []
        for i in range(len(outputs["logits"])):
            preds.append({
                "scores": outputs["logits"][i].softmax(dim=-1),
                "labels": outputs["logits"][i].argmax(dim=-1)
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
        for i in range(len(targets["class_labels"])):
            target_list.append({
                "labels": targets["class_labels"][i]
            })
        return target_list

class ResNetOutputAdapter(OutputAdapter):
    """Adapter for ResNet model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt ResNet outputs to standard format.
        
        Args:
            outputs: Raw ResNet outputs
            
        Returns:
            Adapted outputs in standard format
        """
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "loss_dict": outputs.loss_dict if hasattr(outputs, "loss_dict") else {}
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to ResNet format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in ResNet format
        """
        return {
            "labels": targets["class_labels"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format ResNet outputs for metric computation.
        
        Args:
            outputs: ResNet outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        preds = []
        for i in range(len(outputs["logits"])):
            preds.append({
                "scores": outputs["logits"][i].softmax(dim=-1),
                "labels": outputs["logits"][i].argmax(dim=-1)
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
        for i in range(len(targets["class_labels"])):
            target_list.append({
                "labels": targets["class_labels"][i]
            })
        return target_list

def get_output_adapter(model_name: str) -> OutputAdapter:
    """Get the appropriate output adapter for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Appropriate output adapter
    """
    if "vit" in model_name.lower():
        return ViTOutputAdapter()
    elif "resnet" in model_name.lower():
        return ResNetOutputAdapter()
    else:
        raise ValueError(f"No output adapter found for model: {model_name}") 