import torch
import torch.nn as nn
import torchvision.models as models

class ClassificationModel(nn.Module):
    """Classification model based on pre-trained architectures."""
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the classification model.
        
        Args:
            model_name: Name of the backbone architecture (e.g., 'resnet50', 'efficientnet_b0')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super().__init__()
        
        # Get the model architecture
        if hasattr(models, model_name):
            self.backbone = getattr(models, model_name)(pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} not found in torchvision.models")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Modify the final layer for our number of classes
        if model_name.startswith('resnet'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif model_name.startswith('efficientnet'):
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported for classification")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x) 