from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, MetricCollection
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

@dataclass
class ClassificationModelConfig:
    """Configuration for classification model."""
    model_name: str
    num_classes: int
    pretrained: bool = True
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    epochs: int = 10
    class_names: Optional[List[str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None

class ClassificationModel(pl.LightningModule):
    """Base classification model that can work with any Hugging Face model."""
    
    def __init__(self, config: Union[Dict[str, Any], ClassificationModelConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = ClassificationModelConfig(**config)
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Initialize model
        self._init_model()
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_model(self) -> None:
        """Initialize the model architecture."""
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            **self.config.model_kwargs or {}
        )
        
        # Initialize model
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            config=model_config,
            **self.config.model_kwargs or {}
        )
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(model_config.hidden_size, self.config.num_classes)
        )
    
    def _init_metrics(self) -> None:
        """Initialize metrics for training, validation, and testing."""
        metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=self.config.num_classes),
            "f1": F1Score(task="multiclass", num_classes=self.config.num_classes)
        })
        
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, SequenceClassifierOutput]:
        """Forward pass of the model.
        
        Args:
            pixel_values: Input images
            labels: Optional target labels
            
        Returns:
            Model outputs
        """
        # Get model outputs
        outputs = self.model(pixel_values=pixel_values)
        
        # Get pooled output
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )
        
        return logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Update metrics
        self.train_metrics(outputs.logits, batch["labels"])
        
        # Log metrics
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return outputs.loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Update metrics
        self.val_metrics(outputs.logits, batch["labels"])
        
        # Log metrics
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Update metrics
        self.test_metrics(outputs.logits, batch["labels"])
        
        # Log metrics
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
            return [optimizer], [scheduler]
        
        return optimizer
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int,
        **kwargs
    ) -> "ClassificationModel":
        """Create a model from a pretrained checkpoint.
        
        Args:
            model_name: Name or path of the pretrained model
            num_classes: Number of output classes
            **kwargs: Additional arguments for model configuration
            
        Returns:
            Initialized model
        """
        config = ClassificationModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
        return cls(config) 