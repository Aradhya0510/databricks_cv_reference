from typing import Dict, Any, Optional, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import mlflow.pytorch
from mlflow.models import infer_signature
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ModelConfig:
    """Configuration for model training."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    task: str = 'classification'
    optimizer: str = 'adamw'
    scheduler: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = None

class BaseModel(pl.LightningModule, ABC):
    """Base class for all computer vision models."""
    
    def __init__(self, model: nn.Module, config: ModelConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step implementation."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step implementation."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers and learning rate schedulers."""
        optimizer = self._get_optimizer()
        if self.config.scheduler:
            scheduler = self._get_scheduler(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        return optimizer
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer based on configuration."""
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD
        }
        optimizer_cls = optimizers.get(self.config.optimizer.lower())
        if not optimizer_cls:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
            
        return optimizer_cls(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _get_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """Get learning rate scheduler based on configuration."""
        schedulers = {
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
            'step': torch.optim.lr_scheduler.StepLR,
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
        }
        scheduler_cls = schedulers.get(self.config.scheduler.lower())
        if not scheduler_cls:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
            
        return scheduler_cls(
            optimizer,
            **self.config.scheduler_params or {}
        )
    
    @abstractmethod
    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss."""
        pass
    
    def log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics)
        
    def save_to_mlflow(self, path: str) -> None:
        """Save model to MLflow."""
        mlflow.pytorch.log_model(
            self.model,
            path,
            registered_model_name=f"{self.config.task}_model",
            signature=infer_signature(
                torch.randn(1, 3, 224, 224),
                self(torch.randn(1, 3, 224, 224))
            )
        ) 