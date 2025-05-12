from typing import Dict, Any, Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
import mlflow.pytorch
from mlflow.models import infer_signature

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        task: str = 'classification'
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task = task
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss."""
        raise NotImplementedError
    
    def log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics)
        
    def save_to_mlflow(self, path: str) -> None:
        """Save model to MLflow."""
        mlflow.pytorch.log_model(
            self.model,
            path,
            registered_model_name=f"{self.task}_model",
            signature=infer_signature(
                torch.randn(1, 3, 224, 224),
                self(torch.randn(1, 3, 224, 224))
            )
        ) 