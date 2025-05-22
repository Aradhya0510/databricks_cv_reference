import pytorch_lightning as pl
import torch
import mlflow
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BaseConfig:
    """Base configuration for all vision tasks."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = None

class BaseVisionModule(pl.LightningModule):
    """Shared optimiser, scheduler, and log helpers."""
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

    def configure_optimizers(self):
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return optimizer

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

    def log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics) 