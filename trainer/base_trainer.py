from typing import Dict, Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ..schemas.model import ModelConfig, ModelOutput
from ..schemas.data import BatchData
import torch
import mlflow

class BaseTrainer(pl.LightningModule):
    """Base trainer for all computer vision models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.model = model
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass through the model."""
        outputs = self.model(x)
        return ModelOutput(
            predictions=outputs.logits if hasattr(outputs, 'logits') else outputs,
            logits=outputs.logits if hasattr(outputs, 'logits') else None,
            features=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        )
    
    def training_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        """Training step implementation."""
        outputs = self(batch.images)
        loss = self.compute_loss(outputs, batch.targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: BatchData, batch_idx: int) -> None:
        """Validation step implementation."""
        outputs = self(batch.images)
        loss = self.compute_loss(outputs, batch.targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    
    def compute_loss(self, outputs: ModelOutput, targets: Any) -> torch.Tensor:
        """Compute task-specific loss."""
        raise NotImplementedError
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer 