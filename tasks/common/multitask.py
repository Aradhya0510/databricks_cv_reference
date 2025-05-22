from typing import Dict, Any, List, Optional
import torch
import pytorch_lightning as pl
from .base_module import BaseVisionModule, BaseConfig
from .factory import make_module

class MultiTaskConfig(BaseConfig):
    """Configuration for multitask learning."""
    tasks: List[str]
    model_checkpoints: Dict[str, str]
    task_weights: Optional[Dict[str, float]] = None
    shared_backbone: bool = False

class MultiTaskModule(BaseVisionModule):
    """Module for training multiple vision tasks simultaneously."""
    
    def __init__(self, config: MultiTaskConfig):
        """Initialize multitask module.
        
        Args:
            config: Configuration for multitask learning
        """
        super().__init__(config)
        self.tasks = config.tasks
        self.task_weights = config.task_weights or {task: 1.0 for task in self.tasks}
        
        # Create task-specific modules
        self.modules = {}
        for task in self.tasks:
            self.modules[task] = make_module(
                task=task,
                model_ckpt=config.model_checkpoints[task]
            )
        
        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass through all task modules.
        
        Args:
            batch: Dictionary containing input tensors for each task
            
        Returns:
            Dictionary of task-specific outputs
        """
        outputs = {}
        for task in self.tasks:
            if task in batch:
                outputs[task] = self.modules[task](batch[task])
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step implementation.
        
        Args:
            batch: Dictionary containing input tensors for each task
            batch_idx: Index of the current batch
            
        Returns:
            Combined loss tensor
        """
        outputs = self(batch)
        total_loss = 0.0
        
        for task in self.tasks:
            if task in outputs:
                task_loss = outputs[task].loss
                weighted_loss = task_loss * self.task_weights[task]
                total_loss += weighted_loss
                self.log(f"train_{task}_loss", task_loss, on_step=True, on_epoch=True)
        
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step implementation.
        
        Args:
            batch: Dictionary containing input tensors for each task
            batch_idx: Index of the current batch
        """
        outputs = self(batch)
        total_loss = 0.0
        
        for task in self.tasks:
            if task in outputs:
                task_loss = outputs[task].loss
                weighted_loss = task_loss * self.task_weights[task]
                total_loss += weighted_loss
                self.log(f"val_{task}_loss", task_loss, on_step=True, on_epoch=True)
                
                # Log task-specific metrics
                if hasattr(self.modules[task], "_log_metrics"):
                    self.modules[task]._log_metrics(outputs[task], batch[task])
        
        self.log("val_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizers for all tasks."""
        # Collect all parameters
        params = []
        for module in self.modules.values():
            params.extend(module.parameters())
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Add scheduler if configured
        if self.config.scheduler:
            scheduler = self._get_scheduler(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        
        return optimizer 