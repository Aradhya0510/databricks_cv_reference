import pytorch_lightning as pl
import torch
import mlflow
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class BaseConfig:
    """Base configuration for all vision tasks."""
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = None
    
    # Model configuration
    model_name: str = "default"
    pretrained: bool = True
    num_classes: Optional[int] = None
    
    # Data processing
    image_size: tuple = (224, 224)
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    
    # Training settings
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: str = "norm"
    accumulate_grad_batches: int = 1
    
    # Logging
    log_every_n_steps: int = 50
    log_metrics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.scheduler and not self.scheduler_params:
            self.scheduler_params = {}

class ModelProcessor(ABC):
    """Abstract base class for model input/output processing."""
    
    def __init__(self):
        """Initialize the model processor."""
        self.config = None
    
    @abstractmethod
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare model inputs from batch."""
        pass
    
    @abstractmethod
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process model outputs to standard format."""
        pass

class MetricLogger:
    """Handles metric computation and logging."""
    
    def __init__(self, metrics: List[str], log_every_n_steps: int = 50):
        self.metrics = metrics
        self.log_every_n_steps = log_every_n_steps
        self.metric_fns = {}
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup metric computation functions."""
        for metric in self.metrics:
            if hasattr(self, f"_compute_{metric}"):
                self.metric_fns[metric] = getattr(self, f"_compute_{metric}")
    
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute all configured metrics."""
        metrics = {}
        for name, fn in self.metric_fns.items():
            metrics[name] = fn(outputs, batch)
        return metrics

class BaseVisionModule(pl.LightningModule):
    """Enhanced base module with common functionality and composition-based approach."""
    
    def __init__(
        self,
        config: BaseConfig,
        model_processor: ModelProcessor,
        metric_logger: Optional[MetricLogger] = None
    ):
        super().__init__()
        self.config = config
        self.model_processor = model_processor
        self.metric_logger = metric_logger
        self.save_hyperparameters()
        
        # Initialize model
        self.model = self._init_model()
        
        # Setup gradient clipping
        if self.config.gradient_clip_val:
            self.grad_clip_val = self.config.gradient_clip_val
            self.grad_clip_algorithm = self.config.gradient_clip_algorithm

    def _init_model(self) -> torch.nn.Module:
        """Initialize the model based on configuration."""
        raise NotImplementedError("Model initialization must be implemented by subclasses")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with input/output processing."""
        # Prepare inputs
        model_inputs = self.model_processor.prepare_inputs(batch)
        
        # Get model outputs
        model_outputs = self.model(**model_inputs)
        
        # Process outputs
        return self.model_processor.process_outputs(model_outputs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Enhanced training step with metric logging."""
        outputs = self(batch)
        loss = self._extract_loss(outputs)
        
        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log additional metrics if configured
        if self.metric_logger and batch_idx % self.config.log_every_n_steps == 0:
            metrics = self.metric_logger.compute_metrics(outputs, batch)
            self._log_metrics("train", metrics)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Enhanced validation step with metric logging."""
        outputs = self(batch)
        loss = self._extract_loss(outputs)
        
        # Log loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log additional metrics
        if self.metric_logger:
            metrics = self.metric_logger.compute_metrics(outputs, batch)
            self._log_metrics("val", metrics)

    def _extract_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract loss from model outputs."""
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        raise ValueError("Model outputs must contain 'loss' key or be a tensor")

    def _log_metrics(self, prefix: str, metrics: Dict[str, float]) -> None:
        """Log metrics with prefix."""
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = self._get_optimizer()
        
        if self.config.scheduler:
            scheduler = self._get_scheduler(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss' if self.config.scheduler == 'plateau' else None
                }
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