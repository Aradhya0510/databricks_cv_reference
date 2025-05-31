# Source Code Architecture

This document explains the architecture of the Databricks Computer Vision Reference Implementation, focusing on its modular design and extensibility.

## Core Components

### 1. Tasks Framework

The tasks framework provides a unified interface for different computer vision tasks while maintaining task-specific implementations. Each task (detection, classification, segmentation) follows a consistent structure:

```
tasks/
├── detection/
│   ├── model.py          # Task-specific model implementation
│   ├── adapters.py       # Model output adapters
│   └── data.py          # Data handling
├── classification/
│   ├── model.py
│   ├── adapters.py
│   └── data.py
└── segmentation/
    ├── model.py
    ├── adapters.py
    └── data.py
```

#### Model Architecture
- Each task has a base model class that inherits from `pl.LightningModule`
- Models are designed to work with Hugging Face's model hub
- Common functionality is shared across tasks
- Task-specific logic is encapsulated in separate modules

#### Adapter Pattern
The adapter pattern is used to standardize model outputs and targets across different architectures:

```python
class OutputAdapter(ABC):
    @abstractmethod
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert model outputs to standard format"""
        pass

    @abstractmethod
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert targets to model-specific format"""
        pass
```

This allows:
- Easy addition of new model architectures
- Consistent interface for metrics and evaluation
- Standardized data format for training

### 2. Training Framework

The training framework provides a unified interface for both local and distributed training:

```
training/
├── trainer.py           # Main trainer implementation
├── local_trainer.py    # Local training utilities
└── distributed_trainer.py  # Ray-based distributed training
```

#### Unified Trainer
- Supports both local and distributed training
- Handles hyperparameter tuning
- Integrates with MLflow for experiment tracking
- Manages model checkpoints and logging

#### Distributed Training
- Uses Ray for distributed training on Databricks
- Efficient resource utilization
- Scalable to large clusters
- Supports hyperparameter tuning

### 3. Data Handling

The data handling layer provides a standardized interface for working with different datasets:

```
data/
├── coco_handler.py     # COCO format handling
├── dataset.py         # Base dataset class
└── transforms.py      # Data augmentation
```

#### COCO Handler
- Standard format for all computer vision tasks
- Supports multiple annotation types
- Easy conversion from other formats
- Efficient data loading and preprocessing

## Extensibility

### Adding New Tasks

1. Create a new task directory under `tasks/`
2. Implement the required components:
   - Model class
   - Output adapters
   - Data handling

### Adding New Models

1. Create a new adapter in the task's `adapters.py`
2. Implement the required adapter methods
3. Register the adapter in the `get_output_adapter` function

Example:
```python
class NewModelAdapter(OutputAdapter):
    def adapt_output(self, outputs):
        # Convert model outputs to standard format
        pass

    def adapt_targets(self, targets):
        # Convert targets to model format
        pass
```

### Adding New Datasets

1. Convert dataset to COCO format
2. Use the COCO handler for data loading
3. Implement task-specific preprocessing if needed

## Best Practices

1. **Modularity**
   - Each component has a single responsibility
   - Clear interfaces between components
   - Easy to test and maintain

2. **Extensibility**
   - Adapter pattern for model outputs
   - Standardized data format
   - Consistent training interface

3. **Reproducibility**
   - MLflow integration for experiment tracking
   - Versioned model checkpoints
   - Configurable training parameters

4. **Performance**
   - Efficient data loading
   - Distributed training support
   - GPU optimization

## Usage Examples

### Training a New Model

```python
from src.training.trainer import UnifiedTrainer
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule

trainer = UnifiedTrainer(
    task="detection",
    model_class=DetectionModel,
    data_module_class=DetectionDataModule,
    config_path="configs/detection.yaml"
)

trainer.train()
```

### Adding a New Model Architecture

```python
# In tasks/detection/adapters.py
class NewModelAdapter(OutputAdapter):
    def adapt_output(self, outputs):
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "boxes": outputs.pred_boxes
        }

    def adapt_targets(self, targets):
        return {
            "boxes": targets["boxes"],
            "labels": targets["labels"]
        }

# Register the adapter
def get_output_adapter(model_name: str) -> OutputAdapter:
    if "new_model" in model_name.lower():
        return NewModelAdapter()
    # ... other adapters
``` 