# Image Classification Task

This module provides a framework for image classification tasks using various models like ViT and ResNet.

## Architecture

The classification task follows a modular architecture:

1. **Base Components**:
   - `ClassificationProcessor`: Base processor for handling model inputs/outputs
   - `ClassificationMetricLogger`: Handles classification-specific metrics (accuracy, F1, precision, recall)

2. **Model-Specific Processors**:
   - `ViTProcessor`: Processor for Vision Transformer models
   - `ResNetProcessor`: Processor for ResNet models

## Usage Examples

### 1. Using ViT

```python
from tasks.classification.lightning_module import ClassificationModule
from tasks.classification.vit_processor import ViTProcessor
from tasks.classification.classification_metric_logger import ClassificationMetricLogger
from tasks.common.base_module import BaseConfig

# Configure the model
config = BaseConfig(
    model_name="google/vit-base-patch16-224",
    num_classes=1000,
    learning_rate=1e-4,
    image_size=224
)

# Create processor and metric logger
processor = ViTProcessor(config)
metric_logger = ClassificationMetricLogger(config)

# Initialize module
module = ClassificationModule(
    model_ckpt="google/vit-base-patch16-224",
    config=config,
    processor=processor,
    metric_logger=metric_logger
)

# Train the model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(module, datamodule)
```

### 2. Using ResNet

```python
from tasks.classification.lightning_module import ClassificationModule
from tasks.classification.resnet_processor import ResNetProcessor
from tasks.classification.classification_metric_logger import ClassificationMetricLogger
from tasks.common.base_module import BaseConfig

# Configure the model
config = BaseConfig(
    model_name="microsoft/resnet-50",
    num_classes=1000,
    learning_rate=1e-4,
    image_size=224
)

# Create processor and metric logger
processor = ResNetProcessor(config)
metric_logger = ClassificationMetricLogger(config)

# Initialize module
module = ClassificationModule(
    model_ckpt="microsoft/resnet-50",
    config=config,
    processor=processor,
    metric_logger=metric_logger
)

# Train the model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(module, datamodule)
```

## Model-Specific Features

### ViT
- Uses transformer-based architecture
- Patches images into fixed-size tokens
- Position embeddings for spatial information
- Global attention mechanism

### ResNet
- Uses CNN-based architecture
- Residual connections for deep networks
- Standard convolution operations
- Global average pooling

## Metrics

The classification task supports the following metrics:
- Accuracy
- F1 Score
- Precision
- Recall
- Confusion Matrix

## Best Practices

1. **Model Selection**:
   - Use ViT for high accuracy and transfer learning
   - Use ResNet for faster training and inference

2. **Configuration**:
   - Set appropriate image size
   - Configure learning rate based on model size
   - Use data augmentation for better generalization

3. **Data Preparation**:
   - Normalize images appropriately
   - Balance class distribution
   - Use appropriate data augmentation

4. **Training**:
   - Use appropriate batch size
   - Monitor accuracy and loss
   - Use learning rate scheduling
   - Implement early stopping 