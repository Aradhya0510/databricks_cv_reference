# Semantic Segmentation Task

This module provides a framework for semantic segmentation tasks using various models like DeepLabV3 and SegFormer.

## Architecture

The semantic segmentation task follows a modular architecture:

1. **Base Components**:
   - `SemanticSegmentationProcessor`: Base processor for handling model inputs/outputs
   - `SemanticSegmentationMetricLogger`: Handles semantic segmentation-specific metrics (IoU, Dice)

2. **Model-Specific Processors**:
   - `DeepLabV3Processor`: Processor for DeepLabV3 models
   - `SegFormerProcessor`: Processor for SegFormer models

## Usage Examples

### 1. Using DeepLabV3

```python
from tasks.semantic_segmentation.lightning_module import SemanticSegmentationModule
from tasks.semantic_segmentation.deeplabv3_processor import DeepLabV3Processor
from tasks.semantic_segmentation.semantic_segmentation_metric_logger import SemanticSegmentationMetricLogger
from tasks.common.base_module import BaseConfig

# Configure the model
config = BaseConfig(
    model_name="nvidia/mit-b0",
    num_classes=19,
    learning_rate=1e-4,
    image_size=512
)

# Create processor and metric logger
processor = DeepLabV3Processor(config)
metric_logger = SemanticSegmentationMetricLogger(config)

# Initialize module
module = SemanticSegmentationModule(
    model_ckpt="nvidia/mit-b0",
    config=config,
    processor=processor,
    metric_logger=metric_logger
)

# Train the model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(module, datamodule)
```

### 2. Using SegFormer

```python
from tasks.semantic_segmentation.lightning_module import SemanticSegmentationModule
from tasks.semantic_segmentation.segformer_processor import SegFormerProcessor
from tasks.semantic_segmentation.semantic_segmentation_metric_logger import SemanticSegmentationMetricLogger
from tasks.common.base_module import BaseConfig

# Configure the model
config = BaseConfig(
    model_name="nvidia/mit-b0",
    num_classes=19,
    learning_rate=1e-4,
    image_size=512
)

# Create processor and metric logger
processor = SegFormerProcessor(config)
metric_logger = SemanticSegmentationMetricLogger(config)

# Initialize module
module = SemanticSegmentationModule(
    model_ckpt="nvidia/mit-b0",
    config=config,
    processor=processor,
    metric_logger=metric_logger
)

# Train the model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(module, datamodule)
```

## Model-Specific Features

### DeepLabV3
- Atrous spatial pyramid pooling
- Encoder-decoder architecture
- Multi-scale feature fusion
- Efficient feature extraction

### SegFormer
- Hierarchical transformer encoder
- Efficient self-attention
- Overlap patch merging
- MLP decoder

## Metrics

The semantic segmentation task supports the following metrics:
- IoU (Intersection over Union)
- Dice Coefficient
- Pixel Accuracy
- Mean IoU
- Per-class IoU

## Best Practices

1. **Model Selection**:
   - Use DeepLabV3 for high accuracy
   - Use SegFormer for efficient training

2. **Configuration**:
   - Set appropriate image size
   - Configure learning rate based on model size
   - Use data augmentation for better generalization

3. **Data Preparation**:
   - Normalize images appropriately
   - Handle class imbalance
   - Use appropriate data augmentation
   - Process segmentation masks correctly

4. **Training**:
   - Use appropriate batch size
   - Monitor IoU and Dice metrics
   - Use learning rate scheduling
   - Implement early stopping 