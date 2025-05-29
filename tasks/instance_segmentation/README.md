# Instance Segmentation Task

This module provides a framework for instance segmentation tasks using various models like Mask R-CNN and DETR.

## Architecture

The instance segmentation task follows a modular architecture:

1. **Base Components**:
   - `InstanceSegmentationProcessor`: Base processor for handling model inputs/outputs
   - `InstanceSegmentationMetricLogger`: Handles instance segmentation-specific metrics (mAP, mask mAP)

2. **Model-Specific Processors**:
   - `MaskRCNNProcessor`: Processor for Mask R-CNN models
   - `DetrProcessor`: Processor for DETR models with segmentation heads

## Usage Examples

### 1. Using Mask R-CNN

```python
from tasks.instance_segmentation.lightning_module import InstanceSegmentationModule
from tasks.instance_segmentation.mask_rcnn_processor import MaskRCNNProcessor
from tasks.instance_segmentation.instance_segmentation_metric_logger import InstanceSegmentationMetricLogger
from tasks.common.base_module import BaseConfig

# Configure the model
config = BaseConfig(
    model_name="facebook/detr-resnet-50-panoptic",
    num_classes=91,
    learning_rate=1e-4,
    confidence_threshold=0.5,
    nms_threshold=0.5
)

# Create processor and metric logger
processor = MaskRCNNProcessor(config)
metric_logger = InstanceSegmentationMetricLogger(config)

# Initialize module
module = InstanceSegmentationModule(
    model_ckpt="facebook/detr-resnet-50-panoptic",
    config=config,
    processor=processor,
    metric_logger=metric_logger
)

# Train the model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(module, datamodule)
```

### 2. Using DETR with Segmentation

```python
from tasks.instance_segmentation.lightning_module import InstanceSegmentationModule
from tasks.instance_segmentation.detr_processor import DetrProcessor
from tasks.instance_segmentation.instance_segmentation_metric_logger import InstanceSegmentationMetricLogger
from tasks.common.base_module import BaseConfig

# Configure the model
config = BaseConfig(
    model_name="facebook/detr-resnet-50-panoptic",
    num_classes=91,
    learning_rate=1e-4,
    confidence_threshold=0.5,
    nms_threshold=0.5
)

# Create processor and metric logger
processor = DetrProcessor(config)
metric_logger = InstanceSegmentationMetricLogger(config)

# Initialize module
module = InstanceSegmentationModule(
    model_ckpt="facebook/detr-resnet-50-panoptic",
    config=config,
    processor=processor,
    metric_logger=metric_logger
)

# Train the model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(module, datamodule)
```

## Model-Specific Features

### Mask R-CNN
- Two-stage architecture
- Region proposal network
- Mask prediction branch
- Box format: [x, y, w, h]

### DETR with Segmentation
- End-to-end architecture
- Transformer-based
- Parallel mask prediction
- Box format: [x1, y1, x2, y2]

## Metrics

The instance segmentation task supports the following metrics:
- mAP (mean Average Precision)
- mAP50 (mAP at IoU=0.50)
- mAP75 (mAP at IoU=0.75)
- Mask mAP
- Mask mAP50
- Mask mAP75

## Best Practices

1. **Model Selection**:
   - Use Mask R-CNN for high accuracy
   - Use DETR for end-to-end training

2. **Configuration**:
   - Set appropriate confidence threshold
   - Adjust NMS threshold based on model
   - Configure learning rate based on model size

3. **Data Preparation**:
   - Ensure COCO format annotations
   - Normalize images appropriately
   - Handle empty annotations
   - Process mask polygons correctly

4. **Training**:
   - Use appropriate batch size
   - Monitor mAP and mask mAP metrics
   - Use learning rate scheduling
   - Implement early stopping 