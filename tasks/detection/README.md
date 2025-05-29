# Object Detection Task

This module provides a framework for object detection tasks using various models like DETR and YOLO.

## Architecture

The detection task follows a modular architecture:

1. **Base Components**:
   - `DetectionProcessor`: Base processor for handling model inputs/outputs
   - `DetectionMetricLogger`: Handles detection-specific metrics (mAP, mAP50)

2. **Model-Specific Processors**:
   - `DetrProcessor`: Processor for DETR models
   - `YoloProcessor`: Processor for YOLO models

## Usage Examples

### 1. Using DETR

```python
from tasks.detection.lightning_module import DetectionModule
from tasks.detection.detr_processor import DetrProcessor
from tasks.detection.detection_metric_logger import DetectionMetricLogger
from tasks.common.base_module import BaseConfig

# Configure the model
config = BaseConfig(
    model_name="facebook/detr-resnet-50",
    num_classes=91,
    learning_rate=1e-4,
    confidence_threshold=0.5,
    nms_threshold=0.5
)

# Create processor and metric logger
processor = DetrProcessor(config)
metric_logger = DetectionMetricLogger(config)

# Initialize module
module = DetectionModule(
    model_ckpt="facebook/detr-resnet-50",
    config=config,
    processor=processor,
    metric_logger=metric_logger
)

# Train the model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(module, datamodule)
```

### 2. Using YOLO

```python
from tasks.detection.lightning_module import DetectionModule
from tasks.detection.yolo_processor import YoloProcessor
from tasks.detection.detection_metric_logger import DetectionMetricLogger
from tasks.common.base_module import BaseConfig

# Configure the model
config = BaseConfig(
    model_name="hustvl/yolos-small",
    num_classes=91,
    learning_rate=1e-4,
    confidence_threshold=0.5,
    nms_threshold=0.5
)

# Create processor and metric logger
processor = YoloProcessor(config)
metric_logger = DetectionMetricLogger(config)

# Initialize module
module = DetectionModule(
    model_ckpt="hustvl/yolos-small",
    config=config,
    processor=processor,
    metric_logger=metric_logger
)

# Train the model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(module, datamodule)
```

## Model-Specific Features

### DETR
- Uses transformer-based architecture
- End-to-end object detection
- No need for NMS
- Box format: [x1, y1, x2, y2]

### YOLO
- Uses CNN-based architecture
- Real-time object detection
- Requires NMS
- Box format: [x, y, w, h]

## Metrics

The detection task supports the following metrics:
- mAP (mean Average Precision)
- mAP50 (mAP at IoU=0.50)
- mAP75 (mAP at IoU=0.75)
- mAP for different object sizes (small, medium, large)

## Best Practices

1. **Model Selection**:
   - Use DETR for high accuracy
   - Use YOLO for real-time detection

2. **Configuration**:
   - Set appropriate confidence threshold
   - Adjust NMS threshold based on model
   - Configure learning rate based on model size

3. **Data Preparation**:
   - Ensure COCO format annotations
   - Normalize images appropriately
   - Handle empty annotations

4. **Training**:
   - Use appropriate batch size
   - Monitor mAP metrics
   - Use learning rate scheduling 