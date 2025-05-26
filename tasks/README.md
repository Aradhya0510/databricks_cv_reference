# Computer Vision Tasks Framework

This framework provides a standardized architecture for various computer vision tasks, making it easy to implement and experiment with different models while maintaining consistent interfaces and evaluation metrics.

## Architecture Overview

The framework follows a modular architecture with the following components:

1. **Base Components**:
   - `BaseConfig`: Common configuration parameters
   - `BaseModule`: PyTorch Lightning module with common functionality
   - `BaseProcessor`: Interface for model-specific processors
   - `MetricLogger`: Base class for task-specific metrics

2. **Task-Specific Components**:
   Each task has its own set of components:
   - Base Processor
   - Base Metric Logger
   - Model-Specific Processors
   - Lightning Module

## Supported Tasks

### 1. Classification
- Base: `ClassificationProcessor`, `ClassificationMetricLogger`
- Models: ViT, ResNet
- Metrics: Accuracy, F1, Precision, Recall
- [More details](classification/README.md)

### 2. Object Detection
- Base: `DetectionProcessor`, `DetectionMetricLogger`
- Models: DETR, YOLO
- Metrics: mAP, mAP50, mAP75
- [More details](detection/README.md)

### 3. Instance Segmentation
- Base: `InstanceSegmentationProcessor`, `InstanceSegmentationMetricLogger`
- Models: Mask R-CNN, DETR with segmentation
- Metrics: mAP, mask mAP
- [More details](instance_segmentation/README.md)

### 4. Semantic Segmentation
- Base: `SemanticSegmentationProcessor`, `SemanticSegmentationMetricLogger`
- Models: DeepLabV3, SegFormer
- Metrics: IoU, Dice, Pixel Accuracy
- [More details](semantic_segmentation/README.md)

## Common Features

1. **Standardized Interfaces**:
   - Consistent input/output formats
   - Common configuration parameters
   - Unified metric computation

2. **Model Support**:
   - HuggingFace Transformers integration
   - Easy addition of new models
   - Model-specific preprocessing

3. **Evaluation**:
   - Task-specific metrics
   - Standardized evaluation protocols
   - Comprehensive logging

4. **Training**:
   - PyTorch Lightning integration
   - Common training loops
   - Distributed training support

## Usage Example

```python
from tasks.common.base_module import BaseConfig
from tasks.detection.lightning_module import DetectionModule
from tasks.detection.detr_processor import DetrProcessor
from tasks.detection.detection_metric_logger import DetectionMetricLogger

# Configure the model
config = BaseConfig(
    model_name="facebook/detr-resnet-50",
    num_classes=91,
    learning_rate=1e-4
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

## Best Practices

1. **Model Selection**:
   - Choose models based on task requirements
   - Consider speed vs. accuracy trade-offs
   - Use appropriate model sizes

2. **Configuration**:
   - Set task-specific parameters
   - Configure model-specific settings
   - Adjust training parameters

3. **Data Preparation**:
   - Follow task-specific formats
   - Use appropriate preprocessing
   - Handle data augmentation

4. **Training**:
   - Monitor task-specific metrics
   - Use appropriate batch sizes
   - Implement early stopping

## Contributing

To add a new model or task:
1. Create model-specific processor
2. Implement task-specific metrics
3. Add example configurations
4. Update documentation

## Future Improvements

1. **Model Support**:
   - Add more model architectures
   - Support custom models
   - Add model zoo

2. **Features**:
   - Add more metrics
   - Improve data augmentation
   - Add model compression

3. **Documentation**:
   - Add more examples
   - Improve API documentation
   - Add tutorials 