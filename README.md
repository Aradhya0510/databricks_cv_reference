# Databricks Computer Vision Reference Architecture

A modular and extensible computer vision framework built on PyTorch Lightning and Ray, designed for distributed training and hyperparameter optimization.

## Architecture Overview

The framework follows a modular design pattern with base classes and task-specific implementations:

```
tasks/
├── common/                 # Shared components
│   ├── base_dataset.py    # Base dataset for COCO format
│   ├── base_datamodule.py # Base data module
│   └── base_module.py     # Base Lightning module
├── detection/            # Object detection task
│   ├── datamodule.py     # Detection data handling
│   ├── lightning_module.py # Detection model
│   ├── detr_module.py    # DETR implementation
│   └── yolo_module.py    # YOLOv8 implementation
├── instance_segmentation/ # Instance segmentation task
│   ├── datamodule.py     # Instance segmentation data handling
│   ├── lightning_module.py # Instance segmentation model
│   └── mask_rcnn_module.py # Mask R-CNN implementation
└── semantic_segmentation/ # Semantic segmentation task
    ├── datamodule.py     # Semantic segmentation data handling
    ├── lightning_module.py # Semantic segmentation model
    └── deeplabv3_module.py # DeepLabV3 implementation
```

### Core Components

1. **Base Classes**
   - `BaseVisionDataset`: Handles COCO format data loading and preprocessing
   - `BaseVisionDataModule`: Manages data loading and transformations
   - `BaseVisionModule`: Provides common training functionality

2. **Task-Specific Modules**
   - Detection: Object detection with DETR and YOLOv8 models
   - Instance Segmentation: Instance-level segmentation with Mask R-CNN
   - Semantic Segmentation: Pixel-wise segmentation with DeepLabV3

## Features

- **Modular Design**: Easy to extend with new tasks
- **COCO Format Support**: Standardized data format across tasks
- **Distributed Training**: Ray integration for distributed training
- **Hyperparameter Optimization**: Built-in support with Ray Tune
- **MLflow Integration**: Experiment tracking and model management
- **Multi-task Learning**: Support for training multiple tasks simultaneously
- **Advanced Metrics**: Task-specific evaluation metrics (COCO mAP, IoU, Dice)
- **Resource Management**: GPU and distributed training optimization

## Usage

### Object Detection Training

```python
from tasks.detection.datamodule import DetectionDataModule
from tasks.detection.yolo_module import YoloModule

# Initialize data module
data_module = DetectionDataModule(
    train_image_dir="path/to/train/images",
    train_annotation_file="path/to/train/annotations.json",
    val_image_dir="path/to/val/images",
    val_annotation_file="path/to/val/annotations.json"
)

# Initialize model
model = YoloModule(
    model_ckpt="yolov8n.pt",
    num_classes=80
)

# Train
trainer = pl.Trainer(...)
trainer.fit(model, data_module)
```

### Instance Segmentation Training

```python
from tasks.instance_segmentation.datamodule import InstanceSegmentationDataModule
from tasks.instance_segmentation.mask_rcnn_module import MaskRCNNModule

# Initialize data module
data_module = InstanceSegmentationDataModule(
    train_image_dir="path/to/train/images",
    train_annotation_file="path/to/train/annotations.json",
    val_image_dir="path/to/val/images",
    val_annotation_file="path/to/val/annotations.json"
)

# Initialize model
model = MaskRCNNModule(
    model_ckpt="facebook/detr-resnet-50",
    num_classes=80
)

# Train
trainer = pl.Trainer(...)
trainer.fit(model, data_module)
```

### Semantic Segmentation Training

```python
from tasks.semantic_segmentation.datamodule import SemanticSegmentationDataModule
from tasks.semantic_segmentation.deeplabv3_module import DeepLabV3Module

# Initialize data module
data_module = SemanticSegmentationDataModule(
    train_image_dir="path/to/train/images",
    train_mask_dir="path/to/train/masks",
    val_image_dir="path/to/val/images",
    val_mask_dir="path/to/val/masks",
    num_classes=21
)

# Initialize model
model = DeepLabV3Module(
    model_ckpt="microsoft/deeplabv3-resnet-50",
    num_classes=21
)

# Train
trainer = pl.Trainer(...)
trainer.fit(model, data_module)
```

### Multi-task Training

```python
from tasks.common.multitask import MultiTaskModule, MultiTaskConfig

# Configure tasks
config = MultiTaskConfig(
    tasks=["detection", "instance_segmentation"],
    model_checkpoints={
        "detection": "yolov8n.pt",
        "instance_segmentation": "facebook/detr-resnet-50"
    }
)

# Initialize multi-task model
model = MultiTaskModule(config)

# Train
trainer = pl.Trainer(...)
trainer.fit(model, data_module)
```

### Hyperparameter Search

```python
from orchestration.task_runner import run_hyperparameter_search

# Define search space
search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "weight_decay": tune.loguniform(1e-5, 1e-3)
}

# Run search
best_config = run_hyperparameter_search(
    task="detection",
    model_ckpt="yolov8n.pt",
    search_space=search_space,
    num_samples=10
)
```

## Data Format

The framework uses COCO format for annotations:

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "image1.jpg",
            "width": 640,
            "height": 480
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [...],  # For segmentation
            "bbox": [x, y, w, h]    # For detection
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "person"
        }
    ]
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Ray 2.0+
- MLflow 2.0+
- Albumentations
- pycocotools
- ultralytics (for YOLOv8)
- torchvision (for Mask R-CNN)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 