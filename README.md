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
├── classification/        # Classification task
│   ├── datamodule.py     # Classification data handling
│   └── lightning_module.py # Classification model
└── segmentation/         # Segmentation task
    ├── datamodule.py     # Segmentation data handling
    └── lightning_module.py # Segmentation model
```

### Core Components

1. **Base Classes**
   - `BaseVisionDataset`: Handles COCO format data loading and preprocessing
   - `BaseVisionDataModule`: Manages data loading and transformations
   - `BaseVisionModule`: Provides common training functionality

2. **Task-Specific Modules**
   - Classification: Image classification using HuggingFace models
   - Segmentation: Semantic segmentation with COCO format support
   - Detection: Object detection with DETR models

## Features

- **Modular Design**: Easy to extend with new tasks
- **COCO Format Support**: Standardized data format across tasks
- **Distributed Training**: Ray integration for distributed training
- **Hyperparameter Optimization**: Built-in support with Ray Tune
- **MLflow Integration**: Experiment tracking and model management
- **Multi-task Learning**: Support for training multiple tasks simultaneously

## Usage

### Single Task Training

```python
from tasks.classification.datamodule import ClassificationDataModule
from tasks.classification.lightning_module import ClassificationModule

# Initialize data module
data_module = ClassificationDataModule(
    train_image_dir="path/to/train/images",
    train_annotation_file="path/to/train/annotations.json",
    val_image_dir="path/to/val/images",
    val_annotation_file="path/to/val/annotations.json"
)

# Initialize model
model = ClassificationModule(
    model_ckpt="microsoft/resnet-50",
    num_labels=1000
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
    tasks=["classification", "detection"],
    model_checkpoints={
        "classification": "microsoft/resnet-50",
        "detection": "facebook/detr-resnet-50"
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
    task="classification",
    model_ckpt="microsoft/resnet-50",
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 