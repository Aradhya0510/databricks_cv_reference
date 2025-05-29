# Databricks Computer Vision Reference Implementation

A reference implementation of computer vision tasks using PyTorch Lightning and Ray on Databricks.

## Features

- **Task Support**:
  - Object Detection (with Hugging Face model support)
  - Image Classification
  - Semantic Segmentation

- **Training Pipeline**:
  - Unified training interface for all tasks
  - Support for both local multi-GPU and distributed training
  - Ray integration for distributed training on Databricks
  - Hyperparameter tuning with Ray Tune
  - MLflow integration for experiment tracking

- **Data Processing**:
  - Task-specific data modules
  - Support for COCO format datasets
  - Efficient data loading and preprocessing
  - Data augmentation and transformation pipelines

- **Model Export**:
  - ONNX export for production deployment
  - Model versioning and registration
  - Unity Catalog integration

- **Visualization**:
  - Training metrics visualization
  - Prediction visualization
  - Dataset exploration tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Databricks_CV_ref.git
cd Databricks_CV_ref
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For distributed training on Databricks, install Ray on Spark:
```bash
pip install "ray[spark]"
```

## Project Structure

```
Databricks_CV_ref/
├── src/
│   ├── tasks/
│   │   ├── classification/
│   │   ├── detection/
│   │   └── segmentation/
│   ├── training/
│   │   └── trainer.py
│   └── utils/
├── notebooks/
│   ├── 00_setup_and_config.py
│   ├── 01_data_preparation.py
│   ├── 02_model_training.py
│   ├── 03_hparam_tuning.py
│   ├── 04_model_evaluation.py
│   ├── 05_model_registration_deployment.py
│   └── 06_model_monitoring.py
├── configs/
│   ├── classification.yaml
│   ├── detection.yaml
│   └── segmentation.yaml
└── requirements.txt
```

## Usage

### Local Training

For local multi-GPU training:

```python
from src.training.trainer import UnifiedTrainer
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule

# Initialize trainer in local mode
trainer = UnifiedTrainer(
    task="detection",
    model_class=DetectionModel,
    data_module_class=DetectionDataModule,
    config_path="configs/detection.yaml",
    distributed=False  # Use PyTorch Lightning's native DDP
)

# Train the model
trainer.train()
```

### Distributed Training on Databricks

For distributed training on a Databricks Spark cluster:

```python
from src.training.trainer import UnifiedTrainer
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule

# Initialize trainer in distributed mode
trainer = UnifiedTrainer(
    task="detection",
    model_class=DetectionModel,
    data_module_class=DetectionDataModule,
    config_path="configs/detection.yaml",
    distributed=True  # Use Ray for distributed training
)

# Train the model
trainer.train()
```

### Hyperparameter Tuning

```python
# Define search space
search_space = {
    "model": {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3)
    },
    "training": {
        "batch_size": tune.choice([16, 32, 64]),
        "max_epochs": tune.choice([50, 100])
    }
}

# Run hyperparameter tuning
best_config = trainer.tune(
    search_space=search_space,
    num_trials=20
)
```

## Configuration

The training configuration is specified in YAML files under the `configs/` directory. Example configuration for detection:

```yaml
model:
  name: "facebook/detr-resnet-50"
  num_classes: 80
  confidence_threshold: 0.7
  iou_threshold: 0.5
  max_detections: 100

data:
  train_path: "/Volumes/catalog/schema/volume/train"
  val_path: "/Volumes/catalog/schema/volume/val"
  test_path: "/Volumes/catalog/schema/volume/test"
  batch_size: 32
  num_workers: 4
  image_size: [800, 800]

training:
  max_epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-4
  warmup_epochs: 5
  gradient_clip_val: 0.1
  early_stopping_patience: 10
  monitor_metric: "val_map"
  monitor_mode: "max"
  checkpoint_dir: "/dbfs/FileStore/checkpoints"
  log_every_n_steps: 50

ray:
  num_workers: 4
  use_gpu: true
  resources_per_worker:
    CPU: 4
    GPU: 1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Lightning for the training framework
- Ray for distributed training and hyperparameter tuning
- Hugging Face for pre-trained models
- COCO dataset for object detection benchmarks 