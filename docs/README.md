# Databricks Computer Vision Pipeline Documentation

## Overview

This documentation provides detailed information about the Databricks Computer Vision Pipeline, a modular framework for training and deploying computer vision models on Databricks.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Inference](#inference)
6. [Monitoring](#monitoring)
7. [Adding New Tasks](#adding-new-tasks)

## Getting Started

### Prerequisites

- Databricks Runtime ML 16.4 GPU
- Python 3.10+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/databricks-cv-pipeline.git
cd databricks-cv-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Databricks:
- Set up Databricks CLI
- Configure Unity Catalog
- Set up MLflow tracking

## Project Structure

```
databricks-cv-pipeline/
├── notebooks/          # Databricks notebooks
├── configs/            # YAML configuration files
├── src/               # Source code
│   ├── tasks/         # Task-specific implementations
│   ├── training/      # Training utilities
│   ├── inference/     # Inference scripts
│   └── utils/         # Common utilities
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Configuration

The pipeline uses YAML configuration files to specify model, training, and deployment parameters. Example configuration:

```yaml
model:
  name: "resnet50"
  pretrained: true
  num_classes: 1000
  dropout: 0.2

training:
  batch_size: 32
  num_workers: 4
  epochs: 100
  learning_rate: 0.001
```

See `configs/` directory for more examples.

## Training

### Local Training

1. Configure your training parameters in `configs/`
2. Run training:
```bash
python -m src.training.train --config configs/classification_resnet50.yaml
```

### Distributed Training on Databricks

1. Use the provided notebooks in `notebooks/`:
   - `00_setup_and_config.ipynb`
   - `01_data_preparation.ipynb`
   - `02_model_training.ipynb`

2. Configure Ray cluster and PyTorch Lightning:
```python
from ray.util.spark import setup_ray_cluster
setup_ray_cluster(num_workers=4, num_gpus_per_worker=1)
```

## Inference

### Batch Inference

1. Use the batch inference script:
```bash
python -m src.inference.batch_inference \
    --config configs/classification_resnet50.yaml \
    --model-uri models:/your-model-uri \
    --input-dir /path/to/images \
    --output-file predictions.json
```

### Real-time Inference

1. Deploy model to Databricks Model Serving:
```python
import mlflow
model = mlflow.pytorch.load_model("models:/your-model-uri")
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=model,
    registered_model_name="your-model-name"
)
```

## Monitoring

### Model Performance

1. Access MLflow tracking UI in Databricks
2. View metrics and parameters
3. Compare model versions

### Drift Detection

1. Enable inference logging in Databricks Model Serving
2. Set up monitoring dashboards
3. Configure drift detection alerts

## Adding New Tasks

To add a new computer vision task:

1. Create a new directory in `src/tasks/`
2. Implement:
   - Model class (inheriting from `pl.LightningModule`)
   - Data module (inheriting from `pl.LightningDataModule`)
   - Configuration template
3. Add tests in `tests/`
4. Update documentation

Example structure for a new task:
```
src/tasks/new_task/
├── __init__.py
├── model.py
├── data.py
└── utils.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 