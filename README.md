# Databricks Computer Vision Architecture

A comprehensive computer vision solution built on Databricks, providing end-to-end capabilities for data processing, model training, evaluation, and deployment. This architecture supports HuggingFace models, Albumentations transforms, and strict data validation using Pydantic, with tight integration with Databricks Unity Catalog for data and model management.

## Architecture Overview

This architecture is designed to handle computer vision tasks on Databricks, leveraging distributed computing capabilities, HuggingFace models, and integrated ML tools. The solution supports:

- Image classification
- Object detection
- Image segmentation
- Custom computer vision tasks

### Key Components

1. **Data Processing Pipeline**
   - MS COCO format support
   - Distributed data processing
   - Pydantic data validation
   - Delta Lake integration
   - Unity Catalog integration for data governance

2. **Model Management**
   - HuggingFace model integration
   - Pretrained model support
   - Model factory pattern
   - Type-safe model configuration
   - Unity Catalog integration for model versioning

3. **Transforms**
   - Albumentations integration
   - Task-specific transform pipelines
   - Configurable augmentation strategies
   - Custom transform support

4. **Training Framework**
   - PyTorch Lightning integration
   - Ray distributed training
   - Multi-GPU support
   - Hyperparameter tuning

5. **Experiment Tracking**
   - MLflow integration
   - Comprehensive metrics logging
   - Model versioning
   - Artifact management
   - Unity Catalog integration for metadata

6. **Evaluation Framework**
   - Task-specific metrics
   - Visualization tools
   - Performance analysis
   - A/B testing support

7. **Model Deployment**
   - Databricks Model Serving
   - CI/CD pipeline
   - Monitoring and logging
   - Version control
   - Unity Catalog integration for deployment tracking

## Getting Started

### Prerequisites

- Databricks workspace with ML runtime
- Python 3.8+
- PyTorch 1.8+
- Ray 2.0+
- MLflow 2.0+
- Unity Catalog enabled workspace

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Aradhya0510/databricks-cv-architecture.git
cd databricks-cv-architecture
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Databricks and Unity Catalog:
```python
from databricks.sdk import WorkspaceClient
from data.unity_catalog.catalog_manager import CatalogManager

# Initialize workspace client
workspace = WorkspaceClient(
    host="https://your-workspace.cloud.databricks.com",
    token="your-token"
)

# Initialize catalog manager
catalog_manager = CatalogManager(
    workspace_url="https://your-workspace.cloud.databricks.com",
    token="your-token",
    catalog_name="cv_project",
    schema_name="models"
)

# Create catalog and schema if they don't exist
catalog_manager.create_catalog_if_not_exists()
catalog_manager.create_schema_if_not_exists()
```

### Quick Start

1. Process your dataset with Unity Catalog integration:
```python
from data.processing.coco_processor import COCOProcessor

# Initialize processor with catalog manager
processor = COCOProcessor(spark, catalog_manager)
processor.load_coco_annotations("/path/to/annotations.json")
df = processor.process_images("/path/to/images")

# Save to Delta Lake and register in Unity Catalog
processor.save_to_delta(
    df=df,
    output_path="dbfs:/path/to/data",
    table_name="coco_dataset"
)
```

2. Train and register a model:
```python
from trainer.ray_trainer import RayTrainer
from models.management.model_registry import ModelRegistry

# Initialize model registry
registry = ModelRegistry(
    workspace_url="https://your-workspace.cloud.databricks.com",
    token="your-token",
    catalog_name="cv_project",
    schema_name="models",
    model_name="object_detector"
)

# Train model
trainer = RayTrainer(model, num_workers=4, use_gpu=True)
trainer.train(config)

# Register model in MLflow and Unity Catalog
model_uri = registry.register_model(
    model=model,
    version="1.0",
    metrics={"mAP": 0.85},
    parameters={"learning_rate": 0.001}
)
```

3. Evaluate results:
```python
from evaluation.metrics.classification_metrics import ClassificationMetrics

metrics = ClassificationMetrics(num_classes=10)
results = metrics.compute_metrics(predictions, ground_truth)
```

4. Deploy model with Unity Catalog integration:
```python
from deployment.ci_cd.deployment_pipeline import DeploymentPipeline

pipeline = DeploymentPipeline(
    workspace_url="https://your-workspace.cloud.databricks.com",
    token="your-token",
    catalog_name="cv_project",
    schema_name="models",
    model_name="object_detector",
    experiment_name="detection_experiment"
)

# Deploy model
deployment = pipeline.run_pipeline(
    metrics_threshold={"mAP": 0.8},
    endpoint_name="object_detector_endpoint"
)

# Monitor endpoint
metrics = pipeline.monitor_endpoint("object_detector_endpoint")
```

## Project Structure

```
Databricks_CV_ref/
├── data/
│   ├── processing/
│   │   └── coco_processor.py
│   └── unity_catalog/
│       └── catalog_manager.py
├── models/
│   ├── management/
│   │   └── model_registry.py
│   └── training/
│       └── ray_trainer.py
├── deployment/
│   └── ci_cd/
│       └── deployment_pipeline.py
├── notebooks/
│   ├── 1_data_processing.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_evaluation.ipynb
└── docs/
    ├── data_processing.md
    ├── model_training.md
    └── deployment.md
```

## Documentation

Detailed documentation for each component is available in the `docs` directory:

- [Data Processing](docs/data_processing.md)
- [Model Training](docs/model_training.md)
- [Experiment Tracking](docs/experiment_tracking.md)
- [Evaluation](docs/evaluation.md)
- [Deployment](docs/deployment.md)

## Best Practices

1. **Data Management**
   - Use Delta Lake for versioned data storage
   - Implement data validation checks
   - Monitor data quality metrics
   - Leverage Unity Catalog for data governance

2. **Model Development**
   - Follow modular design principles
   - Implement proper error handling
   - Use type hints and documentation
   - Track model metadata in Unity Catalog

3. **Training**
   - Monitor GPU utilization
   - Implement early stopping
   - Use checkpointing
   - Track experiments in MLflow

4. **Deployment**
   - Implement proper monitoring
   - Use A/B testing for new models
   - Maintain version control
   - Track deployment metrics in Unity Catalog

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 