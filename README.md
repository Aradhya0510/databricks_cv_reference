# Databricks Computer Vision Reference Architecture

## Project Overview

This project provides a comprehensive reference architecture for implementing computer vision solutions in Databricks. It is designed to bridge the gap between cutting-edge computer vision research and practical business applications by providing:

1. **Accessibility**: A user-friendly interface through Databricks notebooks that allows non-technical users to leverage state-of-the-art computer vision models without deep ML expertise.

2. **Best Practices**: A robust backend implementation that incorporates industry best practices for:
   - Model training and evaluation
   - Data preprocessing and augmentation
   - Hyperparameter tuning
   - Model deployment and monitoring
   - Experiment tracking and model versioning

3. **Extensibility**: A modular architecture that makes it easy to:
   - Add new model architectures
   - Support new datasets
   - Implement custom training routines
   - Integrate with existing ML pipelines

## Technology Stack

The project leverages a carefully selected set of technologies to provide a modern, well-supported, and scalable solution:

1. **MS COCO Dataset Format**
   - Industry standard format for computer vision tasks
   - Comprehensive annotation support for multiple tasks
   - Extensive tooling and community support
   - Easy conversion from other formats

2. **Hugging Face Transformers**
   - State-of-the-art model architectures
   - Regular updates with latest research
   - Extensive model hub with pretrained weights
   - Consistent API across different models

3. **PyTorch Lightning**
   - Clean, modular training code
   - Built-in support for best practices
   - Easy local training setup
   - Seamless integration with distributed training

4. **Ray**
   - Distributed training and hyperparameter tuning
   - Efficient resource utilization
   - Scalable to large clusters
   - Integration with Databricks

5. **MLflow**
   - Experiment tracking and logging
   - Model versioning and registration
   - Integration with Unity Catalog
   - Reproducible experiments

## Getting Started

The project includes a series of notebooks that demonstrate the complete workflow for training a DETR (DEtection TRansformer) model on the COCO 2017 dataset:

1. `00_setup_and_config.ipynb`: Environment setup and configuration
2. `01_data_preparation.ipynb`: Dataset preparation and preprocessing
3. `02_model_training.ipynb`: Model training and evaluation
4. `03_hparam_tuning.ipynb`: Hyperparameter optimization
5. `04_model_evaluation.ipynb`: Comprehensive model evaluation
6. `05_model_registration_deployment.ipynb`: Model registration and deployment
7. `06_model_monitoring.ipynb`: Model monitoring and maintenance

These notebooks provide a step-by-step guide to:
- Setting up the environment
- Preparing and preprocessing data
- Training and evaluating models
- Optimizing hyperparameters
- Registering and deploying models
- Monitoring model performance

## Project Structure

```
Databricks_CV_ref/
├── notebooks/           # Example notebooks
├── src/                # Source code
│   ├── tasks/         # Task-specific implementations
│   ├── training/      # Training utilities
│   └── data/          # Data handling utilities
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Contributing

We welcome contributions to improve the reference architecture. Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 