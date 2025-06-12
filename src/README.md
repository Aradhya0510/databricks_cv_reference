# Technical Overview: Databricks Computer Vision Framework

This document provides a detailed technical walkthrough of the Databricks Computer Vision Framework, designed to give developers and machine learning engineers comprehensive insight into the internal workings of the architecture.

---

## 📂 Project Structure

```
databricks-cv-architecture/
├── configs/                  # YAML configuration files
├── notebooks/                # Interactive Databricks notebooks
├── src/
│   ├── data/                 # Data preparation modules
│   ├── tasks/
│   │   ├── detection/
│   │   │   ├── model.py      # Model definition
│   │   │   ├── data.py       # DataModule definition
│   │   │   └── adapters.py   # Model output adapters
│   └── training/
│       └── trainer.py        # Training orchestration
└── tests/                    # Unit tests
```

---

## 🔍 Detailed Technical Workflow

### 1. **Configuration Management (`configs/`)**

* YAML files define model parameters, dataset paths, hyperparameters, and training strategies.
* Ensures reproducibility and ease of experimentation.

---

### 2. **Data Management (`DetectionDataModule` - `data.py`)**

* Built on PyTorch Lightning's `LightningDataModule` abstraction, `DetectionDataModule` encapsulates all logic related to dataset preparation and loading, ensuring a clean separation from the model and training logic.

* This modular structure ensures:

  * A consistent API for `setup()`, `train_dataloader()`, `val_dataloader()`, and `test_dataloader()`.
  * Simplified debugging, testing, and substitution of datasets.
  * Cleaner integration with PyTorch Lightning Trainer and distributed training frameworks like Ray.

* The dataset used within this module expects:

  * A folder of images.
  * A COCO-format JSON annotation file.

* By adhering to the COCO format (via `pycocotools`), the framework standardizes how datasets are interpreted and evaluated. This standardization ensures:

  * Compatibility with popular model architectures like DETR and YOLO.
  * Consistency in evaluation metrics (e.g., mAP, IoU).
  * Interchangeability of datasets without requiring changes to model logic.

#### Key methods:

* `setup(stage)`: Initializes training, validation, and testing datasets using specified transformations (Albumentations or Hugging Face extractors).

* `train_dataloader()`, `val_dataloader()`, `test_dataloader()`: Wrap the datasets into PyTorch `DataLoader` objects, configuring batch size, shuffling, worker count, and custom collation logic.

* `_collate_fn(batch)`: Custom batch collation method that ensures inputs and targets are organized in a format compatible with the model and adapter expectations.

* Manages data loading and preprocessing.

#### Key methods:

* `setup(stage)`: Initializes training, validation, and testing datasets using specified transformations (Albumentations or Hugging Face).
* `train_dataloader()`, `val_dataloader()`, `test_dataloader()`: Create DataLoader instances for each stage.
* `_collate_fn(batch)`: Custom batch collation method, preparing batched images and annotations for model input.

---

### 3. **Model Management (`DetectionModel` - `model.py`)**

* The `DetectionModel` class is a subclass of PyTorch Lightning's `LightningModule`, a standardized interface that structures model training, validation, and testing in a highly organized and modular fashion.

* By extending `LightningModule`, the model encapsulates key responsibilities:

  * Model architecture instantiation
  * Forward pass
  * Loss computation
  * Optimizer and learning rate scheduler definition
  * Logging and metric tracking

#### Key Components and Their Roles:

* `_init_model()`:

  * Leverages Hugging Face's `AutoModelForObjectDetection` to dynamically load a pretrained architecture based on configuration.
  * Injects task-specific parameters such as the number of classes, confidence thresholds, and max detections.
  * Ensures standardized instantiation across various object detection models.

* `_init_metrics()`:

  * Initializes evaluation metrics using `torchmetrics.detection.MeanAveragePrecision`.
  * One instance each is created for training, validation, and testing.
  * This provides a consistent interface for computing mAP, mAR, and related statistics across epochs.

* `forward(pixel_values, pixel_mask, labels)`:

  * Defines the forward pass through the backbone model.
  * Delegates input/output transformation to an `OutputAdapter` for model-specific pre/post-processing.

* `training_step()`, `validation_step()`, `test_step()`:

  * Implement task-specific logic for each phase.
  * Compute and log primary losses and secondary metrics.
  * Use `self.log()` to interface directly with PyTorch Lightning’s unified logging API.

* `on_train_epoch_end()`, `on_validation_epoch_end()`, `on_test_epoch_end()`:

  * Trigger computation and logging of aggregated metrics at the end of each epoch.
  * Supports logging of overall and per-class statistics (mAP, mAR, etc.).

* `configure_optimizers()`:

  * Uses AdamW optimizer by default.
  * Supports cosine annealing learning rate scheduler, configurable via YAML.

* `on_train_epoch_start()`, `on_validation_epoch_start()`, `on_test_epoch_start()`:

  * Reset torchmetrics accumulators to ensure accurate per-epoch tracking.

* `on_save_checkpoint()` and `on_load_checkpoint()`:

  * Customize saving and loading of model state including configuration metadata.
  * Ensures reproducibility and compatibility with model registry and deployment environments.

#### Significance:

* This class offers a powerful encapsulation of training logic, enabling a plug-and-play experience for diverse models.

* Torchmetrics ensures consistent, reliable evaluation across datasets and models.

* PyTorch Lightning’s abstractions reduce boilerplate, enforce best practices, and improve readability and debuggability.

* The custom hooks (e.g., metric resets and checkpoint handling) further refine this control, enabling precision and flexibility in both development and production settings.

* Encapsulates model initialization, forward pass, loss computation, and metrics calculation.

* Built upon Hugging Face Transformers for standardization.

#### Key methods:

* `_init_model()`: Initializes the model architecture and loads pretrained weights.
* `forward(pixel_values, pixel_mask, labels)`: Handles forward propagation through the model, including loss calculation.
* `training_step()`, `validation_step()`, `test_step()`: Implement training, validation, and testing logic, including metric computations.
* `configure_optimizers()`: Sets up optimization algorithms (e.g., AdamW) and learning rate schedulers (e.g., CosineAnnealingLR).
* Lifecycle hooks (`on_train_epoch_start`, `on_validation_epoch_end`, etc.): Reset metrics and log performance.

---

### 4. **Adapter Framework (`OutputAdapter` - `adapters.py`)**

* Ensures modular and extensible integration of diverse Hugging Face models without needing to alter core classes (`DetectionModel` and `DetectionDataModule`).

#### Methods and their significance:

* `adapt_output(outputs)`: Translates raw model outputs into a standardized format, ensuring compatibility across different architectures.
* `adapt_targets(targets)`: Prepares standardized dataset annotations to match model-specific expectations.
* `format_predictions(outputs)`: Formats predictions for consistent evaluation metrics computation (e.g., mean Average Precision).
* `format_targets(targets)`: Structures targets into metric-friendly formats, ensuring accurate evaluation.

---

### 5. **Unified Trainer (`UnifiedTrainer` - `trainer.py`)**

* The `UnifiedTrainer` class is the orchestration layer that stitches together the core components of the framework: PyTorch Lightning for structured model training, Ray for distributed and scalable execution, and MLflow for experiment tracking and logging.

#### How It Works:

* Upon initialization, `UnifiedTrainer` accepts a configuration object, a model instance (LightningModule), a data module (LightningDataModule), and an optional logger.
* Depending on whether distributed training is enabled in the config, it dynamically switches between:

  * A standard PyTorch Lightning `Trainer` instance for local GPU/CPU training.
  * A Ray-integrated `Trainer` using `RayDDPStrategy` and `RayLightningEnvironment` for distributed training across multiple nodes.

#### MLflow Integration:

* The trainer hooks into MLflow via Lightning's built-in logging support or Ray's `MLflowLoggerCallback`.
* Metrics, parameters, and model checkpoints are automatically tracked, versioned, and stored.

#### Ray Integration:

* For distributed training, the `UnifiedTrainer` uses Ray's `TorchTrainer` to execute training across workers.
* `setup_ray_cluster()` is used to provision the compute environment dynamically inside a Databricks cluster.
* Ray Tune is also used for efficient hyperparameter optimization with ASHAScheduler.

#### Key Methods and Responsibilities:

* `_init_callbacks()`:

  * Sets up early stopping and model checkpointing logic.
  * Includes `RayTrainReportCallback` if distributed mode is active.

* `_init_trainer()`:

  * Initializes the PyTorch Lightning `Trainer` with the appropriate strategy, accelerator, device settings, and logger.
  * Prepares the trainer via Ray's `prepare_trainer()` method when in distributed mode.

* `train()`:

  * Launches model training using either local or Ray-based Trainer.
  * For Ray: defines a lambda wrapper to run `trainer.fit()` inside a `TorchTrainer` context.

* `tune(search_space, num_trials)`:

  * Uses Ray Tune for hyperparameter optimization.
  * Sets up a `train_func` that dynamically updates the configuration and calls `train()` inside Ray Tune's search loop.
  * MLflow logging callback is included to track every trial.

* `get_metrics()`:

  * Uses MLflow client to retrieve and expose the final run's tracked metrics.

#### Significance:

* `UnifiedTrainer` brings together multiple technologies under a unified, clean abstraction.

* Enables consistent execution whether on a single machine or distributed GPU cluster.

* Keeps the logic decoupled from model and data classes, preserving the modular design of the framework.

* Manages the overall training process, handling both local and distributed training setups seamlessly.

#### Key methods:

* `_init_callbacks()`: Initializes checkpointing, early stopping, and reporting callbacks.
* `_init_trainer()`: Configures the PyTorch Lightning Trainer for local or distributed environments using Ray.
* `train()`: Executes the training process, setting up Ray clusters when in distributed mode, and integrates MLflow logging.
* `tune(search_space, num_trials)`: Runs hyperparameter tuning using Ray Tune, leveraging the ASHA scheduler for efficient search.
* `get_metrics()`: Retrieves training metrics logged via MLflow for easy monitoring and reporting.

---

## 🔄 Flow Integration Overview

1. **Configuration files** guide the setup of dataset paths, model parameters, and training strategies.
2. **DataModule** leverages these configurations to prepare standardized data inputs.
3. **DetectionModel** loads configurations to initialize Hugging Face models and handles training logic.
4. **Adapters** manage data and model output compatibility, enabling seamless integration of multiple models without modifications to core logic.
5. **UnifiedTrainer** orchestrates the training workflow, logging results, and optionally enabling distributed training and hyperparameter tuning with Ray.

---

## ⚙️ Adding New Models

To add new models:

* Implement a new `OutputAdapter` subclass in `adapters.py`.
* Register this adapter in the adapter factory function.
* Define your model configuration in a YAML file under `configs/`.

This modular approach eliminates the need for altering core logic in `model.py` or `data.py`, greatly simplifying the introduction and experimentation with new architectures.

---

## 📖 Understanding the Framework

Each class and method in this framework is designed to promote clarity, modularity, and extensibility. By clearly defining roles and responsibilities, the framework makes it intuitive to maintain, extend, and scale computer vision solutions within the Databricks ecosystem.

---

## 🛠️ Contributions and Improvements

Contributions are encouraged. Please adhere to the provided standards, ensure thorough documentation, and write comprehensive unit tests when adding new functionality or modifying existing logic.
