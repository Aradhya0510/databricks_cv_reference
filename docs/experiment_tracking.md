# Experiment Tracking Documentation

## Overview

The experiment tracking system leverages MLflow to manage experiments, track metrics, and version models in the Databricks environment.

## Components

### ExperimentTracker

The `ExperimentTracker` class provides a comprehensive interface for experiment tracking.

#### Initialization

```python
from mlflow.tracking.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="cv_experiment",
    tracking_uri="databricks",
    run_name="run_1"
)
```

#### Key Methods

1. **start_run**
   ```python
   def start_run(self) -> None
   ```
   Start a new MLflow run.

2. **log_parameters**
   ```python
   def log_parameters(self, params: Dict[str, Any]) -> None
   ```
   Log training parameters.

3. **log_metrics**
   ```python
   def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None
   ```
   Log training metrics.

4. **log_model**
   ```python
   def log_model(
       self,
       model: torch.nn.Module,
       model_name: str,
       input_example: Optional[torch.Tensor] = None
   ) -> None
   ```
   Log model to MLflow.

## Usage Examples

### Basic Experiment Tracking

```python
# Initialize tracker
tracker = ExperimentTracker(
    experiment_name="cv_experiment",
    run_name="training_run_1"
)

# Start run
tracker.start_run()

# Log parameters
tracker.log_parameters({
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 100
})

# Log metrics
tracker.log_metrics({
    "train_loss": 0.5,
    "val_loss": 0.4
})

# Log model
tracker.log_model(model, "cv_model")

# End run
tracker.end_run()
```

### Custom Metric Tracking

```python
# Log custom metrics
tracker.log_metrics({
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score
})

# Log visualizations
tracker.log_visualization(fig, "confusion_matrix")
```

## Best Practices

1. **Experiment Organization**
   - Use meaningful experiment names
   - Implement proper tagging
   - Maintain experiment hierarchy

2. **Metric Tracking**
   - Log comprehensive metrics
   - Include system metrics
   - Track resource utilization

3. **Model Versioning**
   - Implement proper model versioning
   - Track model dependencies
   - Document model changes

## Configuration

MLflow configuration can be specified in `config/mlflow_config.yaml`:

```yaml
mlflow:
  tracking_uri: "databricks"
  experiment_name: "cv_experiment"
  run_name: "run_1"
  logging:
    metrics:
      - name: "loss"
        type: "float"
      - name: "accuracy"
        type: "float"
    parameters:
      - name: "learning_rate"
        type: "float"
      - name: "batch_size"
        type: "int"
```

## Monitoring

The experiment tracking system includes:

1. **Experiment Metrics**
   - Run duration
   - Resource usage
   - Model performance

2. **System Metrics**
   - Storage usage
   - API calls
   - Response times

## Common Issues and Solutions

1. **Storage Issues**
   - Implement artifact cleanup
   - Monitor storage usage
   - Use efficient storage formats

2. **Performance Issues**
   - Optimize logging frequency
   - Batch metric updates
   - Use efficient serialization

3. **Versioning Issues**
   - Implement proper versioning
   - Track dependencies
   - Document changes 