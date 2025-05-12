# API Reference

## Data Processing

### COCOProcessor

```python
class COCOProcessor:
    def __init__(self, spark: SparkSession)
    def load_coco_annotations(self, annotation_file: str) -> None
    def process_images(self, image_dir: str) -> pd.DataFrame
    def validate_data(self, df: pyspark.sql.DataFrame) -> Dict[str, List[str]]
    def save_to_delta(self, df: pyspark.sql.DataFrame, output_path: str) -> None
```

### COCODataset

```python
class COCODataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        annotations: List[Dict],
        transform: Optional[A.Compose] = None,
        task: str = 'detection'
    )
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]
```

## Model Training

### BaseModel

```python
class BaseModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        task: str = 'classification'
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor
    def validation_step(self, batch: tuple, batch_idx: int) -> None
    def configure_optimizers(self) -> torch.optim.Optimizer
```

### RayTrainer

```python
class RayTrainer:
    def __init__(
        self,
        model: pl.LightningModule,
        num_workers: int = 4,
        use_gpu: bool = True,
        resources_per_worker: Dict[str, float] = None
    )
    def train(self, config: Dict[str, Any]) -> Dict[str, Any]
```

## Experiment Tracking

### ExperimentTracker

```python
class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "databricks",
        run_name: Optional[str] = None
    )
    def start_run(self) -> None
    def log_parameters(self, params: Dict[str, Any]) -> None
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None
    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_example: Optional[torch.Tensor] = None
    ) -> None
```

## Evaluation

### ClassificationMetrics

```python
class ClassificationMetrics:
    def __init__(self, num_classes: int)
    def compute_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None
    ) -> Dict[str, float]
    def plot_confusion_matrix(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> plt.Figure
```

### DetectionMetrics

```python
class DetectionMetrics:
    def __init__(self, iou_threshold: float = 0.5)
    def compute_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        num_classes: int
    ) -> Dict[str, float]
    def plot_precision_recall_curve(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        class_names: Optional[List[str]] = None
    ) -> plt.Figure
```

### SegmentationMetrics

```python
class SegmentationMetrics:
    def __init__(self, num_classes: int)
    def compute_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, float]
    def plot_segmentation_results(
        self,
        images: List[np.ndarray],
        predictions: List[Dict],
        ground_truth: List[Dict],
        class_names: Optional[List[str]] = None,
        max_images: int = 4
    ) -> plt.Figure
```

## Deployment

### DeploymentPipeline

```python
class DeploymentPipeline:
    def __init__(
        self,
        workspace_url: str,
        token: str,
        model_name: str,
        experiment_name: str
    )
    def run_pipeline(
        self,
        metrics_threshold: Dict[str, float],
        endpoint_name: Optional[str] = None
    ) -> Dict[str, Any]
```

### ModelServer

```python
class ModelServer:
    def __init__(
        self,
        workspace_url: str,
        token: str,
        model_name: str,
        model_version: Optional[str] = None
    )
    def create_endpoint(
        self,
        endpoint_name: str,
        instance_type: str = "Standard_DS3_v2",
        min_instances: int = 1,
        max_instances: int = 5,
        scale_to_zero: bool = True
    ) -> str
    def update_endpoint(
        self,
        endpoint_name: str,
        new_model_version: str,
        traffic_percentage: int = 100
    ) -> None
    def monitor_endpoint(self, endpoint_name: str) -> Dict[str, Any]
```

## Configuration Files

### data_config.yaml
```yaml
data_processing:
  batch_size: 32
  num_workers: 4
  validation:
    min_image_size: 100
    max_image_size: 2000
```

### training_config.yaml
```yaml
training:
  model:
    learning_rate: 1e-4
    weight_decay: 1e-5
  distributed:
    num_workers: 4
    use_gpu: true
```

### deployment_config.yaml
```yaml
deployment:
  endpoint:
    instance_type: "Standard_DS3_v2"
    min_instances: 1
    max_instances: 5
  monitoring:
    metrics:
      - latency
      - throughput
      - error_rate
```
```

This completes the comprehensive documentation for the computer vision architecture. The documentation covers:

1. Detailed component descriptions
2. Usage examples
3. Best practices
4. Configuration options
5. Common issues and solutions
6. Complete API reference

Would you like me to:
1. Add more specific examples for certain use cases?
2. Create additional documentation for specific components?
3. Add troubleshooting guides?
4. Create a quick-start guide?

Let me know what aspect you'd like to focus on next! 