# Evaluation Framework Documentation

## Overview

The evaluation framework provides comprehensive tools for assessing model performance across different computer vision tasks, including classification, object detection, and segmentation.

## Components

### ClassificationMetrics

The `ClassificationMetrics` class handles evaluation for classification tasks.

#### Initialization

```python
from evaluation.metrics.classification_metrics import ClassificationMetrics

metrics = ClassificationMetrics(num_classes=10)
```

#### Key Methods

1. **compute_metrics**
   ```python
   def compute_metrics(
       self,
       y_true: torch.Tensor,
       y_pred: torch.Tensor,
       y_prob: Optional[torch.Tensor] = None
   ) -> Dict[str, float]
   ```
   Compute classification metrics including accuracy, precision, recall, and F1 score.

2. **plot_confusion_matrix**
   ```python
   def plot_confusion_matrix(
       self,
       y_true: torch.Tensor,
       y_pred: torch.Tensor,
       class_names: Optional[List[str]] = None
   ) -> plt.Figure
   ```
   Generate confusion matrix visualization.

3. **plot_roc_curve**
   ```python
   def plot_roc_curve(
       self,
       y_true: torch.Tensor,
       y_prob: torch.Tensor,
       class_names: Optional[List[str]] = None
   ) -> plt.Figure
   ```
   Generate ROC curve visualization.

### DetectionMetrics

The `DetectionMetrics` class handles evaluation for object detection tasks.

#### Initialization

```python
from evaluation.metrics.detection_metrics import DetectionMetrics

metrics = DetectionMetrics(iou_threshold=0.5)
```

#### Key Methods

1. **compute_metrics**
   ```python
   def compute_metrics(
       self,
       predictions: List[Dict],
       ground_truth: List[Dict],
       num_classes: int
   ) -> Dict[str, float]
   ```
   Compute detection metrics including mAP, precision, and recall.

2. **plot_precision_recall_curve**
   ```python
   def plot_precision_recall_curve(
       self,
       predictions: List[Dict],
       ground_truth: List[Dict],
       class_names: Optional[List[str]] = None
   ) -> plt.Figure
   ```
   Generate precision-recall curve visualization.

### SegmentationMetrics

The `SegmentationMetrics` class handles evaluation for segmentation tasks.

#### Initialization

```python
from evaluation.metrics.segmentation_metrics import SegmentationMetrics

metrics = SegmentationMetrics(num_classes=10)
```

#### Key Methods

1. **compute_metrics**
   ```python
   def compute_metrics(
       self,
       predictions: List[Dict],
       ground_truth: List[Dict]
   ) -> Dict[str, float]
   ```
   Compute segmentation metrics including IoU, pixel accuracy, and dice coefficient.

2. **plot_segmentation_results**
   ```python
   def plot_segmentation_results(
       self,
       images: List[np.ndarray],
       predictions: List[Dict],
       ground_truth: List[Dict],
       class_names: Optional[List[str]] = None,
       max_images: int = 4
   ) -> plt.Figure
   ```
   Generate segmentation visualization.

## Usage Examples

### Classification Evaluation

```python
# Initialize metrics
metrics = ClassificationMetrics(num_classes=10)

# Compute metrics
results = metrics.compute_metrics(y_true, y_pred, y_prob)
print("Classification metrics:", results)

# Generate visualizations
confusion_matrix = metrics.plot_confusion_matrix(y_true, y_pred, class_names)
roc_curve = metrics.plot_roc_curve(y_true, y_prob, class_names)
```

### Object Detection Evaluation

```python
# Initialize metrics
metrics = DetectionMetrics(iou_threshold=0.5)

# Compute metrics
results = metrics.compute_metrics(predictions, ground_truth, num_classes=10)
print("Detection metrics:", results)

# Generate visualizations
pr_curve = metrics.plot_precision_recall_curve(predictions, ground_truth, class_names)
```

### Segmentation Evaluation

```python
# Initialize metrics
metrics = SegmentationMetrics(num_classes=10)

# Compute metrics
results = metrics.compute_metrics(predictions, ground_truth)
print("Segmentation metrics:", results)

# Generate visualizations
segmentation_plot = metrics.plot_segmentation_results(
    images, predictions, ground_truth, class_names
)
```

## Best Practices

1. **Metric Selection**
   - Choose appropriate metrics for the task
   - Consider multiple evaluation criteria
   - Account for class imbalance

2. **Visualization**
   - Generate clear and informative plots
   - Include proper labels and legends
   - Use appropriate color schemes

3. **Performance Analysis**
   - Analyze errors and failure cases
   - Track metric trends over time
   - Compare with baseline models

## Configuration

Evaluation configuration can be specified in `config/evaluation_config.yaml`:

```yaml
evaluation:
  classification:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
    visualization:
      confusion_matrix: true
      roc_curve: true
  detection:
    iou_threshold: 0.5
    metrics:
      - mAP
      - precision
      - recall
    visualization:
      pr_curve: true
      detection_plot: true
  segmentation:
    metrics:
      - iou
      - pixel_accuracy
      - dice
    visualization:
      segmentation_plot: true
```

## Common Issues and Solutions

1. **Memory Issues**
   - Process data in batches
   - Use efficient data structures
   - Implement proper cleanup

2. **Performance Issues**
   - Optimize metric computation
   - Use efficient visualization
   - Implement caching when appropriate

3. **Visualization Issues**
   - Handle large datasets
   - Implement proper scaling
   - Use appropriate color maps 