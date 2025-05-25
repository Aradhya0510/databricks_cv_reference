# Vision Tasks

This directory contains implementations for various computer vision tasks, built on a common base architecture.

## Directory Structure

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

## Base Components

### BaseVisionDataset

The base dataset class that handles COCO format data loading and preprocessing:

```python
class BaseVisionDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        label_map: Optional[Dict[int, str]] = None
    ):
        # Initialize dataset with COCO format data
```

Key features:
- COCO format annotation loading
- Image preprocessing
- Category mapping
- Transform pipeline support

### BaseVisionDataModule

The base data module that manages data loading and transformations:

```python
class BaseVisionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_image_dir: str,
        train_annotation_file: str,
        val_image_dir: str,
        val_annotation_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None,
        label_map: Optional[Dict[int, str]] = None
    ):
        # Initialize data module
```

Key features:
- Training and validation data management
- Default transform pipelines
- DataLoader configuration
- Worker management

## Task-Specific Implementations

### Detection

The detection task implementation for object detection:

```python
class DetectionDataset(BaseVisionDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image and bounding boxes
        # Apply transforms
        # Return pixel values and target dict
```

```python
class DetectionDataModule(BaseVisionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        # Set up detection datasets
```

Model implementations:
- DETR: Transformer-based end-to-end detection
- YOLOv8: Real-time object detection

### Instance Segmentation

The instance segmentation task implementation:

```python
class InstanceSegmentationDataset(BaseVisionDataset):
    def _process_segmentation(self, segmentation: List[List[float]], image_size: Tuple[int, int]) -> np.ndarray:
        # Process segmentation polygons into binary masks
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image, boxes, and masks
        # Apply transforms
        # Return pixel values and target dict
```

```python
class InstanceSegmentationDataModule(BaseVisionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        # Set up instance segmentation datasets
```

Model implementation:
- Mask R-CNN: Two-stage detection with instance masks

### Semantic Segmentation

The semantic segmentation task implementation:

```python
class SemanticSegmentationDataset(BaseVisionDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image and semantic mask
        # Apply transforms
        # Return pixel values and mask
```

```python
class SemanticSegmentationDataModule(BaseVisionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        # Set up semantic segmentation datasets
```

Model implementation:
- DeepLabV3: Atrous convolutions for multi-scale feature extraction

## Usage Examples

### Detection

```python
# Initialize detection dataset
dataset = DetectionDataset(
    image_dir="path/to/images",
    annotation_file="path/to/annotations.json"
)

# Initialize detection data module
data_module = DetectionDataModule(
    train_image_dir="path/to/train/images",
    train_annotation_file="path/to/train/annotations.json",
    val_image_dir="path/to/val/images",
    val_annotation_file="path/to/val/annotations.json"
)

# Initialize YOLOv8 model
model = YoloModule(
    model_ckpt="yolov8n.pt",
    num_classes=80
)
```

### Instance Segmentation

```python
# Initialize instance segmentation dataset
dataset = InstanceSegmentationDataset(
    image_dir="path/to/images",
    annotation_file="path/to/annotations.json"
)

# Initialize instance segmentation data module
data_module = InstanceSegmentationDataModule(
    train_image_dir="path/to/train/images",
    train_annotation_file="path/to/train/annotations.json",
    val_image_dir="path/to/val/images",
    val_annotation_file="path/to/val/annotations.json"
)

# Initialize Mask R-CNN model
model = MaskRCNNModule(
    model_ckpt="facebook/detr-resnet-50",
    num_classes=80
)
```

### Semantic Segmentation

```python
# Initialize semantic segmentation dataset
dataset = SemanticSegmentationDataset(
    image_dir="path/to/images",
    mask_dir="path/to/masks",
    num_classes=21
)

# Initialize semantic segmentation data module
data_module = SemanticSegmentationDataModule(
    train_image_dir="path/to/train/images",
    train_mask_dir="path/to/train/masks",
    val_image_dir="path/to/val/images",
    val_mask_dir="path/to/val/masks",
    num_classes=21
)

# Initialize DeepLabV3 model
model = DeepLabV3Module(
    model_ckpt="microsoft/deeplabv3-resnet-50",
    num_classes=21
)
```

## Adding New Tasks

To add a new task:

1. Create a new directory under `tasks/` for your task
2. Create task-specific dataset and data module classes
3. Inherit from base classes
4. Implement task-specific logic
5. Add task-specific transforms and metrics

Example:

```python
from ..common.base_dataset import BaseVisionDataset
from ..common.base_datamodule import BaseVisionDataModule

class NewTaskDataset(BaseVisionDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Implement task-specific data loading
        pass

class NewTaskDataModule(BaseVisionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        # Implement task-specific setup
        pass
``` 