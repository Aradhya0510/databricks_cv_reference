# Vision Tasks

This directory contains implementations for various computer vision tasks, built on a common base architecture.

## Directory Structure

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

### Classification

The classification task implementation for image classification:

```python
class ClassificationDataset(BaseVisionDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image and category ID
        # Apply transforms
        # Return pixel values and label
```

```python
class ClassificationDataModule(BaseVisionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        # Set up classification datasets
```

### Segmentation

The segmentation task implementation for semantic segmentation:

```python
class SegmentationDataset(BaseVisionDataset):
    def _create_semantic_mask(self, image_size: Tuple[int, int], annotations: List[Dict[str, Any]]) -> np.ndarray:
        # Create semantic mask from COCO annotations
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image and mask
        # Apply transforms
        # Return pixel values and mask
```

```python
class SegmentationDataModule(BaseVisionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        # Set up segmentation datasets
```

## Usage Examples

### Classification

```python
# Initialize classification dataset
dataset = ClassificationDataset(
    image_dir="path/to/images",
    annotation_file="path/to/annotations.json"
)

# Initialize classification data module
data_module = ClassificationDataModule(
    train_image_dir="path/to/train/images",
    train_annotation_file="path/to/train/annotations.json",
    val_image_dir="path/to/val/images",
    val_annotation_file="path/to/val/annotations.json"
)
```

### Segmentation

```python
# Initialize segmentation dataset
dataset = SegmentationDataset(
    image_dir="path/to/images",
    annotation_file="path/to/annotations.json",
    ignore_index=255
)

# Initialize segmentation data module
data_module = SegmentationDataModule(
    train_image_dir="path/to/train/images",
    train_annotation_file="path/to/train/annotations.json",
    val_image_dir="path/to/val/images",
    val_annotation_file="path/to/val/annotations.json",
    ignore_index=255
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