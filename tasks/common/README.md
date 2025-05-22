# Common Components

This directory contains base classes and utilities shared across all vision tasks.

## Base Classes

### BaseVisionDataset

The base dataset class that provides common functionality for COCO format data loading and preprocessing.

#### Key Features

1. **COCO Format Support**
   - Loads and parses COCO format annotations
   - Maintains image ID to annotation mapping
   - Handles category mapping

2. **Image Processing**
   - Standardized image loading
   - RGB conversion
   - Basic preprocessing

3. **Transform Pipeline**
   - Default transform pipeline
   - Custom transform support
   - Albumentations integration

#### Usage

```python
from .base_dataset import BaseVisionDataset

class CustomDataset(BaseVisionDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image using base class functionality
        image = self.load_image(self.image_ids[idx])
        
        # Get annotations using base class functionality
        annotations = self.get_annotations(self.image_ids[idx])
        
        # Implement custom logic
        # ...
        
        return {
            'pixel_values': image,
            'labels': labels
        }
```

### BaseVisionDataModule

The base data module that provides common functionality for data loading and management.

#### Key Features

1. **Data Management**
   - Training and validation data handling
   - Batch size configuration
   - Worker management

2. **Transform Pipeline**
   - Default training transforms
   - Default validation transforms
   - Custom transform support

3. **DataLoader Configuration**
   - Standardized DataLoader setup
   - Shuffle configuration
   - Worker configuration

#### Usage

```python
from .base_datamodule import BaseVisionDataModule

class CustomDataModule(BaseVisionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CustomDataset(
                self.train_image_dir,
                self.train_annotation_file,
                transform=self.train_transform
            )
            self.val_dataset = CustomDataset(
                self.val_image_dir,
                self.val_annotation_file,
                transform=self.val_transform
            )
```

### BaseVisionModule

The base Lightning module that provides common training functionality.

#### Key Features

1. **Model Management**
   - HuggingFace model integration
   - Pretrained model support
   - Model configuration

2. **Training Loop**
   - Standardized training step
   - Validation step
   - Metric logging

3. **Optimization**
   - Learning rate scheduling
   - Weight decay
   - Gradient clipping

#### Usage

```python
from .base_module import BaseVisionModule

class CustomModule(BaseVisionModule):
    def __init__(self, model_ckpt: str, **kwargs):
        super().__init__(model_ckpt, **kwargs)
        # Custom initialization
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Custom training step
        pass
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        # Custom validation step
        pass
```

## Best Practices

1. **Inheritance**
   - Always inherit from base classes
   - Override only necessary methods
   - Use super() for initialization

2. **Type Hints**
   - Use proper type hints
   - Document return types
   - Use Optional for optional parameters

3. **Error Handling**
   - Validate inputs
   - Handle edge cases
   - Provide meaningful error messages

4. **Documentation**
   - Document all methods
   - Include usage examples
   - Explain parameters and returns

## Example: Adding a New Task

```python
# tasks/new_task/datamodule.py
from ..common.base_dataset import BaseVisionDataset
from ..common.base_datamodule import BaseVisionDataModule

class NewTaskDataset(BaseVisionDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.load_image(self.image_ids[idx])
        annotations = self.get_annotations(self.image_ids[idx])
        
        # Task-specific processing
        processed_data = self._process_data(image, annotations)
        
        return processed_data

class NewTaskDataModule(BaseVisionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = NewTaskDataset(
                self.train_image_dir,
                self.train_annotation_file,
                transform=self.train_transform
            )
            self.val_dataset = NewTaskDataset(
                self.val_image_dir,
                self.val_annotation_file,
                transform=self.val_transform
            )
``` 