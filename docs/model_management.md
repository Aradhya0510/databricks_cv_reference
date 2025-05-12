# Model Management Documentation

## Overview

The model management system provides a flexible and type-safe way to work with HuggingFace computer vision models, supporting various tasks and configurations.

## Components

### Model Schemas

The model management system uses Pydantic schemas for type-safe configuration:

```python
from schemas.model import ModelConfig, ModelType

config = ModelConfig(
    model_name="google/vit-base-patch16-224",
    model_type=ModelType.CLASSIFICATION,
    pretrained=True,
    num_classes=10
)
```

### HuggingFace Model Loader

The `HuggingFaceModelLoader` class provides a unified interface for loading HuggingFace models:

```python
from models.huggingface.model_loader import HuggingFaceModelLoader

# Load a model
model = HuggingFaceModelLoader.load_model(config)

# Get available models
available_models = HuggingFaceModelLoader.get_available_models(ModelType.CLASSIFICATION)
```

## Supported Model Types

1. **Classification**
   - Vision Transformer (ViT)
   - ResNet
   - ConvNeXt
   - Custom classification models

2. **Object Detection**
   - DETR
   - Table Transformer
   - Custom detection models

3. **Segmentation**
   - SegFormer
   - Mask2Former
   - Custom segmentation models

## Usage Examples

### Loading a Pretrained Model

```python
from models.huggingface.model_loader import HuggingFaceModelLoader
from schemas.model import ModelConfig, ModelType

# Configure model
config = ModelConfig(
    model_name="google/vit-base-patch16-224",
    model_type=ModelType.CLASSIFICATION,
    pretrained=True,
    num_classes=10
)

# Load model
model = HuggingFaceModelLoader.load_model(config)
```

### Fine-tuning a Model

```python
# Configure model for fine-tuning
config = ModelConfig(
    model_name="google/vit-base-patch16-224",
    model_type=ModelType.CLASSIFICATION,
    pretrained=True,
    num_classes=10
)

# Load model
model = HuggingFaceModelLoader.load_model(config)

# Configure trainer
trainer = BaseTrainer(
    model=model,
    model_config=config,
    learning_rate=1e-4
)

# Start training
ray_trainer = RayTrainer(trainer, num_workers=4, use_gpu=True)
ray_trainer.train(training_config)
```

## Best Practices

1. **Model Selection**
   - Choose appropriate model architecture
   - Consider computational requirements
   - Evaluate model performance

2. **Configuration**
   - Use type-safe configuration
   - Validate model parameters
   - Document model choices

3. **Fine-tuning**
   - Start with pretrained models
   - Use appropriate learning rates
   - Monitor training progress

## Common Issues and Solutions

1. **Memory Issues**
   - Use model quantization
   - Implement gradient checkpointing
   - Optimize batch sizes

2. **Performance Issues**
   - Use appropriate model size
   - Implement proper caching
   - Optimize data loading

## Configuration

Model configuration can be specified in `config/model_config.yaml`:

```yaml
model:
  name: "google/vit-base-patch16-224"
  type: "classification"
  pretrained: true
  num_classes: 10
  image_size: 224
  channels: 3
```

## API Reference

For detailed API documentation, see the [API Reference](api_reference.md). 