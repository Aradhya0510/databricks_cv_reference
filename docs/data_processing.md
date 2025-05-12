# Data Processing Documentation

## Overview

The data processing pipeline is designed to handle MS COCO format datasets efficiently on Databricks, leveraging distributed computing capabilities and Delta Lake for storage.

## Components

### COCOProcessor

The `COCOProcessor` class handles the processing of MS COCO format datasets.

#### Initialization

```python
from data.processing.coco_processor import COCOProcessor

processor = COCOProcessor(spark)
```

#### Methods

1. **load_coco_annotations**
   ```python
   processor.load_coco_annotations(annotation_file: str) -> None
   ```
   Loads COCO format annotations from a JSON file.

2. **process_images**
   ```python
   processor.process_images(image_dir: str) -> pd.DataFrame
   ```
   Processes images and creates a DataFrame with image metadata.

3. **validate_data**
   ```python
   processor.validate_data(df: pyspark.sql.DataFrame) -> Dict[str, List[str]]
   ```
   Validates the processed data for quality and consistency.

4. **save_to_delta**
   ```python
   processor.save_to_delta(df: pyspark.sql.DataFrame, output_path: str) -> None
   ```
   Saves processed data to Delta Lake format.

### DataLoader

The `DataLoader` class handles efficient data loading for training.

#### Initialization

```python
from data.processing.data_loader import COCODataset, get_transforms

dataset = COCODataset(
    image_paths=paths,
    annotations=anns,
    transform=get_transforms(mode='train')
)
```

#### Methods

1. **get_transforms**
   ```python
   get_transforms(mode: str = 'train') -> A.Compose
   ```
   Returns appropriate data augmentation transforms.

2. **create_dataloader**
   ```python
   create_dataloader(
       dataset: Dataset,
       batch_size: int,
       num_workers: int,
       shuffle: bool = True
   ) -> DataLoader
   ```
   Creates a DataLoader with optimal settings for Databricks.

## Usage Examples

### Basic Processing

```python
# Initialize processor
processor = COCOProcessor(spark)

# Load annotations
processor.load_coco_annotations("/dbfs/path/to/annotations.json")

# Process images
df = processor.process_images("/dbfs/path/to/images")

# Validate data
validation_results = processor.validate_data(df)

# Save to Delta Lake
processor.save_to_delta(df, "/dbfs/path/to/processed_data")
```

### Data Loading for Training

```python
# Create dataset
dataset = COCODataset(
    image_paths=paths,
    annotations=anns,
    transform=get_transforms(mode='train')
)

# Create dataloader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)
```

## Best Practices

1. **Data Validation**
   - Always validate data before processing
   - Check for missing or corrupted images
   - Verify annotation consistency

2. **Performance Optimization**
   - Use appropriate batch sizes
   - Leverage distributed processing
   - Optimize data loading patterns

3. **Storage**
   - Use Delta Lake for versioned storage
   - Implement proper partitioning
   - Monitor storage costs

## Common Issues and Solutions

1. **Memory Issues**
   - Use appropriate batch sizes
   - Implement data streaming
   - Monitor memory usage

2. **Performance Bottlenecks**
   - Optimize data loading
   - Use appropriate number of workers
   - Leverage caching when appropriate

## Configuration

The data processing pipeline can be configured through the `config/data_config.yaml` file:

```yaml
data_processing:
  batch_size: 32
  num_workers: 4
  validation:
    min_image_size: 100
    max_image_size: 2000
    required_fields:
      - image_id
      - file_name
      - width
      - height
  storage:
    format: delta
    partition_by:
      - date
      - category
```

## Monitoring

The pipeline includes monitoring capabilities:

1. **Data Quality Metrics**
   - Missing data percentage
   - Annotation coverage
   - Image quality scores

2. **Performance Metrics**
   - Processing time
   - Memory usage
   - Storage utilization

## API Reference

For detailed API documentation, see the [API Reference](api_reference.md). 