from typing import Dict, List, Optional
import json
from pathlib import Path
import pycocotools.coco as coco
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

class COCOProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.coco_api = None
        
    def load_coco_annotations(self, annotation_file: str) -> None:
        """Load COCO format annotations."""
        self.coco_api = coco.COCO(annotation_file)
        
    def process_images(self, image_dir: str) -> pd.DataFrame:
        """Process images and create a DataFrame with image metadata."""
        if not self.coco_api:
            raise ValueError("COCO annotations not loaded. Call load_coco_annotations first.")
            
        images = []
        for img_id in self.coco_api.getImgIds():
            img_info = self.coco_api.loadImgs(img_id)[0]
            ann_ids = self.coco_api.getAnnIds(imgIds=img_id)
            anns = self.coco_api.loadAnns(ann_ids)
            
            image_data = {
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': anns
            }
            images.append(image_data)
            
        return pd.DataFrame(images)
    
    def create_spark_dataframe(self, df: pd.DataFrame) -> 'pyspark.sql.DataFrame':
        """Convert pandas DataFrame to Spark DataFrame with proper schema."""
        schema = StructType([
            StructField('image_id', StringType(), False),
            StructField('file_name', StringType(), False),
            StructField('width', FloatType(), False),
            StructField('height', FloatType(), False),
            StructField('annotations', ArrayType(
                StructType([
                    StructField('id', StringType(), False),
                    StructField('category_id', StringType(), False),
                    StructField('bbox', ArrayType(FloatType()), False),
                    StructField('segmentation', ArrayType(ArrayType(FloatType())), False),
                    StructField('area', FloatType(), False),
                    StructField('iscrowd', FloatType(), False)
                ])
            ), False)
        ])
        
        return self.spark.createDataFrame(df, schema)
    
    def validate_data(self, df: 'pyspark.sql.DataFrame') -> Dict[str, List[str]]:
        """Validate the processed data for quality and consistency."""
        validation_results = {
            'errors': [],
            'warnings': []
        }
        
        # Check for missing images
        missing_images = df.filter(df.file_name.isNull()).count()
        if missing_images > 0:
            validation_results['errors'].append(f"Found {missing_images} images with missing file names")
            
        # Check for invalid dimensions
        invalid_dims = df.filter((df.width <= 0) | (df.height <= 0)).count()
        if invalid_dims > 0:
            validation_results['errors'].append(f"Found {invalid_dims} images with invalid dimensions")
            
        # Check for empty annotations
        empty_anns = df.filter(df.annotations.isNull()).count()
        if empty_anns > 0:
            validation_results['warnings'].append(f"Found {empty_anns} images without annotations")
            
        return validation_results
    
    def save_to_delta(self, df: 'pyspark.sql.DataFrame', output_path: str) -> None:
        """Save processed data to Delta Lake format."""
        df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(output_path) 