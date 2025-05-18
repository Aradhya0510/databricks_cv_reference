from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import pycocotools.coco as coco
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, IntegerType, LongType, BinaryType
from ..unity_catalog.catalog_manager import CatalogManager
from PIL import Image
import io

class COCOProcessor:
    def __init__(
        self,
        spark: SparkSession,
        catalog_manager: Optional[CatalogManager] = None
    ):
        self.spark = spark
        self.coco_api = None
        self.catalog_manager = catalog_manager
        
    def load_coco_annotations(self, annotation_file: str) -> None:
        """Load COCO format annotations."""
        self.coco_api = coco.COCO(annotation_file)
        
    def process_images(self, image_dir: str) -> 'pyspark.sql.DataFrame':
        """Process images and create a Spark DataFrame with image metadata.
        
        Args:
            image_dir: Directory containing the images
            
        Returns:
            pyspark.sql.DataFrame: DataFrame containing image metadata and annotations
        """
        if not self.coco_api:
            raise ValueError("COCO annotations not loaded. Call load_coco_annotations first.")
            
        images = []
        for img_id in self.coco_api.getImgIds():
            img_info = self.coco_api.loadImgs(img_id)[0]
            ann_ids = self.coco_api.getAnnIds(imgIds=img_id)
            anns = self.coco_api.loadAnns(ann_ids)
            
            # Convert bbox values to float
            for ann in anns:
                ann['bbox'] = [float(x) for x in ann['bbox']]
                # Convert segmentation to string representation
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):
                        # Convert polygon format to string
                        ann['segmentation'] = str(ann['segmentation'])
                    else:
                        # Keep RLE format as is
                        ann['segmentation'] = str(ann['segmentation'])
                # Convert area to float
                ann['area'] = float(ann['area'])
            
            # Create full image path
            full_image_path = str(Path(image_dir) / img_info['file_name'])
            
            image_data = {
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'image_path': full_image_path,  # Add full path
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': anns
            }
            images.append(image_data)
            
        # Convert to pandas DataFrame first (for easier list handling)
        pdf = pd.DataFrame(images)
        # Convert to Spark DataFrame with proper schema
        return self.create_spark_dataframe(pdf)
    
    def create_spark_dataframe(self, df: pd.DataFrame) -> 'pyspark.sql.DataFrame':
        """Convert pandas DataFrame to Spark DataFrame with proper schema."""
        schema = StructType([
            StructField('image_id', LongType(), False),
            StructField('file_name', StringType(), False),
            StructField('image_path', StringType(), False),  # Add image_path field
            StructField('width', IntegerType(), False),
            StructField('height', IntegerType(), False),
            StructField('annotations', ArrayType(
                StructType([
                    StructField('id', LongType(), False),
                    StructField('category_id', IntegerType(), False),
                    StructField('bbox', ArrayType(FloatType()), False),
                    StructField('segmentation', StringType(), False),
                    StructField('area', FloatType(), False),
                    StructField('iscrowd', IntegerType(), False)
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
    
    def save_to_delta(
        self,
        df: 'pyspark.sql.DataFrame',
        catalog_name: str,
        schema_name: str,
        table_name: str,
        comment: Optional[str] = None
    ) -> None:
        """Save processed data to Delta Lake format and register in Unity Catalog.
        
        Args:
            df: Spark DataFrame to save
            catalog_name: Name of the catalog in Unity Catalog
            schema_name: Name of the schema in Unity Catalog
            table_name: Name of the table in Unity Catalog
            comment: Optional comment for the table
        """
        if not self.catalog_manager:
            raise ValueError("CatalogManager is required for Unity Catalog integration")
            
        # Create full table path
        table_path = f"{catalog_name}.{schema_name}.{table_name}"
        
        # Save to Delta Lake with Unity Catalog
        df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(table_path)
            
        # Add table properties and comment
        if comment:
            self.spark.sql(f"COMMENT ON TABLE {table_path} IS '{comment}'")
            
        # Add table properties for better performance
        self.spark.sql(f"""
            ALTER TABLE {table_path} 
            SET TBLPROPERTIES (
                'delta.autoOptimize.optimizeWrite' = 'true',
                'delta.autoOptimize.autoCompact' = 'true'
            )
        """)
            
    def save_batch_inference_results(
        self,
        df: 'pyspark.sql.DataFrame',
        output_path: str,
        table_name: str,
        model_name: str,
        version: str
    ) -> None:
        """Save batch inference results to Delta Lake and register in Unity Catalog."""
        if not self.catalog_manager:
            raise ValueError("CatalogManager is required for saving batch inference results")
            
        # Save to Delta Lake
        df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(output_path)
            
        # Register in Unity Catalog
        self.catalog_manager.create_coco_table(
            table_name=table_name,
            location=output_path,
            comment=f"Batch inference results for {model_name} version {version}"
        )

    def process_images_with_binary(
        self,
        image_dir: str,
        max_image_size: Optional[Tuple[int, int]] = None
    ) -> 'pyspark.sql.DataFrame':
        """Process images and create a Spark DataFrame with image metadata and binary image data.
        
        Args:
            image_dir: Directory containing the images
            max_image_size: Optional tuple of (width, height) to resize images before storage
            
        Returns:
            pyspark.sql.DataFrame: DataFrame containing image metadata, annotations, and binary image data
        """
        if not self.coco_api:
            raise ValueError("COCO annotations not loaded. Call load_coco_annotations first.")
            
        images = []
        for img_id in self.coco_api.getImgIds():
            img_info = self.coco_api.loadImgs(img_id)[0]
            ann_ids = self.coco_api.getAnnIds(imgIds=img_id)
            anns = self.coco_api.loadAnns(ann_ids)
            
            # Load and process image
            image_path = Path(image_dir) / img_info['file_name']
            with Image.open(image_path) as img:
                if max_image_size:
                    img = img.resize(max_image_size, Image.Resampling.LANCZOS)
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format or 'JPEG')
                img_bytes = img_byte_arr.getvalue()
            
            # Process annotations
            for ann in anns:
                ann['bbox'] = [float(x) for x in ann['bbox']]
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):
                        ann['segmentation'] = str(ann['segmentation'])
                    else:
                        ann['segmentation'] = str(ann['segmentation'])
                ann['area'] = float(ann['area'])
            
            image_data = {
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': anns,
                'image_data': img_bytes
            }
            images.append(image_data)
            
        # Convert to pandas DataFrame
        pdf = pd.DataFrame(images)
        # Convert to Spark DataFrame with proper schema
        return self.create_spark_dataframe_with_binary(pdf)
        
    def create_spark_dataframe_with_binary(self, df: pd.DataFrame) -> 'pyspark.sql.DataFrame':
        """Convert pandas DataFrame to Spark DataFrame with binary image data."""
        schema = StructType([
            StructField('image_id', LongType(), False),
            StructField('file_name', StringType(), False),
            StructField('width', IntegerType(), False),
            StructField('height', IntegerType(), False),
            StructField('annotations', ArrayType(
                StructType([
                    StructField('id', LongType(), False),
                    StructField('category_id', IntegerType(), False),
                    StructField('bbox', ArrayType(FloatType()), False),
                    StructField('segmentation', StringType(), False),
                    StructField('area', FloatType(), False),
                    StructField('iscrowd', IntegerType(), False)
                ])
            ), False),
            StructField('image_data', BinaryType(), False)
        ])
        
        return self.spark.createDataFrame(df, schema) 