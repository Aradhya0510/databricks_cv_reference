from typing import Dict, Any, Optional, List
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    CreateCatalog,
    CreateSchema,
    TableInfo
)

class CatalogManager:
    """Manages Unity Catalog operations for data and model management."""
    
    def __init__(
        self,
        workspace_url: str,
        token: str,
        catalog_name: str,
        schema_name: str
    ):
        self.workspace = WorkspaceClient(
            host=workspace_url,
            token=token
        )
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        
    def create_catalog_if_not_exists(self) -> None:
        """Create a catalog if it doesn't exist."""
        try:
            self.workspace.catalogs.create(
                CreateCatalog(
                    name=self.catalog_name,
                    comment="Computer Vision Project Catalog"
                )
            )
        except Exception as e:
            print(f"Catalog {self.catalog_name} already exists or error: {e}")
                
    def create_schema_if_not_exists(self) -> None:
        """Create a schema if it doesn't exist."""
        try:
            self.workspace.schemas.create(
                CreateSchema(
                    name=self.schema_name,
                    catalog_name=self.catalog_name,
                    comment="Computer Vision Project Schema"
                )
            )
        except Exception as e:
            print(f"Schema {self.schema_name} already exists or error: {e}")
                
    def create_coco_table(
        self,
        table_name: str,
        location: str,
        comment: Optional[str] = None
    ) -> None:
        """Create a table for COCO format data."""
        try:
            self.workspace.tables.create(
                name=table_name,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_type="EXTERNAL",
                data_source_format="DELTA",
                location=location,
                comment=comment or "COCO format dataset table"
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise e
                
    def register_model_metadata(
        self,
        table_name: str,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any]
    ) -> None:
        """Register model metadata in Unity Catalog."""
        try:
            self.workspace.tables.create(
                name=table_name,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_type="MANAGED",
                data_source_format="DELTA",
                comment=f"Model metadata for {model_name} version {version}"
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise e
                
    def get_table_info(self, table_name: str) -> TableInfo:
        """Get information about a table."""
        try:
            return self.workspace.tables.get(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                name=table_name
            )
        except Exception as e:
            print(f"Error getting table info: {e}")
            return None
        
    def list_tables(self) -> List[str]:
        """List all tables in the schema."""
        try:
            tables = self.workspace.tables.list(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name
            )
            return [table.name for table in tables]
        except Exception as e:
            print(f"Error listing tables: {e}")
            return [] 