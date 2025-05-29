import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ["model", "training", "data", "distributed", "mlflow", "logging"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model configuration
    if "name" not in config["model"]:
        raise ValueError("Model name not specified in configuration")
    
    # Validate training configuration
    required_training_params = ["batch_size", "epochs", "learning_rate"]
    for param in required_training_params:
        if param not in config["training"]:
            raise ValueError(f"Missing required training parameter: {param}")
    
    # Validate data configuration
    required_data_params = ["train_path", "val_path", "image_size"]
    for param in required_data_params:
        if param not in config["data"]:
            raise ValueError(f"Missing required data parameter: {param}")
    
    # Validate distributed configuration
    required_distributed_params = ["backend", "num_workers", "use_gpu"]
    for param in required_distributed_params:
        if param not in config["distributed"]:
            raise ValueError(f"Missing required distributed parameter: {param}")
    
    # Validate MLflow configuration
    required_mlflow_params = ["experiment_name", "tracking_uri", "registry_uri", "model_name"]
    for param in required_mlflow_params:
        if param not in config["mlflow"]:
            raise ValueError(f"Missing required MLflow parameter: {param}")
    
    return config

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False) 