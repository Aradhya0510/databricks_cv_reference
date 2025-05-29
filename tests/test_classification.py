import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.tasks.classification.model import ClassificationModel
from src.tasks.classification.data import ClassificationDataset, ClassificationDataModule

@pytest.fixture
def sample_config():
    return {
        "model": {
            "name": "resnet18",
            "pretrained": False,
            "num_classes": 10,
            "dropout": 0.2
        },
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "warmup_epochs": 1
        },
        "data": {
            "train_path": "dummy_path",
            "val_path": "dummy_path",
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation": 15,
            "brightness_contrast": 0.2,
            "random_crop": True,
            "random_erasing": True
        },
        "distributed": {
            "backend": "ray",
            "num_workers": 1,
            "use_gpu": False,
            "strategy": "ddp"
        },
        "mlflow": {
            "experiment_name": "test",
            "tracking_uri": "test",
            "registry_uri": "test",
            "model_name": "test",
            "tags": {
                "task": "classification",
                "model": "resnet18",
                "dataset": "test"
            }
        },
        "logging": {
            "log_every_n_steps": 1,
            "save_top_k": 1,
            "monitor": "val_loss",
            "mode": "min"
        }
    }

@pytest.fixture
def temp_data_dir():
    # Create a temporary directory with dummy data
    temp_dir = tempfile.mkdtemp()
    
    # Create class directories and dummy images
    for class_idx in range(3):
        class_dir = Path(temp_dir) / f"class_{class_idx}"
        class_dir.mkdir()
        
        # Create dummy images
        for i in range(2):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = class_dir / f"image_{i}.jpg"
            from PIL import Image
            Image.fromarray(img).save(img_path)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_model_initialization(sample_config):
    model = ClassificationModel(sample_config)
    assert isinstance(model, ClassificationModel)
    assert model.model.num_classes == sample_config["model"]["num_classes"]

def test_model_forward(sample_config):
    model = ClassificationModel(sample_config)
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)
    assert output.shape == (batch_size, sample_config["model"]["num_classes"])

def test_dataset_initialization(temp_data_dir, sample_config):
    dataset = ClassificationDataset(temp_data_dir, is_train=True)
    assert len(dataset) == 6  # 3 classes * 2 images per class

def test_dataset_getitem(temp_data_dir, sample_config):
    dataset = ClassificationDataset(temp_data_dir, is_train=True)
    image, label = dataset[0]
    assert isinstance(image, np.ndarray)
    assert isinstance(label, int)
    assert 0 <= label < 3

def test_datamodule_initialization(sample_config):
    data_module = ClassificationDataModule(sample_config)
    assert isinstance(data_module, ClassificationDataModule)

def test_datamodule_setup(temp_data_dir, sample_config):
    sample_config["data"]["train_path"] = temp_data_dir
    sample_config["data"]["val_path"] = temp_data_dir
    
    data_module = ClassificationDataModule(sample_config)
    data_module.setup()
    
    assert hasattr(data_module, "train_dataset")
    assert hasattr(data_module, "val_dataset")
    assert hasattr(data_module, "test_dataset")

def test_datamodule_dataloaders(temp_data_dir, sample_config):
    sample_config["data"]["train_path"] = temp_data_dir
    sample_config["data"]["val_path"] = temp_data_dir
    
    data_module = ClassificationDataModule(sample_config)
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    
    # Test batch
    batch = next(iter(train_loader))
    assert len(batch) == 2
    assert batch[0].shape == (sample_config["training"]["batch_size"], 3, 224, 224)
    assert batch[1].shape == (sample_config["training"]["batch_size"],) 