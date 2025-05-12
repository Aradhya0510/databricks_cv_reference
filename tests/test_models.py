import pytest
import torch
from models.base.base_model import BaseModel, ModelConfig
from models.architectures.classification import ClassificationModel

class TestBaseModel:
    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            learning_rate=1e-4,
            weight_decay=1e-5,
            task='classification'
        )
    
    @pytest.fixture
    def model(self, model_config):
        return ClassificationModel(model_config)
    
    def test_forward_pass(self, model):
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 10)  # Assuming 10 classes
        
    def test_training_step(self, model):
        x = torch.randn(1, 3, 224, 224)
        y = torch.randint(0, 10, (1,))
        loss = model.training_step((x, y), 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        
    def test_validation_step(self, model):
        x = torch.randn(1, 3, 224, 224)
        y = torch.randint(0, 10, (1,))
        model.validation_step((x, y), 0)
        # Check if metrics were logged
        assert 'val_loss' in model.logged_metrics 