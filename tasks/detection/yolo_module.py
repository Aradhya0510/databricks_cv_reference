from ultralytics import YOLO
from tasks.detection.lightning_module import DetectionModule
import torch
from typing import Dict, Any, List
import numpy as np

class YoloModule(DetectionModule):
    """YOLOv8 object detection module."""
    
    def __init__(self, model_ckpt: str, config: Any = None):
        """Initialize YOLOv8 module.
        
        Args:
            model_ckpt: Path to model checkpoint or YOLOv8 model size (e.g., 'yolov8n.pt')
            config: Optional configuration overrides
        """
        super().__init__(model_ckpt, config)
        # Load YOLOv8 model
        self.model = YOLO(model_ckpt)
        self.save_hyperparameters()

    def _prepare_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for YOLOv8 model.
        
        Args:
            batch: Dictionary containing image tensors
            
        Returns:
            Dictionary of model inputs
        """
        # YOLOv8 expects images in [B, C, H, W] format
        return {
            'images': batch['pixel_values']
        }

    def _process_model_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process YOLOv8 outputs to COCO format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary containing loss and predictions in COCO format
        """
        # YOLOv8 outputs are in [x1, y1, x2, y2, conf, cls] format
        predictions = []
        
        # Process each image's predictions
        for i, pred in enumerate(outputs):
            if pred is None:
                continue
                
            # Convert predictions to COCO format
            boxes = pred.boxes
            if len(boxes) == 0:
                continue
                
            # Get box coordinates, confidence scores, and class IDs
            xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            
            # Convert to COCO format [x, y, w, h]
            for box, score, category_id in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                predictions.append({
                    'image_id': i,
                    'category_id': int(category_id),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score)
                })
        
        return {
            'loss': outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0),
            'predictions': predictions
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through YOLOv8 model.
        
        Args:
            batch: Dictionary containing image tensors
            
        Returns:
            Dictionary containing model outputs
        """
        # Prepare inputs
        model_inputs = self._prepare_model_inputs(batch)
        
        # Get model outputs
        results = self.model(model_inputs['images'])
        
        # Process outputs
        return self._process_model_outputs(results) 