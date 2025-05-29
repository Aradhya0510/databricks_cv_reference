from tasks.detection.detection_processor import DetectionProcessor
import torch
from typing import Dict, Any, List
from transformers import DetrImageProcessor

class DetrProcessor(DetectionProcessor):
    """Processor for DETR model inputs and outputs."""
    
    def __init__(self, config: Any):
        """Initialize DETR processor.
        
        Args:
            config: Configuration object containing model parameters
        """
        # Initialize the parent class (DetectionProcessor) which inherits from ModelProcessor
        DetectionProcessor.__init__(self, config)
        self.processor = DetrImageProcessor.from_pretrained(
            config.model_name,
            do_resize=False,  # Disable resizing since images are already resized
            do_rescale=False,  # Disable rescaling since images are already normalized
            do_normalize=False  # Disable normalization since images are already normalized
        )
        # Store precision from config
        self.precision = getattr(config, 'precision', '32')
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs in the format expected by DETR.
        
        Args:
            batch: Dictionary containing COCO format tensors
            
        Returns:
            Dictionary of model inputs
        """
        # Convert annotations to DETR format
        detections = []
        for i, (boxes, labels) in enumerate(zip(batch['boxes'], batch['labels'])):
            if len(boxes) > 0:
                # Move tensors to CPU before converting to list
                boxes_cpu = boxes.cpu()
                labels_cpu = labels.cpu()
                detections.append({
                    'image_id': i,
                    'annotations': [
                        {
                            'bbox': box.tolist(),
                            'category_id': label.item(),
                            'area': float((box[2] - box[0]) * (box[3] - box[1])),
                            'iscrowd': 0
                        }
                        for box, label in zip(boxes_cpu, labels_cpu)
                    ]
                })
            else:
                detections.append({
                    'image_id': i,
                    'annotations': []
                })
        
        # Move images to CPU if they're on GPU
        pixel_values = batch['pixel_values'].cpu() if batch['pixel_values'].is_cuda else batch['pixel_values']
        
        # Process images using DETR processor
        processed = self.processor(
            images=pixel_values,
            annotations=detections,
            return_tensors="pt"
        )
        
        # Convert to appropriate precision based on config
        if self.precision == '16' or self.precision == '16-mixed':
            processed.pixel_values = processed.pixel_values.to(dtype=torch.float16)
            processed.labels = processed.labels.to(dtype=torch.float16)
        
        return {
            'pixel_values': processed.pixel_values,
            'labels': processed.labels
        }
    
    def process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process DETR model outputs to COCO format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary of COCO format outputs
        """
        # Process raw outputs
        processed = self._postprocess_outputs(outputs)
        
        # Convert to COCO format
        predictions = self._convert_to_coco_format(processed)
        
        return {
            'loss': processed['loss'],
            'predictions': predictions
        }
    
    def _postprocess_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process raw DETR outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs
        """
        predictions = []
        for pred in outputs.predictions:
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            
            # Apply confidence threshold
            mask = scores > self.config.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h]
            boxes = self._convert_boxes_format(boxes)
            
            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        
        return {
            'loss': outputs.loss,
            'predictions': predictions
        }
    
    def _convert_boxes_format(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format.
        
        Args:
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            
        Returns:
            Bounding boxes in [x, y, w, h] format
        """
        x1, y1, x2, y2 = boxes.unbind(-1)
        w = x2 - x1
        h = y2 - y1
        return torch.stack([x1, y1, w, h], dim=-1)
    
    def _convert_to_coco_format(self, processed: Dict[str, torch.Tensor]) -> List[Dict]:
        """Convert DETR outputs to COCO format.
        
        Args:
            processed: Processed model outputs
            
        Returns:
            List of predictions in COCO format
        """
        predictions = []
        for pred in processed['predictions']:
            # Move tensors to CPU before converting to numpy
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            
            for box, score, label in zip(boxes, scores, labels):
                predictions.append({
                    'image_id': 0,  # This should be set appropriately
                    'category_id': int(label.item()),
                    'bbox': box.numpy().tolist(),
                    'score': float(score.item())
                })
        
        return predictions 