from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import seaborn as sns

class DetectionMetrics:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        
    def compute_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        num_classes: int
    ) -> Dict[str, float]:
        """Compute detection metrics using COCO evaluation."""
        coco_gt = self._convert_to_coco_format(ground_truth)
        coco_dt = self._convert_to_coco_format(predictions)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            'mAP': coco_eval.stats[0],
            'mAP_50': coco_eval.stats[1],
            'mAP_75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5],
            'AR_max_1': coco_eval.stats[6],
            'AR_max_10': coco_eval.stats[7],
            'AR_max_100': coco_eval.stats[8],
            'AR_small': coco_eval.stats[9],
            'AR_medium': coco_eval.stats[10],
            'AR_large': coco_eval.stats[11]
        }
        
        return metrics
        
    def _convert_to_coco_format(self, annotations: List[Dict]) -> COCO:
        """Convert annotations to COCO format."""
        coco_anns = []
        for ann in annotations:
            coco_ann = {
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'score': ann.get('score', 1.0),
                'area': ann['bbox'][2] * ann['bbox'][3]
            }
            coco_anns.append(coco_ann)
            
        return coco_anns
        
    def plot_precision_recall_curve(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        class_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot precision-recall curves for each class."""
        coco_gt = self._convert_to_coco_format(ground_truth)
        coco_dt = self._convert_to_coco_format(predictions)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, class_id in enumerate(coco_eval.cocoGt.getCatIds()):
            precision = coco_eval.eval['precision'][0, :, i, 0, 2]
            recall = np.linspace(0, 1, len(precision))
            
            label = class_names[i] if class_names else f'Class {class_id}'
            ax.plot(recall, precision, label=label)
            
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='lower left')
        ax.grid(True)
        
        return fig
        
    def plot_detection_examples(
        self,
        images: List[np.ndarray],
        predictions: List[Dict],
        ground_truth: List[Dict],
        class_names: Optional[List[str]] = None,
        max_images: int = 4
    ) -> plt.Figure:
        """Plot example detections with ground truth boxes."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx, (img, pred, gt) in enumerate(zip(images, predictions, ground_truth)):
            if idx >= max_images:
                break
                
            ax = axes[idx]
            ax.imshow(img)
            
            # Plot ground truth boxes
            for box in gt['bboxes']:
                x, y, w, h = box
                rect = plt.Rectangle(
                    (x, y), w, h,
                    fill=False,
                    edgecolor='green',
                    linewidth=2
                )
                ax.add_patch(rect)
                
            # Plot prediction boxes
            for box, score, cls in zip(pred['bboxes'], pred['scores'], pred['labels']):
                x, y, w, h = box
                rect = plt.Rectangle(
                    (x, y), w, h,
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax.add_patch(rect)
                
                label = class_names[cls] if class_names else f'Class {cls}'
                ax.text(
                    x, y - 5,
                    f'{label}: {score:.2f}',
                    color='red',
                    backgroundcolor='white'
                )
                
            ax.axis('off')
            
        plt.tight_layout()
        return fig 