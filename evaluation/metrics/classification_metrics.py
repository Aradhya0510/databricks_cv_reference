from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        
    def compute_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        if y_prob is not None:
            y_prob = y_prob.cpu().numpy()
            
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # ROC AUC if probabilities are provided
        if y_prob is not None:
            if self.num_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr'
                )
                
        return metrics
        
    def plot_confusion_matrix(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot confusion matrix."""
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        return fig
        
    def plot_roc_curve(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        y_true = y_true.cpu().numpy()
        y_prob = y_prob.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                
                label = class_names[i] if class_names else f'Class {i}'
                ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
                
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc="lower right")
        
        return fig 