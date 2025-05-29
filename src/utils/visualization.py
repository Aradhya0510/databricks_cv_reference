import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Tuple, Optional
import cv2
from matplotlib.colors import ListedColormap

def create_colormap(num_classes: int, seed: int = 42) -> ListedColormap:
    """Create a colormap for segmentation visualization.
    
    Args:
        num_classes: Number of classes (including background)
        seed: Random seed for reproducibility
    
    Returns:
        ListedColormap: Colormap for visualization
    """
    np.random.seed(seed)
    colors = np.random.rand(num_classes, 3)
    colors[0] = [0, 0, 0]  # Set background to black
    return ListedColormap(colors)

def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    score_threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """Visualize object detection results.
    
    Args:
        image: Input image (H, W, C)
        boxes: Bounding boxes (N, 4) in [x1, y1, x2, y2] format
        scores: Detection scores (N,)
        labels: Class labels (N,)
        class_names: List of class names
        score_threshold: Minimum score to display
        figsize: Figure size
        save_path: Path to save visualization
    """
    # Create figure
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()
    
    # Filter detections by score
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Plot boxes
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = f'{class_names[label]}: {score:.2f}'
        plt.text(
            x1, y1 - 5,
            label_text,
            color='white',
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def visualize_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    class_names: List[str],
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """Visualize segmentation results.
    
    Args:
        image: Input image (H, W, C)
        mask: Segmentation mask (H, W)
        class_names: List of class names
        alpha: Transparency of the overlay
        figsize: Figure size
        save_path: Path to save visualization
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Create colormap
    cmap = create_colormap(len(class_names))
    
    # Plot segmentation mask
    ax2.imshow(image)
    ax2.imshow(mask, alpha=alpha, cmap=cmap)
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cmap(i))
        for i in range(len(class_names))
    ]
    ax2.legend(
        legend_elements,
        class_names,
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def visualize_batch_detection(
    images: torch.Tensor,
    predictions: List[Dict[str, torch.Tensor]],
    class_names: List[str],
    score_threshold: float = 0.5,
    num_images: int = 4,
    figsize: Tuple[int, int] = (15, 15),
    save_path: Optional[str] = None
) -> None:
    """Visualize batch of detection results.
    
    Args:
        images: Batch of images (B, C, H, W)
        predictions: List of prediction dictionaries
        class_names: List of class names
        score_threshold: Minimum score to display
        num_images: Number of images to display
        figsize: Figure size
        save_path: Path to save visualization
    """
    # Convert images to numpy
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))  # (B, H, W, C)
    
    # Normalize images
    images = (images - images.min()) / (images.max() - images.min())
    
    # Create figure
    n_cols = min(num_images, len(images))
    n_rows = (num_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each image
    for i, (image, pred) in enumerate(zip(images, predictions)):
        if i >= num_images:
            break
        
        ax = axes[i]
        ax.imshow(image)
        
        # Get predictions
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter by score
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Plot boxes
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f'{class_names[label]}: {score:.2f}'
            ax.text(
                x1, y1 - 5,
                label_text,
                color='white',
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def visualize_batch_segmentation(
    images: torch.Tensor,
    masks: torch.Tensor,
    class_names: List[str],
    alpha: float = 0.5,
    num_images: int = 4,
    figsize: Tuple[int, int] = (15, 15),
    save_path: Optional[str] = None
) -> None:
    """Visualize batch of segmentation results.
    
    Args:
        images: Batch of images (B, C, H, W)
        masks: Batch of masks (B, H, W)
        class_names: List of class names
        alpha: Transparency of the overlay
        num_images: Number of images to display
        figsize: Figure size
        save_path: Path to save visualization
    """
    # Convert to numpy
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    # Transpose images
    images = np.transpose(images, (0, 2, 3, 1))  # (B, H, W, C)
    
    # Normalize images
    images = (images - images.min()) / (images.max() - images.min())
    
    # Create colormap
    cmap = create_colormap(len(class_names))
    
    # Create figure
    n_cols = min(num_images, len(images))
    n_rows = (num_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each image
    for i, (image, mask) in enumerate(zip(images, masks)):
        if i >= num_images:
            break
        
        ax = axes[i]
        ax.imshow(image)
        ax.imshow(mask, alpha=alpha, cmap=cmap)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cmap(i))
        for i in range(len(class_names))
    ]
    fig.legend(
        legend_elements,
        class_names,
        loc='center right',
        bbox_to_anchor=(0.98, 0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show() 