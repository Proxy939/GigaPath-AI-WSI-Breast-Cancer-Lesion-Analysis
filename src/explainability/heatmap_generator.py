"""
Attention Heatmap Generator for MIL Explainability.
Visualizes model attention on WSI thumbnails.
"""
import numpy as np
import cv2
import h5py
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HeatmapGenerator:
    """Generate attention-based heatmaps for WSI visualization."""
    
    def __init__(
        self,
        colormap: str = 'jet',
        alpha: float = 0.5,
        gaussian_sigma: Optional[float] = None
    ):
        """
        Initialize heatmap generator.
        
        Args:
            colormap: Matplotlib colormap name ('jet', 'hot', 'viridis')
            alpha: Transparency for overlay (0.0-1.0)
            gaussian_sigma: Sigma for Gaussian smoothing (None = auto)
        """
        self.colormap = cm.get_cmap(colormap)
        self.alpha = alpha
        self.gaussian_sigma = gaussian_sigma
    
    def create_heatmap(
        self,
        attention_weights: np.ndarray,
        coordinates: np.ndarray,
        canvas_size: Tuple[int, int],
        tile_size: int = 256
    ) -> np.ndarray:
        """
        Create heatmap from attention weights and tile coordinates.
        
        Args:
            attention_weights: Attention scores (K,)
            coordinates: Tile coordinates in level-0 space (K, 2)
            canvas_size: Output canvas size (H, W)
            tile_size: Size of tiles at level-0
        
        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        # Initialize heatmap
        heatmap = np.zeros(canvas_size, dtype=np.float32)
        
        # Map coordinates to canvas
        # Coordinates are in level-0, need to scale to canvas
        for i, (x, y) in enumerate(coordinates):
            # Place attention weight at tile location
            heatmap[int(y):int(y)+tile_size, int(x):int(x)+tile_size] += attention_weights[i]
        
        # Apply Gaussian smoothing
        if self.gaussian_sigma is None:
            sigma = tile_size / 4  # Auto sigma
        else:
            sigma = self.gaussian_sigma
        
        heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize to [0, 1]
        if heatmap_smooth.max() > 0:
            heatmap_norm = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min())
        else:
            heatmap_norm = heatmap_smooth
        
        return heatmap_norm
    
    def apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Apply colormap to normalized heatmap.
        
        Args:
            heatmap: Normalized heatmap (H, W) in [0, 1]
        
        Returns:
            Colored heatmap (H, W, 3) uint8
        """
        colored = self.colormap(heatmap)  # (H, W, 4) RGBA
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        return colored_rgb
    
    def create_overlay(
        self,
        background: np.ndarray,
        heatmap: np.ndarray
    ) -> np.ndarray:
        """
        Overlay heatmap on background image.
        
        Args:
            background: Background image (H, W, 3) uint8
            heatmap: Normalized heatmap (H, W) in [0, 1]
        
        Returns:
            Overlay image (H, W, 3) uint8
        """
        # Resize heatmap to match background
        if heatmap.shape != background.shape[:2]:
            heatmap_resized = cv2.resize(heatmap, (background.shape[1], background.shape[0]))
        else:
            heatmap_resized = heatmap
        
        # Apply colormap
        heatmap_colored = self.apply_colormap(heatmap_resized)
        
        # Blend
        overlay = cv2.addWeighted(background, 1.0 - self.alpha, heatmap_colored, self.alpha, 0)
        
        return overlay
    
    def highlight_top_k_tiles(
        self,
        image: np.ndarray,
        attention_weights: np.ndarray,
        coordinates: np.ndarray,
        k: int = 10,
        tile_size: int = 256,
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Draw bounding boxes around top-K tiles.
        
        Args:
            image: Background image (H, W, 3)
            attention_weights: Attention scores (K,)
            coordinates: Tile coordinates (K, 2)
            k: Number of top tiles to highlight
            tile_size: Tile size in pixels
            color: Box color (B, G, R)
        
        Returns:
            Image with boxes (H, W, 3)
        """
        image_copy = image.copy()
        
        # Get top-k indices
        top_k_indices = np.argsort(attention_weights)[-k:][::-1]
        
        # Draw boxes
        for idx in top_k_indices:
            x, y = coordinates[idx]
            cv2.rectangle(
                image_copy,
                (int(x), int(y)),
                (int(x) + tile_size, int(y) + tile_size),
                color,
                thickness=3
            )
        
        return image_copy


def load_attention_and_coords(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load attention weights and coordinates from HDF5.
    
    Note: This assumes attention weights were saved during inference.
    If not, they need to be computed on-the-fly.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Tuple of (attention_weights, coordinates)
    """
    with h5py.File(hdf5_path, 'r') as f:
        coords = f['coordinates'][:]
        
        # Check if attention weights are stored
        if 'attention_weights' in f:
            attention = f['attention_weights'][:]
        else:
            # Placeholder: uniform attention
            logger.warning("No attention weights in HDF5, using uniform weights")
            attention = np.ones(len(coords)) / len(coords)
    
    return attention, coords


if __name__ == '__main__':
    print("HeatmapGenerator module - use generate_heatmaps.py for visualization")
