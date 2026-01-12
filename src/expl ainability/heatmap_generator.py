"""
Attention Heatmap Generator for MIL Explainability.

Fixed Implementation:
- Percentile-based attention normalization (1%-99%)
- Tur colormap for high-contrast medical visualization
- Tile-level rendering with proper spatial layout
- Image-aware overlays (WSI thumbnail or canvas fallback)
- Generates 3 outputs: heatmap, overlay, top-K tiles
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from pathlib import Path
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HeatmapGenerator:
    """Generate high-contrast attention-based heatmaps for WSI visualization."""
    
    def __init__(
        self,
        colormap: str = 'turbo',
        alpha: float = 0.6,
        tile_size: int = 256
    ):
        """
        Initialize heatmap generator.
        
        Args:
            colormap: Matplotlib colormap ('turbo', 'jet', 'viridis')
            alpha: Transparency for overlay (0.0-1.0)
            tile_size: Default tile size in pixels
        """
        self.colormap_name = colormap
        self.alpha = alpha
        self.tile_size = tile_size
        self.logger = logger
    
    def normalize_attention(self, attention_scores: np.ndarray) -> np.ndarray:
        """
        Normalize attention scores with percentile clipping for contrast.
        
        Args:
            attention_scores: Raw attention scores
        
        Returns:
            Normalized scores in [0, 1] with enhanced contrast
        
        Raises:
            RuntimeError: If attention scores are empty
        """
        if len(attention_scores) == 0:
            raise RuntimeError("Cannot normalize: attention scores are empty")
        
        scores = attention_scores.copy()
        
        if np.all(scores == scores[0]):
            # All scores identical - return uniform mid-value
            self.logger.warning("All attention scores are identical - using uniform 0.5")
            return np.full_like(scores, 0.5)
        
        # Percentile clipping to remove outliers (1%-99%)
        p1, p99 = np.percentile(scores, [1, 99])
        scores_clipped = np.clip(scores, p1, p99)
        
        # Min-max normalization to [0, 1]
        min_val = scores_clipped.min()
        max_val = scores_clipped.max()
        
        if max_val - min_val < 1e-10:
            # Nearly identical after clipping - use mid-value
            return np.full_like(scores, 0.5)
        
        normalized = (scores_clipped - min_val) / (max_val - min_val)
        
        self.logger.info(f"Attention normalization: min={min_val:.4f}, max={max_val:.4f}, "
                        f"range=[{normalized.min():.4f}, {normalized.max():.4f}]")
        
        return normalized
    
    def generate_attention_heatmap(
        self,
        attention_scores: np.ndarray,
        coords: np.ndarray,
        output_path: str,
        wsi_path: Optional[str] = None,
        tile_size: Optional[int] = None,
        mode: str = 'attention'
    ):
        """
        Generate tile-level attention heatmap with proper normalization and colormap.
        
        Args:
            attention_scores: Attention weights for each tile
            coords: Tile coordinates (N, 2) - level-0 pixel space
            output_path: Output directory
            wsi_path: Optional WSI path for thumbnail overlay
            tile_size: Tile size in pixels (None = use default)
            mode: Heatmap mode
        
        Raises:
            RuntimeError: If attention scores or coordinates are missing/mismatched
        """
        if len(attention_scores) == 0:
            raise RuntimeError("Cannot generate heatmap: attention scores are empty")
        
        if len(coords) == 0:
            raise RuntimeError("Cannot generate heatmap: coordinates are missing")
        
        if len(attention_scores) != len(coords):
            raise RuntimeError(
                f"Mismatch: {len(attention_scores)} attention scores vs {len(coords)} coordinates"
            )
        
        if tile_size is None:
            tile_size = self.tile_size
        
        self.logger.info(f"Generating attention heatmap for {len(attention_scores)} tiles")
        
        # Normalize attention scores with percentile clipping
        normalized_scores = self.normalize_attention(attention_scores)
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use perceptual colormap (turbo for medical visualization)
        colormap = cm.get_cmap(self.colormap_name)
        
        # Generate 3 outputs
        self._generate_heatmap_canvas(normalized_scores, coords, tile_size, colormap, output_dir)
        self._generate_overlay(normalized_scores, coords, tile_size, colormap, wsi_path, output_dir)
        self._generate_topk_tiles(normalized_scores, coords, tile_size, colormap, output_dir)
        
        self.logger.info(f"Heatmaps saved to {output_dir}")
    
    def _generate_heatmap_canvas(
        self,
        normalized_scores: np.ndarray,
        coords: np.ndarray,
        tile_size: int,
        colormap,
        output_dir: Path
    ):
        """Generate standalone attention heatmap on canvas."""
        # Determine canvas size from coordinates
        max_x = coords[:, 0].max() + tile_size
        max_y = coords[:, 1].max() + tile_size
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)  # Invert Y for image coordinates
        ax.set_aspect('equal')
        
        # Render each tile as colored rectangle
        for score, (x, y) in zip(normalized_scores, coords):
            color = colormap(score)
            rect = mpatches.Rectangle(
                (x, y), tile_size, tile_size,
                facecolor=color,
                edgecolor='none',
                alpha=0.9
            )
            ax.add_patch(rect)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Score', rotation=270, labelpad=15)
        
        ax.set_title('MIL Attention Heatmap (Tile-Level)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        output_path = output_dir / 'attention_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Heatmap canvas saved: {output_path}")
    
    def _generate_overlay(
        self,
        normalized_scores: np.ndarray,
        coords: np.ndarray,
        tile_size: int,
        colormap,
        wsi_path: Optional[str],
        output_dir: Path
    ):
        """Generate attention overlay on WSI thumbnail or reconstructed image."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Try to load WSI thumbnail
        background_loaded = False
        if wsi_path and Path(wsi_path).exists():
            try:
                import openslide
                slide = openslide.OpenSlide(str(wsi_path))
                
                # Get thumbnail
                thumb_size = (2048, 2048)
                thumbnail = slide.get_thumbnail(thumb_size)
                
                # Calculate scaling factor
                level_0_dims = slide.level_dimensions[0]
                scale_x = thumbnail.width / level_0_dims[0]
                scale_y = thumbnail.height / level_0_dims[1]
                
                # Display thumbnail
                ax.imshow(thumbnail)
                ax.set_xlim(0, thumbnail.width)
                ax.set_ylim(thumbnail.height, 0)
                
                # Scale coordinates
                scaled_coords = coords * [scale_x, scale_y]
                scaled_tile_size = int(tile_size * min(scale_x, scale_y))
                
                background_loaded = True
                slide.close()
            except Exception as e:
                self.logger.warning(f"Could not load WSI thumbnail: {e}")
        
        if not background_loaded:
            # Fallback: black canvas
            max_x = coords[:, 0].max() + tile_size
            max_y = coords[:, 1].max() + tile_size
            ax.set_xlim(0, max_x)
            ax.set_ylim(max_y, 0)
            ax.set_facecolor('black')
            scaled_coords = coords
            scaled_tile_size = tile_size
        
        # Overlay attention tiles with transparency
        for score, (x, y) in zip(normalized_scores, scaled_coords):
            color = colormap(score)
            rect = mpatches.Rectangle(
                (x, y), scaled_tile_size, scaled_tile_size,
                facecolor=color,
                edgecolor='none',
                alpha=self.alpha
            )
            ax.add_patch(rect)
        
        ax.set_aspect('equal')
        ax.set_title('Attention Overlay on WSI', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        output_path = output_dir / 'attention_overlay.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Overlay saved: {output_path}")
    
    def _generate_topk_tiles(
        self,
        normalized_scores: np.ndarray,
        coords: np.ndarray,
        tile_size: int,
        colormap,
        output_dir: Path,
        k: int = 10
    ):
        """Generate visualization of top-K most attended tiles."""
        # Get top-K indices
        k = min(k, len(normalized_scores))
        top_indices = np.argsort(normalized_scores)[-k:][::-1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Determine canvas size
        max_x = coords[:, 0].max() + tile_size
        max_y = coords[:, 1].max() + tile_size
        
        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        
        # Render all tiles in gray
        for x, y in coords:
            rect = mpatches.Rectangle(
                (x, y), tile_size, tile_size,
                facecolor='gray',
                edgecolor='none',
                alpha=0.3
            )
            ax.add_patch(rect)
        
        # Highlight top-K tiles in color
        for rank, idx in enumerate(top_indices):
            score = normalized_scores[idx]
            x, y = coords[idx]
            color = colormap(score)
            
            rect = mpatches.Rectangle(
                (x, y), tile_size, tile_size,
                facecolor=color,
                edgecolor='yellow',
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(rect)
            
            # Add rank label
            ax.text(
                x + tile_size / 2, y + tile_size / 2,
                str(rank + 1),
                color='white',
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7)
            )
        
        ax.set_title(f'Top-{k} Most Attended Tiles', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        output_path = output_dir / 'attention_topk.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Top-K visualization saved: {output_path}")


def load_attention_and_coords(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load attention weights and coordinates from HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Tuple of (attention_weights, coordinates)
    
    Raises:
        RuntimeError: If coordinates are missing from HDF5
    """
    with h5py.File(hdf5_path, 'r') as f:
        if 'coordinates' not in f and 'coords' not in f:
            raise RuntimeError(f"No coordinates found in {hdf5_path}")
        
        coords = f['coordinates'][:] if 'coordinates' in f else f['coords'][:]
        
        # Check if attention weights are stored
        if 'attention_weights' in f:
            attention = f['attention_weights'][:]
        elif 'attention' in f:
            attention = f['attention'][:]
        else:
            # Placeholder: uniform attention
            logger.warning("No attention weights in HDF5, using uniform weights")
            attention = np.ones(len(coords)) / len(coords)
    
    return attention, coords
