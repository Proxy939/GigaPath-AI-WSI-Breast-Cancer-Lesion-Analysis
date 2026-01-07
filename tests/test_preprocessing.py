"""
Basic unit tests for preprocessing module.
Run with: python -m pytest tests/test_preprocessing.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.preprocessing.tissue_detector import TissueDetector
from src.preprocessing.slide_utils import get_tile_coordinates, coords_to_thumbnail


def test_tissue_detector_init():
    """Test TissueDetector initialization."""
    detector = TissueDetector(thumbnail_size=2000)
    assert detector.thumbnail_size == 2000
    assert detector.gray_threshold == 200


def test_tile_coordinates_generation():
    """Test tile coordinate generation."""
    slide_dims = (10000, 8000)
    tile_size = 256
    overlap = 0
    
    coords = get_tile_coordinates(slide_dims, tile_size, overlap)
    
    # Check all coords are valid
    for x, y in coords:
        assert x >= 0 and x + tile_size <= slide_dims[0]
        assert y >= 0 and y + tile_size <= slide_dims[1]
    
    # Check expected number of tiles
    expected_tiles_x = slide_dims[0] // tile_size
    expected_tiles_y = slide_dims[1] // tile_size
    assert len(coords) == expected_tiles_x * expected_tiles_y


def test_tile_coordinates_with_overlap():
    """Test tile coordinate generation with overlap."""
    slide_dims = (1000, 1000)
    tile_size = 256
    overlap = 64
    
    coords = get_tile_coordinates(slide_dims, tile_size, overlap)
    
    # With overlap, should have more tiles
    assert len(coords) > 0
    
    # Check stride
    if len(coords) > 1:
        stride = tile_size - overlap
        assert coords[1][0] == coords[0][0] + stride or coords[1][1] == coords[0][1] + stride


def test_coords_to_thumbnail_mapping():
    """Test coordinate mapping from slide to thumbnail space."""
    coords = (5000, 3000)
    tile_size = 256
    thumbnail_shape = (800, 1000)  # (height, width)
    slide_dims = (10000, 8000)  # (width, height)
    
    thumb_x, thumb_y, thumb_w, thumb_h = coords_to_thumbnail(
        coords, tile_size, thumbnail_shape, slide_dims
    )
    
    # Check within bounds
    assert 0 <= thumb_x < thumbnail_shape[1]
    assert 0 <= thumb_y < thumbnail_shape[0]
    assert thumb_w > 0
    assert thumb_h > 0


def test_tissue_mask_creation():
    """Test tissue mask creation from synthetic image."""
    detector = TissueDetector()
    
    # Create synthetic thumbnail (white background with dark tissue region)
    thumbnail = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # White background
    thumbnail[300:700, 300:700, :] = 100  # Dark tissue region
    
    # Create tissue mask
    tissue_mask = detector._create_tissue_mask(thumbnail)
    
    # Check mask shape
    assert tissue_mask.shape == (1000, 1000)
    
    # Check that tissue region is detected
    tissue_ratio = np.sum(tissue_mask) / tissue_mask.size
    assert tissue_ratio > 0.1  # Should detect some tissue


if __name__ == '__main__':
    # Run basic tests
    print("Running preprocessing module tests...")
    
    test_tissue_detector_init()
    print("✓ TissueDetector initialization")
    
    test_tile_coordinates_generation()
    print("✓ Tile coordinate generation")
    
    test_tile_coordinates_with_overlap()
    print("✓ Tile coordinates with overlap")
    
    test_coords_to_thumbnail_mapping()
    print("✓ Coordinate mapping")
    
    test_tissue_mask_creation()
    print("✓ Tissue mask creation")
    
    print("\nAll tests passed! ✓")
