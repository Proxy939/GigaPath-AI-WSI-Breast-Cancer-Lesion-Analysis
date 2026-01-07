"""
Test Top-K sampling modules.
Run from project root: python -m tests.test_sampling_simple
"""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
import tempfile
import os

from src.sampling import TileRanker, TopKSelector


def test_tile_ranker():
    """Test TileRanker functionality."""
    print("="*60)
    print("Testing TileRanker")
    print("="*60)
    
    # Create dummy features
    features = np.random.randn(2000, 2048).astype(np.float32)
    
    # Test feature norm ranking
    scores = TileRanker.rank_by_feature_norm(features)
    print(f"✓ Feature norm scores: shape={scores.shape}")
    assert scores.shape == (2000,), "Score shape mismatch"
    
    # Test top-K selection
    k = 1000
    top_k_indices = TileRanker.select_top_k(scores, k)
    print(f"✓ Top-K selection: selected {len(top_k_indices)} / {len(scores)}")
    assert len(top_k_indices) == k, "Top-K count mismatch"
    
    # Verify descending order
    for i in range(len(top_k_indices)-1):
        assert scores[top_k_indices[i]] >= scores[top_k_indices[i+1]], \
            "Scores not in descending order!"
    print(f"✓ Scores in descending order")
    
    # Test score stats
    stats = TileRanker.get_score_stats(scores[top_k_indices])
    print(f"✓ Score stats: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
    assert all(k in stats for k in ['min', 'max', 'mean', 'std']), "Missing stats"
    
    print("\n✅ TileRanker tests passed!\n")


def test_top_k_selector():
    """Test TopKSelector functionality."""
    print("="*60)
    print("Testing TopKSelector")
    print("="*60)
    
    # Create temp files
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_input:
        input_path = tmp_input.name
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_output:
        output_path = tmp_output.name
    
    try:
        # Create test HDF5
        features = np.random.randn(2000, 2048).astype(np.float32)
        coordinates = np.random.randint(0, 10000, (2000, 2)).astype(np.int32)
        
        print(f"Creating test HDF5 with {len(features)} tiles...")
        with h5py.File(input_path, 'w') as f:
            features_ds = f.create_dataset('features', data=features)
            coords_ds = f.create_dataset('coordinates', data=coordinates)
            
            features_ds.attrs['slide_name'] = 'test_slide'
            features_ds.attrs['model_name'] = 'resnet50-imagenet'
            features_ds.attrs['feature_dim'] = 2048
            coords_ds.attrs['space'] = 'level_0'
        
        print(f"✓ Created test HDF5")
        
        # Run Top-K selection
        selector = TopKSelector(k=1000, ranking_method='feature_norm')
        stats = selector.select_top_k_from_hdf5(input_path, output_path)
        
        print(f"✓ Top-K selection complete")
        print(f"  Original: {stats['original_num_tiles']} tiles")
        print(f"  Selected: {stats['selected_k']} tiles")
        print(f"  Reduction: {stats['reduction_ratio']:.1%}")
        
        # Verify output
        print("Verifying output HDF5...")
        with h5py.File(output_path, 'r') as f:
            assert f['features'].shape == (1000, 2048), f"Features shape mismatch: {f['features'].shape}"
            assert f['coordinates'].shape == (1000, 2), f"Coords shape mismatch: {f['coordinates'].shape}"
            assert f['coordinates'].attrs['space'] == 'level_0', "Coordinate space not level_0"
            assert 'ranking_method' in f['features'].attrs, "Missing ranking_method"
            assert 'score_min' in f['features'].attrs, "Missing score_min"
            assert 'score_max' in f['features'].attrs, "Missing score_max"
            assert 'score_mean' in f['features'].attrs, "Missing score_mean"
            assert 'score_std' in f['features'].attrs, "Missing score_std"
            
            print(f"✓ Output HDF5 structure verified")
            print(f"  Features: {f['features'].shape}")
            print(f"  Coordinates: {f['coordinates'].shape}")
            print(f"  Coord space: {f['coordinates'].attrs['space']}")
            print(f"  Score stats: min={f['features'].attrs['score_min']:.2f}, "
                  f"max={f['features'].attrs['score_max']:.2f}")
        
        print("\n✅ TopKSelector tests passed!\n")
    
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    print("\nRunning Top-K Sampling Tests...\n")
    
    test_tile_ranker()
    test_top_k_selector()
    
    print("="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
