"""
Comprehensive Test Suite for GigaPath Pipeline

Run all tests:
    pytest tests/ -v

Run specific module:
    pytest tests/test_full_pipeline.py -v
    
Run with coverage:
    pytest tests/ --cov=src --cov-report=html
"""
import pytest
import torch
import numpy as np
import h5py
import tempfile
from pathlib import Path

# Test imports
from src.feature_extraction import BackboneLoader, FeatureExtractor
from src.sampling import TileRanker, TopKSelector
from src.mil import AttentionMIL, MILTrainer, MILDataset
from src.explainability import HeatmapGenerator


class TestFullPipeline:
    """Integration tests for complete pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature data."""
        features = np.random.randn(1000, 2048).astype(np.float32)
        coords = np.random.randint(0, 10000, (1000, 2)).astype(np.int32)
        return features, coords
    
    def test_phase1_feature_extraction(self, dummy_features):
        """Test Phase 1: Feature extraction pipeline."""
        # Load backbone
        model, feature_dim = BackboneLoader.load_backbone('resnet50-imagenet', freeze=True)
        
        assert feature_dim == 2048
        assert not next(model.parameters()).requires_grad  # Frozen
        
        # Test forward pass
        dummy_tiles = torch.randn(4, 3, 256, 256)
        with torch.no_grad():
            features = model(dummy_tiles)
        
        assert features.shape == (4, 2048)
        print("✓ Phase 1: Feature extraction working")
    
    def test_phase2_topk_sampling(self, temp_dir, dummy_features):
        """Test Phase 2: Top-K sampling pipeline."""
        features, coords = dummy_features
        
        # Save to HDF5
        input_h5 = temp_dir / 'test_features.h5'
        with h5py.File(input_h5, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('coordinates', data=coords)
            f['coordinates'].attrs['space'] = 'level_0'
        
        # Run Top-K sampling
        selector = TopKSelector(k=500, ranking_method='feature_norm')
        output_h5 = temp_dir / 'test_topk.h5'
        stats = selector.select_top_k_from_hdf5(str(input_h5), str(output_h5))
        
        assert stats['selected_k'] == 500
        assert stats['original_num_tiles'] == 1000
        
        # Verify output
        with h5py.File(output_h5, 'r') as f:
            assert f['features'].shape == (500, 2048)
            assert f['coordinates'].attrs['space'] == 'level_0'
        
        print("✓ Phase 2: Top-K sampling working")
    
    def test_phase3_mil_model(self):
        """Test Phase 3: MIL model architecture."""
        model = AttentionMIL(input_dim=2048, num_classes=2, dropout=0.25)
        
        # Test forward pass
        features = torch.randn(1000, 2048)
        logit, attention = model.forward(features, return_attention=True)
        
        # Verify shapes
        assert logit.shape == (1,), f"Expected (1,), got {logit.shape}"
        assert attention.shape == (1000,)
        
        # Verify attention normalization
        assert abs(attention.sum().item() - 1.0) < 1e-4
        
        # Test prediction
        result = model.predict_slide(features)
        assert 'prediction' in result
        assert 'probability' in result
        assert 0 <= result['probability'] <= 1
        
        print("✓ Phase 3: MIL model working")
    
    def test_phase4_heatmap_generation(self):
        """Test Phase 4: Heatmap generation."""
        heatmap_gen = HeatmapGenerator(colormap='jet', alpha=0.5)
        
        # Create dummy attention and coords
        attention = np.random.rand(100)
        attention = attention / attention.sum()  # Normalize
        coords = np.random.randint(0, 2048, (100, 2))
        
        # Generate heatmap
        heatmap = heatmap_gen.create_heatmap(
            attention_weights=attention,
            coordinates=coords,
            canvas_size=(2048, 2048),
            tile_size=256
        )
        
        assert heatmap.shape == (2048, 2048)
        assert heatmap.min() >= 0 and heatmap.max() <= 1
        
        # Apply colormap
        colored = heatmap_gen.apply_colormap(heatmap)
        assert colored.shape == (2048, 2048, 3)
        
        print("✓ Phase 4: Heatmap generation working")
    
    def test_gpu_memory_safety(self):
        """Test GPU memory usage is within limits."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device = torch.device('cuda')
        
        # Load model
        model = AttentionMIL(input_dim=2048, num_classes=2).to(device)
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass with K=1000 tiles
        features = torch.randn(1000, 2048).to(device)
        
        with torch.no_grad():
            logit, attention = model.forward(features, return_attention=True)
        
        # Check VRAM usage
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Peak VRAM usage: {peak_memory_mb:.2f} MB")
        
        # Should be under 8GB for RTX 4070
        assert peak_memory_mb < 8000, f"VRAM usage too high: {peak_memory_mb:.2f} MB"
        
        print(f"✓ GPU memory safe: {peak_memory_mb:.2f} MB / 8000 MB")
    
    def test_end_to_end_workflow(self, temp_dir, dummy_features):
        """Test complete pipeline end-to-end."""
        features, coords = dummy_features
        
        # 1. Save features (simulating Phase 1 output)
        features_h5 = temp_dir / 'slide_001.h5'
        with h5py.File(features_h5, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('coordinates', data=coords)
            f['coordinates'].attrs['space'] = 'level_0'
        
        # 2. Top-K sampling (Phase 2)
        selector = TopKSelector(k=500)
        topk_h5 = temp_dir / 'slide_001_topk.h5'
        selector.select_top_k_from_hdf5(str(features_h5), str(topk_h5))
        
        # 3. Load features for MIL (Phase 3)
        with h5py.File(topk_h5, 'r') as f:
            topk_features = torch.from_numpy(f['features'][:]).float()
            topk_coords = f['coordinates'][:]
        
        # 4. MIL inference
        model = AttentionMIL(input_dim=2048, num_classes=2)
        result = model.predict_slide(topk_features, return_attention=True)
        
        # 5. Heatmap generation (Phase 4)
        heatmap_gen = HeatmapGenerator()
        heatmap = heatmap_gen.create_heatmap(
            attention_weights=result['attention_weights'],
            coordinates=topk_coords,
            canvas_size=(2048, 2048)
        )
        
        # Verify complete workflow
        assert result['prediction'] in [0, 1]
        assert heatmap.shape == (2048, 2048)
        
        print("✓ End-to-end pipeline working")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
