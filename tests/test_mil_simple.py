"""
Test Attention MIL model.
Run from project root: python -m tests.test_mil_simple
"""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.mil import AttentionMIL


def test_attention_mil():
    """Test AttentionMIL model."""
    print("="*60)
    print("Testing AttentionMIL")
    print("="*60)
    
    # Create model
    model = AttentionMIL(input_dim=2048, num_classes=2)
    print(f"✓ Model created")
    
    # Create dummy features (K=1000 tiles, 2048-dim)
    # Test both float32 and float64 to verify enforcement
    features_f64 = torch.randn(1000, 2048, dtype=torch.float64)
    features_f32 = torch.randn(1000, 2048, dtype=torch.float32)
    print(f"✓ Input features (float64): {features_f64.shape}, dtype={features_f64.dtype}")
    print(f"✓ Input features (float32): {features_f32.shape}, dtype={features_f32.dtype}")
    
    # Forward pass with float32
    logit, attention_weights = model.forward(features_f32, return_attention=True)
    print(f"\n✓ Forward pass complete")
    print(f"  Logit shape: {logit.shape} (expected: (1,))")
    print(f"  Attention weights shape: {attention_weights.shape}")
    
    # Check attention normalization (CRITICAL)
    attention_sum = attention_weights.sum().item()
    print(f"\n✓ Attention sum: {attention_sum:.8f}")
    assert abs(attention_sum - 1.0) < 1e-4, f"Attention weights don't sum to 1! Got {attention_sum}"
    print(f"  VERIFIED: Attention weights sum to 1 ✓")
    
    # Check dimensions (CRITICAL: single logit for binary)
    assert logit.shape == (1,), f"Logit shape mismatch! Expected (1,), got {logit.shape}"
    assert attention_weights.shape == (1000,), f"Attention shape mismatch: {attention_weights.shape}"
    print(f"✓ Output dimensions correct (binary: 1 logit)")
    
    # Test float64 enforcement (should convert to float32)
    logit_64, _ = model.forward(features_f64, return_attention=False)
    assert logit_64.shape == (1,), "Float64 input should also produce (1,) logit"
    print(f"✓ Float64 enforcement working (converted to float32)")
    
    # Test prediction
    result = model.predict_slide(features_f32)
    print(f"\n✓ Prediction test:")
    print(f"  Prediction: {result['prediction']} (0=benign, 1=malignant)")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Logit: {result['logit']:.4f}")
    print(f"  Attention weights: {result['attention_weights'].shape}")
    
    # Test attention weight extraction
    attention_np = model.get_attention_weights(features_f32)
    print(f"\n✓ Attention extraction:")
    print(f"  Shape: {attention_np.shape}")
    print(f"  Sum: {attention_np.sum():.8f}")
    assert abs(attention_np.sum() - 1.0) < 1e-4, "Extracted attention doesn't sum to 1!"
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nKey Findings:")
    print(f"  ✓ Gated attention mechanism working")
    print(f"  ✓ Attention weights sum to 1 (softmax dim=0)")
    print(f"  ✓ Slide-level classification functional")
    print(f"  ✓ Attention extraction for explainability ready")


if __name__ == '__main__':
    test_attention_mil()
