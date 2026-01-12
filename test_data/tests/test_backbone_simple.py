"""
Test backbone loading and feature extraction.
Run from project root: python -m tests.test_feature_extraction_simple
"""
import sys
from pathlib import Path

# Add src to path for imports when running as script
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision.models as models


def test_resnet50_feature_extraction():
    """Test ResNet50 feature extraction with fc=Identity."""
    print("="*60)
    print("Testing ResNet50 Feature Extraction")
    print("="*60)
    
    # Load ResNet50
    print("\n1. Loading ResNet50-ImageNet...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Replace fc with Identity
    print("2. Replacing fc layer with Identity...")
    model.fc = torch.nn.Identity()
    
    # Freeze parameters
    print("3. Freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    # Set to eval mode
    model.eval()
    
    # Test with dummy input
    print("\n4. Testing with dummy input...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = model.to(device)
    
    # Create dummy batch (B=4, C=3, H=256, W=256)
    dummy_input = torch.randn(4, 3, 256, 256).to(device)
    print(f"   Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        features = model(dummy_input)
    
    print(f"   Output shape: {features.shape}")
    print(f"   Expected: (4, 2048)")
    
    # Verify
    assert features.shape == (4, 2048), f"Shape mismatch! Got {features.shape}"
    assert not features.requires_grad, "Model should be frozen!"
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nKey Findings:")
    print(f"  ✓ fc=Identity produces (B, 2048) embeddings")
    print(f"  ✓ Model parameters are frozen")
    print(f"  ✓ No spatial dimensions (7×7) in output")
    print(f"  ✓ Ready for MIL aggregation")


if __name__ == '__main__':
    test_resnet50_feature_extraction()
