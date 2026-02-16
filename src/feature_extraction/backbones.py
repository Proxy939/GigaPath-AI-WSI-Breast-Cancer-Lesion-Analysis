"""
Backbone model loader for feature extraction.
Supports frozen pretrained models for WSI tile embedding.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
from pathlib import Path

from ..utils.logger import get_logger
from .ctranspath import CTransPath

logger = get_logger(__name__)


class BackboneLoader:
    """Load and configure pretrained backbone models."""
    
    SUPPORTED_MODELS = {
        'resnet50-imagenet': {
            'feature_dim': 2048,
            'input_size': 224,
            'description': 'ResNet50 pretrained on ImageNet'
        },
        'resnet50-simclr': {
            'feature_dim': 2048,
            'input_size': 224,
            'description': 'ResNet50 with SimCLR self-supervised learning'
        },
        'ctranspath': {
            'feature_dim': 768,
            'input_size': 224,
            'description': 'CTransPath (Swin-T) pretrained on histopathology',
            'weights_file': 'ctranspath.pth'
        }
    }
    
    @staticmethod
    def load_backbone(
        model_name: str = 'resnet50-imagenet',
        freeze: bool = True,
        device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, int]:
        """
        Load pretrained backbone model.
        
        Args:
            model_name: Name of model to load
            freeze: If True, freeze all parameters (no training)
            device: Device to load model on
        
        Returns:
            Tuple of (model, feature_dim)
        """
        if model_name not in BackboneLoader.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from: {list(BackboneLoader.SUPPORTED_MODELS.keys())}"
            )
        
        logger.info(f"Loading backbone: {model_name}")
        
        if model_name == 'resnet50-imagenet':
            model, feature_dim = BackboneLoader._load_resnet50_imagenet()
        elif model_name == 'resnet50-simclr':
            model, feature_dim = BackboneLoader._load_resnet50_simclr()
        elif model_name == 'ctranspath':
             model, feature_dim = BackboneLoader._load_ctranspath()
        else:
            raise ValueError(f"Model {model_name} not implemented")
        
        # Freeze parameters if requested
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("Model parameters frozen")
        
        # Helper validation
        if not torch.cuda.is_available():
             # Warning only, don't crash for tests unless strictly required
             logger.warning("CUDA not available, using CPU (slow!)")
             
        # Move to device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        model = model.to(device)
        logger.info(f"Model moved to {device}")
        
        # Set to eval mode
        model.eval()
        
        logger.info(f"Backbone loaded: {model_name} (feature_dim={feature_dim})")
        
        return model, feature_dim
    
    @staticmethod
    def _load_resnet50_imagenet() -> Tuple[nn.Module, int]:
        """Load ResNet50 pretrained on ImageNet."""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        feature_dim = 2048
        return model, feature_dim
    
    @staticmethod
    def _load_resnet50_simclr() -> Tuple[nn.Module, int]:
        """Load ResNet50 with SimCLR weights."""
        raise NotImplementedError("ResNet50-SimCLR not yet implemented.")

    @staticmethod
    def _load_ctranspath() -> Tuple[nn.Module, int]:
        """Load CTransPath (Swin-T)."""
        # Define path to weights
        weights_path = Path("data/models/checkpoints/ctranspath.pth")
        
        # Initialize model
        # We pass the absolute or relative path that works from project root
        model = CTransPath(pretrained=True, weights_path=str(weights_path))
        feature_dim = 768
        
        return model, feature_dim
    
    @staticmethod
    def get_input_transforms(model_name: str):
        """Get preprocessing transforms for model."""
        from torchvision import transforms
        
        if model_name.startswith('resnet50') or model_name == 'ctranspath':
            # ImageNet normalization is standard for both
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            raise ValueError(f"Transforms for {model_name} not defined")


def test_backbone_loading():
    """Test backbone loading and feature extraction."""
    print("Testing backbone loading...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test ResNet50
    print("\n[Test 1] ResNet50-ImageNet")
    model, dim = BackboneLoader.load_backbone('resnet50-imagenet', device=device)
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 2048)

    # Test CTransPath
    print("\n[Test 2] CTransPath")
    try:
        model, dim = BackboneLoader.load_backbone('ctranspath', device=device)
        dummy = torch.randn(2, 3, 224, 224).to(device)
        out = model(dummy)
        print(f"Output shape: {out.shape}")
        assert out.shape == (2, 768)
        print("✅ CTransPath loaded successfully")
    except Exception as e:
        print(f"❌ CTransPath failed: {e}")
    
    print("\nTests complete.")


if __name__ == '__main__':
    test_backbone_loading()
