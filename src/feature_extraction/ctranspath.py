"""
CTransPath backbone adapter.
Wraps timm's Swin Transformer and loads custom CTransPath weights.
"""
import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

class CTransPath(nn.Module):
    """
    CTransPath: Transformer-based Unsupervised Contrastive Learning for Histopathology.
    Uses Swin-T architecture.
    """
    
    def __init__(self, pretrained: bool = False, weights_path: Optional[str] = None):
        super(CTransPath, self).__init__()
        
        # Load Swin-T from timm
        # CTransPath uses swin_tiny_patch4_window7_224
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
        
        # Standard Swin-T has a classification head.
        # We need to remove it or use forward_features.
        # timm's forward_features returns (B, 768, 7, 7) or similar depending on output_fmt
        # But we want the global pool.
        # Let's use reset_classifier(0) to remove the head and global pool
        self.model.reset_classifier(0) # This makes it output (B, 768) flattened features
        
        self.feature_dim = 768
        
        if pretrained and weights_path:
            self._load_weights(weights_path)
    
    def _load_weights(self, weights_path: str):
        """Load CTransPath custom weights."""
        path = Path(weights_path)
        if not path.exists():
            logger.warning(f"CTransPath weights not found at {weights_path}")
            logger.warning("Initializing with random weights! (Low performance expected)")
            return
            
        logger.info(f"Loading CTransPath weights from {weights_path}")
        
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # CTransPath weights might have 'model.' prefix or different keys
            # timm keys usually match standard Swin
            # We might need to map keys if they differ. 
            # Usual CTransPath weights from official repo are for the full model.
            
            # Let's try flexible loading
            msg = self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
            
        except Exception as e:
            logger.error(f"Failed to load CTransPath weights: {e}")
            raise e

    def forward(self, x):
        return self.model(x)
