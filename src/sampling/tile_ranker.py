"""
Tile ranking module for Top-K sampling.
Ranks tiles by informativeness for efficient MIL training.
"""
import numpy as np
import torch
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TileRanker:
    """Rank tiles by informativeness."""
    
    @staticmethod
    def rank_by_feature_norm(features: np.ndarray) -> np.ndarray:
        """
        Rank tiles by L2 norm of feature vectors.
        
        Higher norm indicates stronger activations = more informative.
        
        Args:
            features: Feature array (N, feature_dim)
        
        Returns:
            Ranking scores (N,) - higher is better
        """
        scores = np.linalg.norm(features, axis=1)
        logger.debug(f"Feature norm ranking: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")
        return scores
    
    @staticmethod
    def rank_by_attention(
        features: np.ndarray,
        mil_model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        Rank tiles by attention weights from trained MIL model.
        
        Note: Requires trained MIL model (Phase 3).
        
        Args:
            features: Feature array (N, feature_dim)
            mil_model: Trained MIL model with attention mechanism
            device: Torch device
        
        Returns:
            Attention scores (N,) - higher is better
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().to(device)
        
        # Get attention weights
        mil_model.eval()
        with torch.no_grad():
            # Assuming model has get_attention_weights method
            attention = mil_model.get_attention_weights(features_tensor)
        
        # Convert to numpy
        scores = attention.cpu().numpy()
        
        logger.debug(f"Attention ranking: min={scores.min():.4f}, max={scores.max():.4f}")
        return scores
    
    @staticmethod
    def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
        """
        Select indices of top-K scores.
        
        Args:
            scores: Ranking scores (N,)
            k: Number of top tiles to select
        
        Returns:
            Indices of top-K tiles (K,) in descending score order
        """
        n = len(scores)
        
        # Handle case where k >= n
        if k >= n:
            logger.warning(f"K={k} >= num_tiles={n}, keeping all tiles")
            # Return all indices sorted by score (descending)
            return np.argsort(scores)[::-1]
        
        # Get top-k indices (descending order)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        logger.debug(f"Selected top {k} tiles from {n} (ratio: {k/n:.2%})")
        
        return top_k_indices
    
    @staticmethod
    def get_score_stats(scores: np.ndarray) -> dict:
        """
        Compute summary statistics of scores.
        
        Storage optimization: Store only stats, not full array.
        
        Args:
            scores: Ranking scores (N,)
        
        Returns:
            Dictionary with min, max, mean, std
        """
        stats = {
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores))
        }
        return stats


if __name__ == '__main__':
    # Test tile ranker
    print("Testing TileRanker...")
    
    # Create dummy features
    features = np.random.randn(2000, 2048).astype(np.float32)
    
    # Test feature norm ranking
    scores = TileRanker.rank_by_feature_norm(features)
    print(f"✓ Feature norm scores: shape={scores.shape}")
    
    # Test top-K selection
    k = 1000
    top_k_indices = TileRanker.select_top_k(scores, k)
    print(f"✓ Top-K selection: selected {len(top_k_indices)} / {len(scores)}")
    
    # Verify descending order
    assert all(scores[top_k_indices[i]] >= scores[top_k_indices[i+1]] 
               for i in range(len(top_k_indices)-1)), "Scores not in descending order!"
    print(f"✓ Scores in descending order")
    
    # Test score stats
    stats = TileRanker.get_score_stats(scores[top_k_indices])
    print(f"✓ Score stats: {stats}")
    
    print("\n✅ TileRanker tests passed!")
