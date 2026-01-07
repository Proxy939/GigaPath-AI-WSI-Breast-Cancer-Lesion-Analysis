"""
Multiple Instance Learning (MIL) module.
"""
from .attention_mil import AttentionMIL, GatedAttention

__all__ = [
    'AttentionMIL',
    'GatedAttention',
]
