"""
src/losses package initialization
"""
from .composite_loss import (
    CompositeReconstructionLoss,
    DifferentiableSSIM
)

__all__ = [
    "CompositeReconstructionLoss",
    "DifferentiableSSIM"
]

