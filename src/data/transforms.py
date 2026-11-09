"""
src/data/transforms.py
Tensor transformations, Fourier operations, and channel formatting for Accelerated MRI.
Conforms directly to Capstone Report Section 4.3.2:
  - 2D Inverse Fourier Transform: x_ZF = F^{-1}(y)
  - Real and Imaginary channel separation and stacking:
        X_input = [Re(x_ZF), Im(x_ZF)]^T in R^{B x 2 x 256 x 256}
  - Intensity normalization and magnitude extraction.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch


def complex_to_chan2(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts complex tensor with shape (..., H, W) to 2-channel real tensor with shape (..., 2, H, W).
    Channel 0: Real component.
    Channel 1: Imaginary component.
    """
    if not torch.is_complex(tensor):
        raise TypeError(f"Expected complex tensor, got {tensor.dtype}")
    real = tensor.real
    imag = tensor.imag
    # Stack along the second-to-last dimension (channel dimension)
    return torch.stack([real, imag], dim=-3)


def chan2_to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts 2-channel real tensor with shape (..., 2, H, W) to complex tensor with shape (..., H, W).
    """
    if tensor.shape[-3] != 2:
        raise ValueError(f"Expected channel dimension of size 2 at dim -3, got shape {tensor.shape}")
    real = tensor.select(-3, 0)
    imag = tensor.select(-3, 1)
    return torch.complex(real, imag)


def complex_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes magnitude image |x| = sqrt(Re^2 + Im^2) from either:
      - a complex tensor (..., H, W)
      - a 2-channel real tensor (..., 2, H, W)
    Returns:
      Magnitude tensor with shape (..., 1, H, W) if 2-channel, or (..., H, W) if complex.
    """
    if torch.is_complex(tensor):
        return torch.abs(tensor)
    elif tensor.ndim >= 3 and tensor.shape[-3] == 2:
        real = tensor.select(-3, 0)
        imag = tensor.select(-3, 1)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
        # Keep channel dimension: (..., 1, H, W)
        return mag.unsqueeze(-3)
    else:
        raise ValueError(f"Unsupported shape or dtype for complex magnitude: {tensor.shape}, {tensor.dtype}")


def ifft2_centered(kspace: torch.Tensor) -> torch.Tensor:
    """
    Computes 2D Inverse Fourier Transform for CC359 k-space.
    In the CC359 raw dataset, the DC peak is natively at index [0, 0].
    Therefore, standard torch.fft.ifft2 reconstructs the centered spatial image directly.
    """
    return torch.fft.ifft2(kspace, dim=(-2, -1), norm="ortho")


def fft2_centered(image: torch.Tensor) -> torch.Tensor:
    """
    Computes 2D Forward Fourier Transform from spatial domain to k-space.
    Outputs k-space with DC at [0, 0] to match CC359 unshifted convention.
    """
    return torch.fft.fft2(image, dim=(-2, -1), norm="ortho")


def zero_filled_reconstruction(
    kspace_undersampled: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the zero-filled reconstruction x_ZF = F^{-1}(y) (Equation 4.5)
    and formats into the two-channel PyTorch input tensor (Equation 4.6 & 4.7):
        X_input = [Re(x_ZF), Im(x_ZF)]^T in R^{B x 2 x 256 x 256}
        
    Args:
        kspace_undersampled: Complex tensor (B, H, W) or (H, W).
        
    Returns:
        x_zf_chan2: 2-channel real tensor (..., 2, H, W).
        x_zf_complex: Complex spatial tensor (..., H, W).
    """
    x_zf_complex = ifft2_centered(kspace_undersampled)
    x_zf_chan2 = complex_to_chan2(x_zf_complex)
    return x_zf_chan2, x_zf_complex


def normalize_slice(
    image_chan2: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes complex 2-channel MRI image by its standard deviation so that
    input magnitudes are standard normal scale O(1).
    
    Args:
        image_chan2: Tensor of shape (..., 2, H, W).
        scale: Optional precomputed scale factor. If None, computes scale as std(image_chan2).
        
    Returns:
        normalized_image: Tensor of shape (..., 2, H, W).
        scale: Scale factor tensor used for normalization.
    """
    if scale is None:
        # Compute standard deviation across spatial dimensions (-2, -1) and channels (-3)
        dims = (-3, -2, -1)
        scale = torch.std(image_chan2, dim=dims, keepdim=True) + eps

    norm_image = image_chan2 / scale
    return norm_image, scale


def unnormalize_slice(
    image_chan2: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Reverses normalization: x_unnorm = x_norm * scale.
    """
    return image_chan2 * scale

