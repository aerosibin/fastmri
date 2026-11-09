"""
src/models/data_consistency.py
Physics-based Data Consistency (DC) projection layer for Accelerated MRI Reconstruction.
Enforces that frequencies actually measured by the MR scanner are preserved in the final output:
    k_pred = F(x_hat)
    k_DC = (1 - M) * k_pred + y_measured
    x_DC = F^{-1}(k_DC)
"""

import torch
import torch.nn as nn
from src.data.transforms import (
    chan2_to_complex,
    complex_to_chan2,
    fft2_centered,
    ifft2_centered
)


class DataConsistency(nn.Module):
    """
    Differentiable Data Consistency layer.
    Replaces sampled k-space lines in the predicted image with original acquired scanner measurements.
    """
    def __init__(self, soft_weight: float = 1.0):
        super().__init__()
        self.soft_weight = soft_weight

    def forward(
        self,
        x_pred_chan2: torch.Tensor,
        kspace_measured_chan2: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_pred_chan2: Predicted complex image (B, 2, H, W).
            kspace_measured_chan2: Undersampled acquired k-space (B, 2, H, W) or complex.
            mask: Binary sampling mask (B, 1, H, W) or (B, H, W).
            
        Returns:
            x_dc_chan2: Data-consistent complex image (B, 2, H, W).
        """
        # Convert prediction to complex domain
        x_pred_complex = chan2_to_complex(x_pred_chan2)
        # Transform to k-space
        k_pred = fft2_centered(x_pred_complex)

        # Convert measured kspace to complex
        if torch.is_complex(kspace_measured_chan2):
            k_meas = kspace_measured_chan2
        else:
            k_meas = chan2_to_complex(kspace_measured_chan2)

        if mask.ndim == 4 and mask.shape[1] == 1:
            mask_squeeze = mask.squeeze(1)
        else:
            mask_squeeze = mask

        # Shift DC to center for matching mask coordinates
        k_pred_shifted = torch.fft.fftshift(k_pred, dim=(-2, -1))
        k_meas_shifted = torch.fft.fftshift(k_meas, dim=(-2, -1))

        # Data consistency formula
        # Retain acquired lines from k_meas, keep predicted lines where mask == 0
        if self.soft_weight >= 1.0:
            k_dc_shifted = (1.0 - mask_squeeze) * k_pred_shifted + mask_squeeze * k_meas_shifted
        else:
            k_dc_shifted = k_pred_shifted + self.soft_weight * mask_squeeze * (k_meas_shifted - k_pred_shifted)

        # Shift back and inverse transform to spatial domain
        k_dc = torch.fft.ifftshift(k_dc_shifted, dim=(-2, -1))
        x_dc_complex = ifft2_centered(k_dc)

        return complex_to_chan2(x_dc_complex)

