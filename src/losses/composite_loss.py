"""
src/losses/composite_loss.py
Differentiable Composite Loss Function for Accelerated MRI Reconstruction.
Conforms directly to Capstone Report Section 4.4 (Equation 4.10):
    L_total = alpha * L_L1 + beta * L_SSIM + gamma * L_freq

Where:
  - L_L1: Pixel-wise L1 reconstruction loss on magnitude and complex channels.
  - L_SSIM: Structural Similarity loss (1 - SSIM) preserving anatomical contours.
  - L_freq: Frequency loss penalizing differences in k-space domain via 2D FFT.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms import (
    chan2_to_complex,
    complex_magnitude,
    fft2_centered
)


def create_gaussian_window_2d(window_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    """Creates a 2D Gaussian smoothing kernel for SSIM calculation."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    gauss = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    gauss = gauss / gauss.sum()

    window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class DifferentiableSSIM(nn.Module):
    """
    Differentiable Structural Similarity Index (SSIM) loss for 2D single-channel magnitude images.
    Returns 1 - SSIM in range [0, 2], where 0 indicates perfect structural match.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channel = 1

        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

        window = create_gaussian_window_2d(window_size, sigma, self.channel)
        self.register_buffer("window", window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1, img2: Tensors of shape (B, 1, H, W).
        Returns:
            mean SSIM value (scalar) or 1 - SSIM loss.
        """
        # Ensure window is on same device & dtype
        window = self.window.to(device=img1.device, dtype=img1.dtype)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / (
            (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2) + 1e-12
        )

        return ssim_map.mean()


class CompositeReconstructionLoss(nn.Module):
    """
    Composite Loss Function (Equation 4.10):
        L_total = alpha * L_L1 + beta * L_SSIM + gamma * L_freq
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        window_size: int = 11,
        sigma: float = 1.5
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.l1_loss = nn.L1Loss()
        self.ssim_module = DifferentiableSSIM(window_size=window_size, sigma=sigma, data_range=1.0)

    def forward(
        self,
        x_pred_chan2: torch.Tensor,
        x_gt_chan2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x_pred_chan2: Predicted image (B, 2, H, W)
            x_gt_chan2: Ground truth reference (B, 2, H, W)
            
        Returns:
            loss_total: Scaled total loss tensor for backprop.
            loss_dict: Dictionary with itemized detached float values for logging.
        """
        # 1. Complex-channel L1 loss
        loss_l1_complex = self.l1_loss(x_pred_chan2, x_gt_chan2)

        # Compute magnitude images: shape (B, 1, H, W)
        mag_pred = complex_magnitude(x_pred_chan2)
        mag_gt = complex_magnitude(x_gt_chan2)

        # Normalize magnitude to [0, 1] relative to gt maximum for reliable SSIM
        batch_max = torch.amax(mag_gt, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        mag_pred_scaled = mag_pred / batch_max
        mag_gt_scaled = mag_gt / batch_max

        loss_l1_mag = self.l1_loss(mag_pred_scaled, mag_gt_scaled)
        l1_total = 0.5 * loss_l1_complex + 0.5 * loss_l1_mag

        # 2. Structural Similarity Loss: L_SSIM = 1 - SSIM(mag_pred, mag_gt)
        ssim_val = self.ssim_module(mag_pred_scaled, mag_gt_scaled)
        loss_ssim = 1.0 - ssim_val

        # 3. Frequency Domain Loss: L_freq = ||F(x_pred) - F(x_gt)||_1
        x_pred_complex = chan2_to_complex(x_pred_chan2)
        x_gt_complex = chan2_to_complex(x_gt_chan2)
        k_pred = fft2_centered(x_pred_complex)
        k_gt = fft2_centered(x_gt_complex)
        loss_freq = F.l1_loss(k_pred.real, k_gt.real) + F.l1_loss(k_pred.imag, k_gt.imag)

        # Composite total loss (Equation 4.10)
        loss_total = (self.alpha * l1_total) + (self.beta * loss_ssim) + (self.gamma * loss_freq)

        loss_dict = {
            "loss_total": float(loss_total.item()),
            "loss_l1": float(l1_total.item()),
            "loss_ssim": float(loss_ssim.item()),
            "ssim": float(ssim_val.item()),
            "loss_freq": float(loss_freq.item())
        }

        return loss_total, loss_dict

