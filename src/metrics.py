"""
src/metrics.py
Evaluation metrics for Accelerated MRI Reconstruction.
Conforms directly to Capstone Report Section 4.5:
  - Peak Signal-to-Noise Ratio (PSNR) (Equation 4.11)
  - Structural Similarity Index Measure (SSIM)
  - Normalized Mean Squared Error (NMSE)
  - Mean Absolute Error (MAE)
"""

from typing import Dict, Optional, Union
import numpy as np
import torch
from skimage.metrics import (
    peak_signal_noise_ratio as skimage_psnr,
    structural_similarity as skimage_ssim
)


def compute_psnr_np(
    target: np.ndarray,
    prediction: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Computes PSNR = 10 * log10(MAX_I^2 / MSE) (Equation 4.11)
    """
    if data_range is None:
        data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = 1.0

    return float(skimage_psnr(target, prediction, data_range=data_range))


def compute_ssim_np(
    target: np.ndarray,
    prediction: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Computes SSIM on magnitude 2D brain images.
    """
    if data_range is None:
        data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = 1.0

    return float(skimage_ssim(target, prediction, data_range=data_range))


def compute_nmse_np(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Computes Normalized Mean Squared Error: sum((target - pred)^2) / sum(target^2)
    """
    err = np.sum((target - prediction) ** 2)
    ref = np.sum(target ** 2)
    return float(err / max(ref, 1e-12))


def compute_mae_np(target: np.ndarray, prediction: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    return float(np.mean(np.abs(target - prediction)))


def evaluate_batch_metrics(
    target_mag: Union[np.ndarray, torch.Tensor],
    prediction_mag: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Evaluates PSNR, SSIM, NMSE, and MAE over a batch of 2D magnitude images.
    """
    if isinstance(target_mag, torch.Tensor):
        target_np = target_mag.detach().cpu().squeeze().numpy()
    else:
        target_np = np.squeeze(target_mag)

    if isinstance(prediction_mag, torch.Tensor):
        pred_np = prediction_mag.detach().cpu().squeeze().numpy()
    else:
        pred_np = np.squeeze(prediction_mag)

    if target_np.ndim == 2:
        target_np = target_np[np.newaxis, ...]
        pred_np = pred_np[np.newaxis, ...]

    psnr_vals = []
    ssim_vals = []
    nmse_vals = []
    mae_vals = []

    for i in range(target_np.shape[0]):
        t = target_np[i]
        p = pred_np[i]

        dr = float(t.max() - t.min())
        if dr <= 0:
            dr = 1.0

        psnr_vals.append(compute_psnr_np(t, p, data_range=dr))
        ssim_vals.append(compute_ssim_np(t, p, data_range=dr))
        nmse_vals.append(compute_nmse_np(t, p))
        mae_vals.append(compute_mae_np(t, p))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "nmse": float(np.mean(nmse_vals)),
        "mae": float(np.mean(mae_vals))
    }
