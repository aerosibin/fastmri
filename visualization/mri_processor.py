"""
mri_processor.py
Core processing utilities for Single-Channel Complex k-Space MRI Data (CC359 Dataset).
Handles:
  - Subject discovery across Train & Val directories
  - Loading complex k-space (.npy format: real + imaginary channels)
  - 2D IFFT / FFT image reconstruction
  - 3D Multi-Planar Reformatting (Sagittal, Coronal, Axial orthogonal views)
  - Retrospective undersampling simulation (Gaussian, Random, Equispaced masks)
  - Quantitative image quality metrics (PSNR, SSIM, NMSE)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def find_all_subjects(dataset_root: Union[str, Path] = "CC359/Raw-data/Single-channel") -> List[Dict[str, Union[str, Path]]]:
    """
    Discovers all available .npy files across Train_part1, Train_part2, Train_part3, and Val.
    Returns a sorted list of subject metadata dicts.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root.resolve()}")

    subjects = []
    # Find all .npy files recursively
    for npy_file in sorted(root.glob("**/*.npy")):
        rel_parts = npy_file.relative_to(root).parts
        split_group = rel_parts[0]  # Train_part1, Train_part2, Train_part3, or Val
        is_val = "Val" in split_group
        subject_id = npy_file.stem.split(".")[0]  # e.g., e14155s3_P69120

        subjects.append({
            "subject_id": subject_id,
            "filename": npy_file.name,
            "filepath": npy_file,
            "split": "Validation" if is_val else "Training",
            "group": split_group,
            "size_mb": round(npy_file.stat().st_size / (1024 * 1024), 2)
        })
    return subjects


def load_subject_kspace(filepath: Union[str, Path], mmap: bool = True) -> np.ndarray:
    """
    Loads raw single-channel MRI array with shape (N_slices, 256, 256, 2)
    and converts to complex64/complex128 array (N_slices, 256, 256).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    raw = np.load(filepath, mmap_mode="r" if mmap else None)
    # Channel 0: Real, Channel 1: Imaginary
    complex_kspace = raw[..., 0].astype(np.float32) + 1j * raw[..., 1].astype(np.float32)
    return complex_kspace


def reconstruct_slice(kspace_slice: np.ndarray, is_shifted: bool = False) -> np.ndarray:
    """
    Reconstructs spatial domain magnitude image from 2D complex k-space.
    In CC359 raw data, DC is natively at (0, 0), so standard ifft2 directly reconstructs.
    If is_shifted=True (DC at center), ifftshift is applied before ifft2.
    """
    if is_shifted:
        kspace_slice = np.fft.ifftshift(kspace_slice)
    img_complex = np.fft.ifft2(kspace_slice)
    return np.abs(img_complex)


def reconstruct_volume(complex_kspace: np.ndarray) -> np.ndarray:
    """
    Reconstructs entire 3D volume (N_slices, 256, 256) via 2D IFFT along slices.
    """
    img_complex = np.fft.ifft2(complex_kspace, axes=(-2, -1))
    return np.abs(img_complex)


def get_orthogonal_views(
    volume: np.ndarray,
    sag_idx: Optional[int] = None,
    cor_idx: Optional[int] = None,
    axi_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts anatomically oriented orthogonal cuts from 3D volume (N_slices, 256, 256).
    - Sagittal: acquired plane volume[sag_idx, :, :]
    - Coronal: volume[:, :, cor_idx].T
    - Axial: volume[:, axi_idx, :].T
    """
    n_sag, n_rows, n_cols = volume.shape
    if sag_idx is None:
        sag_idx = n_sag // 2
    if cor_idx is None:
        cor_idx = n_cols // 2
    if axi_idx is None:
        axi_idx = n_rows // 2

    sag = volume[sag_idx, :, :]
    cor = volume[:, :, cor_idx].T
    axi = volume[:, axi_idx, :].T

    return sag, cor, axi


def generate_undersampling_mask(
    shape: Tuple[int, int] = (256, 256),
    acceleration: float = 4.0,
    center_fraction: float = 0.08,
    mask_type: str = "gaussian",
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Creates a 1D phase-encode sampling mask replicated across readout direction.
    Args:
        shape: (H, W) = (PhaseEncode, Readout)
        acceleration: Target acceleration factor R (e.g. 4.0, 8.0)
        center_fraction: Proportion of low-frequency lines fully sampled (ACS region)
        mask_type: 'gaussian' (variable density), 'random' (uniform random), or 'equispaced'
        seed: Random seed for reproducibility
    Returns:
        Binary mask of shape (H, W) with 1 for sampled lines, 0 for unsampled.
    """
    if seed is not None:
        np.random.seed(seed)

    n_lines = shape[0]
    num_low_freqs = int(round(n_lines * center_fraction))
    pad = (n_lines - num_low_freqs + 1) // 2

    mask_1d = np.zeros(n_lines, dtype=np.float32)
    # Keep center Auto-Calibration Signal (ACS) lines
    mask_1d[pad : pad + num_low_freqs] = 1.0

    target_total_samples = int(round(n_lines / acceleration))
    remaining_samples_needed = target_total_samples - num_low_freqs

    if remaining_samples_needed > 0:
        outer_indices = np.where(mask_1d == 0)[0]
        if mask_type == "gaussian":
            # Variable-density Gaussian probability favoring mid frequencies over outer edges
            center = n_lines / 2.0
            sigma = n_lines / 3.0
            weights = np.exp(-0.5 * ((outer_indices - center) / sigma) ** 2)
            probs = weights / weights.sum()
            chosen = np.random.choice(
                outer_indices,
                size=min(remaining_samples_needed, len(outer_indices)),
                replace=False,
                p=probs
            )
            mask_1d[chosen] = 1.0
        elif mask_type == "random":
            chosen = np.random.choice(
                outer_indices,
                size=min(remaining_samples_needed, len(outer_indices)),
                replace=False
            )
            mask_1d[chosen] = 1.0
        elif mask_type == "equispaced":
            step = int(round(acceleration))
            equispaced_idx = np.arange(0, n_lines, step)
            mask_1d[equispaced_idx] = 1.0
            # Re-ensure center is intact
            mask_1d[pad : pad + num_low_freqs] = 1.0

    # Expand 1D mask to 2D along readout (columns)
    mask_2d = np.repeat(mask_1d[:, None], shape[1], axis=1)
    return mask_2d


def simulate_undersampling(
    kspace_slice: np.ndarray,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies undersampling mask to centered k-space and computes zero-filled reconstruction.
    Returns:
        (undersampled_kspace_shifted, zero_filled_image)
    """
    # Shift DC to center to apply centered mask
    k_shifted = np.fft.fftshift(kspace_slice)
    k_masked_shifted = k_shifted * mask

    # Shift back and compute IFFT
    k_masked_unshifted = np.fft.ifftshift(k_masked_shifted)
    zf_image = np.abs(np.fft.ifft2(k_masked_unshifted))

    return k_masked_shifted, zf_image


def compute_reconstruction_metrics(gt: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    """
    Computes standard quantitative metrics between Ground Truth and Reconstructed/Zero-Filled image.
    - PSNR: Peak Signal to Noise Ratio (dB)
    - SSIM: Structural Similarity Index
    - NMSE: Normalized Mean Squared Error
    - MAE: Mean Absolute Error
    """
    data_range = float(gt.max() - gt.min())
    if data_range == 0:
        data_range = 1.0

    psnr_val = float(compute_psnr(gt, recon, data_range=data_range))
    ssim_val = float(compute_ssim(gt, recon, data_range=data_range))
    nmse_val = float(np.sum((gt - recon) ** 2) / np.maximum(np.sum(gt ** 2), 1e-12))
    mae_val = float(np.mean(np.abs(gt - recon)))

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "nmse": nmse_val,
        "mae": mae_val
    }
