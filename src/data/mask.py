"""
src/data/mask.py
Physics simulation module for retrospective undersampling and acquisition error modeling.
Conforms directly to Capstone Report Chapter 4:
  - 4.2.2: Retrospective undersampling (R=4 primary, R=8 stress test) with variable-density Gaussian sampling
           preserving 8%-16% central ACS lines.
  - 4.3.1: Acquisition error models (Undersampling artifacts, Gaussian noise, Phase errors, Motion artifacts).
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch


def generate_undersampling_mask(
    shape: Tuple[int, int] = (256, 256),
    acceleration: float = 4.0,
    center_fraction: float = 0.08,
    mask_type: str = "gaussian",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a 1D phase-encoding undersampling mask replicated across the readout dimension.
    
    Args:
        shape: (H, W) = (PhaseEncode, Readout), standard (256, 256).
        acceleration: Acceleration factor R (e.g., 4.0 for primary, 8.0 for stress test).
        center_fraction: Fraction of low-frequency central lines preserved for Auto-Calibration Signal (ACS).
        mask_type: 'gaussian' (variable density, default), 'random' (uniform), or 'equispaced'.
        seed: Random seed for deterministic mask generation.
        
    Returns:
        mask_2d: Binary numpy array of shape (H, W) with 1 for sampled lines and 0 for unsampled lines.
    """
    rng = np.random.default_rng(seed)
    n_lines, readout_len = shape

    # Compute number of fully sampled low-frequency lines (ACS region)
    num_low_freqs = int(round(n_lines * center_fraction))
    pad = (n_lines - num_low_freqs + 1) // 2

    mask_1d = np.zeros(n_lines, dtype=np.float32)
    # Center lines are guaranteed to be sampled (ACS)
    mask_1d[pad : pad + num_low_freqs] = 1.0

    # Number of lines to acquire in total for acceleration factor R
    target_total_samples = int(round(n_lines / acceleration))
    remaining_samples_needed = target_total_samples - num_low_freqs

    if remaining_samples_needed > 0:
        outer_indices = np.where(mask_1d == 0)[0]

        if mask_type == "gaussian":
            # Variable-density Gaussian sampling favoring mid frequencies
            center = n_lines / 2.0
            sigma = n_lines / 3.0
            weights = np.exp(-0.5 * ((outer_indices - center) / sigma) ** 2)
            probs = weights / weights.sum()
            chosen = rng.choice(
                outer_indices,
                size=min(remaining_samples_needed, len(outer_indices)),
                replace=False,
                p=probs
            )
            mask_1d[chosen] = 1.0

        elif mask_type == "random":
            # Uniform random sampling outside ACS
            chosen = rng.choice(
                outer_indices,
                size=min(remaining_samples_needed, len(outer_indices)),
                replace=False
            )
            mask_1d[chosen] = 1.0

        elif mask_type == "equispaced":
            # Regular equispaced subsampling
            step = int(round(acceleration))
            equispaced_idx = np.arange(0, n_lines, step)
            mask_1d[equispaced_idx] = 1.0
            # Re-ensure central ACS region is fully sampled
            mask_1d[pad : pad + num_low_freqs] = 1.0

        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}. Choose 'gaussian', 'random', or 'equispaced'.")

    # Replicate 1D mask across the readout direction (columns)
    mask_2d = np.repeat(mask_1d[:, None], readout_len, axis=1)
    return mask_2d


def apply_undersampling(
    kspace: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor]
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Applies the binary mask M to fully sampled k-space:
        y = M * kspace
    Handles shifted k-space where DC is centered before multiplying.
    """
    if isinstance(kspace, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).to(device=kspace.device, dtype=kspace.real.dtype)
        
        # Shift DC to center, apply mask, shift back
        k_shifted = torch.fft.fftshift(kspace, dim=(-2, -1))
        k_masked_shifted = k_shifted * mask
        y = torch.fft.ifftshift(k_masked_shifted, dim=(-2, -1))
        return y, k_masked_shifted
    else:
        k_shifted = np.fft.fftshift(kspace, axes=(-2, -1))
        k_masked_shifted = k_shifted * mask
        y = np.fft.ifftshift(k_masked_shifted, axes=(-2, -1))
        return y, k_masked_shifted


# =========================================================================
# Acquisition Error Simulation Functions (Capstone Report Section 4.3.1)
# =========================================================================

def add_gaussian_noise(
    kspace: Union[np.ndarray, torch.Tensor],
    snr_db: Optional[float] = None,
    noise_std: Optional[float] = None,
    seed: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Simulates complex Gaussian thermal measurement noise (Equation 4.3):
        epsilon = epsilon_r + j * epsilon_i
    where epsilon_r, epsilon_i are independent zero-mean Gaussian variables.
    """
    if isinstance(kspace, torch.Tensor):
        generator = torch.Generator(device=kspace.device).manual_seed(seed) if seed is not None else None
        signal_power = torch.mean(torch.abs(kspace) ** 2)

        if snr_db is not None:
            snr_linear = 10.0 ** (snr_db / 10.0)
            target_noise_power = signal_power / snr_linear
            std_per_channel = torch.sqrt(target_noise_power / 2.0).item()
        elif noise_std is not None:
            std_per_channel = noise_std
        else:
            return kspace

        noise_r = torch.randn(kspace.shape, generator=generator, device=kspace.device, dtype=kspace.real.dtype) * std_per_channel
        noise_i = torch.randn(kspace.shape, generator=generator, device=kspace.device, dtype=kspace.real.dtype) * std_per_channel
        return kspace + torch.complex(noise_r, noise_i)
    else:
        rng = np.random.default_rng(seed)
        signal_power = np.mean(np.abs(kspace) ** 2)

        if snr_db is not None:
            snr_linear = 10.0 ** (snr_db / 10.0)
            target_noise_power = signal_power / snr_linear
            std_per_channel = np.sqrt(target_noise_power / 2.0)
        elif noise_std is not None:
            std_per_channel = noise_std
        else:
            return kspace

        noise_r = rng.normal(0, std_per_channel, size=kspace.shape)
        noise_i = rng.normal(0, std_per_channel, size=kspace.shape)
        return kspace + (noise_r + 1j * noise_i).astype(kspace.dtype)


def add_phase_errors(
    kspace: Union[np.ndarray, torch.Tensor],
    max_phase_rad: float = 0.2,
    seed: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Simulates phase perturbations into complex measurements (Equation 4.4):
        y_phase = y * exp(j * phi)
    where phi represents smooth background phase variation (e.g. from B0 inhomogeneity).
    """
    H, W = kspace.shape[-2], kspace.shape[-1]
    if isinstance(kspace, torch.Tensor):
        generator = torch.Generator(device=kspace.device).manual_seed(seed) if seed is not None else None
        # Generate low-frequency 2D phase map using smooth polynomial/trigonometric variation
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=kspace.device),
            torch.linspace(-1, 1, W, device=kspace.device),
            indexing="ij"
        )
        rand_coeff = torch.randn(3, generator=generator, device=kspace.device)
        phi = max_phase_rad * (rand_coeff[0] * x_grid + rand_coeff[1] * y_grid + rand_coeff[2] * (x_grid ** 2 + y_grid ** 2))
        phase_factor = torch.exp(1j * phi.to(dtype=kspace.real.dtype))
        return kspace * phase_factor
    else:
        rng = np.random.default_rng(seed)
        y_grid, x_grid = np.meshgrid(
            np.linspace(-1, 1, H),
            np.linspace(-1, 1, W),
            indexing="ij"
        )
        rand_coeff = rng.normal(0, 1, size=3)
        phi = max_phase_rad * (rand_coeff[0] * x_grid + rand_coeff[1] * y_grid + rand_coeff[2] * (x_grid ** 2 + y_grid ** 2))
        phase_factor = np.exp(1j * phi).astype(kspace.dtype)
        return kspace * phase_factor


def add_motion_artifacts(
    kspace: Union[np.ndarray, torch.Tensor],
    affected_lines_fraction: float = 0.08,
    max_phase_shift: float = np.pi / 2,
    seed: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Simulates patient motion artifacts by introducing sudden phase steps and inconsistencies
    across randomly selected k-space phase-encode lines during acquisition.
    """
    n_lines = kspace.shape[-2]
    num_affected = max(1, int(round(n_lines * affected_lines_fraction)))

    if isinstance(kspace, torch.Tensor):
        generator = torch.Generator(device=kspace.device).manual_seed(seed) if seed is not None else None
        affected_idx = torch.randperm(n_lines, generator=generator)[:num_affected]
        phase_shifts = (torch.rand(num_affected, generator=generator, device=kspace.device) * 2 - 1) * max_phase_shift

        kspace_corrupted = kspace.clone()
        for idx, shift in zip(affected_idx, phase_shifts):
            kspace_corrupted[..., idx, :] = kspace_corrupted[..., idx, :] * torch.exp(1j * shift)
        return kspace_corrupted
    else:
        rng = np.random.default_rng(seed)
        affected_idx = rng.choice(n_lines, size=num_affected, replace=False)
        phase_shifts = rng.uniform(-max_phase_shift, max_phase_shift, size=num_affected)

        kspace_corrupted = kspace.copy()
        for idx, shift in zip(affected_idx, phase_shifts):
            kspace_corrupted[..., idx, :] = kspace_corrupted[..., idx, :] * np.exp(1j * shift)
        return kspace_corrupted

