"""
src/data/dataset.py
PyTorch Dataset for Calgary-Campinas Brain MR Dataset (CC359).
Conforms directly to Capstone Report Chapter 4:
  - 4.2.1: Data Ingestion and Preprocessing (25 train subjects, 10 val subjects, ~9.65 GB).
           Uses numpy.memmap for zero-RAM bloat slice streaming.
  - 4.2.2: Physics Simulation and Undersampling (R=4 primary, R=8 stress test).
  - 4.3.2: Zero-Filled Reconstruction & Real/Imag channel separation (shape: B x 2 x 256 x 256).
  - 4.4:   Supervised sample formulation: Input (X_input), Ground Truth (X_gt), Target (Delta_x).
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from .mask import (
    generate_undersampling_mask,
    apply_undersampling,
    add_gaussian_noise,
    add_phase_errors,
    add_motion_artifacts
)
from .transforms import (
    complex_to_chan2,
    ifft2_centered,
    normalize_slice
)


class CC359Dataset(Dataset):
    """
    High-performance PyTorch Dataset for single-channel complex k-space MRI data.
    Indexes slices across all subjects and streams them via numpy.memmap.
    """
    def __init__(
        self,
        data_root: Union[str, Path] = "CC359/Raw-data/Single-channel",
        split: str = "train",
        acceleration: float = 4.0,
        center_fraction: float = 0.08,
        mask_type: str = "gaussian",
        slice_range: Optional[Tuple[int, int]] = None,
        simulate_noise: bool = False,
        noise_snr_db: Optional[float] = 30.0,
        simulate_phase_error: bool = False,
        simulate_motion: bool = False,
        seed: Optional[int] = 42
    ):
        """
        Args:
            data_root: Root path to single-channel directory.
            split: 'train' or 'val'.
            acceleration: Retrospective acceleration factor R (e.g. 4.0 or 8.0).
            center_fraction: Fraction of low-frequency lines kept in ACS region (default 0.08).
            mask_type: 'gaussian', 'random', or 'equispaced'.
            slice_range: Optional tuple (start_slice, end_slice) to exclude empty air slices (e.g. (15, 155)).
                         If None, uses all available slices.
            simulate_noise: Whether to add complex Gaussian noise (Section 4.3.1).
            noise_snr_db: Target SNR in dB if noise is added.
            simulate_phase_error: Whether to add smooth phase perturbations.
            simulate_motion: Whether to add simulated motion phase inconsistency.
            seed: Random seed for deterministic mask/noise in validation.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        if self.split not in ["train", "training", "val", "validation"]:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'val'.")
        self.is_train = self.split in ["train", "training"]

        self.acceleration = float(acceleration)
        self.center_fraction = float(center_fraction)
        self.mask_type = mask_type
        self.slice_range = slice_range
        self.simulate_noise = simulate_noise
        self.noise_snr_db = noise_snr_db
        self.simulate_phase_error = simulate_phase_error
        self.simulate_motion = simulate_motion
        self.seed = seed

        # Discover subjects belonging to this split
        self.subjects = self._discover_subjects()
        if not self.subjects:
            raise RuntimeError(f"No subjects found for split '{self.split}' under {self.data_root.resolve()}")

        # Build index of (subject_idx, slice_idx)
        self.slice_index: List[Tuple[int, int]] = []
        self._build_slice_index()

        # Cache for opened memmap arrays (opened lazily per worker process)
        self._memmap_cache: Dict[int, np.ndarray] = {}

    def _discover_subjects(self) -> List[Dict[str, Union[str, Path]]]:
        """Discovers subject .npy files for the requested split."""
        all_npy = sorted(self.data_root.glob("**/*.npy"))
        subjects = []
        for p in all_npy:
            rel_parts = p.relative_to(self.data_root).parts
            group = rel_parts[0]  # Train_part1, Train_part2, Train_part3, Val
            is_val_file = "val" in group.lower()

            if (self.is_train and not is_val_file) or (not self.is_train and is_val_file):
                sub_id = p.stem.split(".")[0]
                subjects.append({
                    "subject_id": sub_id,
                    "filepath": p,
                    "group": group
                })
        return subjects

    def _build_slice_index(self):
        """Indexes all slices across subjects."""
        for sub_idx, sub in enumerate(self.subjects):
            # Read shape without reading data into RAM
            m = np.load(sub["filepath"], mmap_mode="r")
            n_slices = m.shape[0]

            start = 0 if self.slice_range is None else max(0, self.slice_range[0])
            end = n_slices if self.slice_range is None else min(n_slices, self.slice_range[1])

            for s_idx in range(start, end):
                self.slice_index.append((sub_idx, s_idx))

    def _get_memmap(self, sub_idx: int) -> np.ndarray:
        """Lazily retrieves or opens the memory-mapped numpy array."""
        if sub_idx not in self._memmap_cache:
            filepath = self.subjects[sub_idx]["filepath"]
            self._memmap_cache[sub_idx] = np.load(filepath, mmap_mode="r")
        return self._memmap_cache[sub_idx]

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int, float]]:
        sub_idx, slice_idx = self.slice_index[idx]
        sub_info = self.subjects[sub_idx]
        mmap_arr = self._get_memmap(sub_idx)

        # 1. Zero-copy extract raw slice: shape (256, 256, 2)
        raw_slice = mmap_arr[slice_idx]  # float64
        # Convert to complex64 PyTorch tensor
        real = torch.from_numpy(raw_slice[..., 0].astype(np.float32))
        imag = torch.from_numpy(raw_slice[..., 1].astype(np.float32))
        kspace_full = torch.complex(real, imag)  # Shape: (256, 256)

        # 2. Retrospective Mask Generation (Chapter 4.2.2)
        # For training, if seed is not fixed, vary seed per slice for diverse masks
        mask_seed = self.seed + idx if (not self.is_train and self.seed is not None) else None
        mask_np = generate_undersampling_mask(
            shape=(256, 256),
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            mask_type=self.mask_type,
            seed=mask_seed
        )
        mask = torch.from_numpy(mask_np).to(dtype=torch.float32)

        # 3. Retrospective Undersampling & Error Simulation (Section 4.3.1)
        y, _ = apply_undersampling(kspace_full, mask)

        if self.simulate_noise:
            y = add_gaussian_noise(y, snr_db=self.noise_snr_db, seed=mask_seed)
        if self.simulate_phase_error:
            y = add_phase_errors(y, max_phase_rad=0.2, seed=mask_seed)
        if self.simulate_motion:
            y = add_motion_artifacts(y, affected_lines_fraction=0.08, seed=mask_seed)

        # 4. Zero-Filled Reconstruction (Section 4.3.2)
        # In CC359 unshifted k-space, ifft2 directly reconstructs spatial domain image
        x_zf_complex = ifft2_centered(y)
        x_gt_complex = ifft2_centered(kspace_full)

        # 5. Channel Separation and Stacking: [Re, Im]^T in R^{2 x 256 x 256}
        x_zf_chan2 = complex_to_chan2(x_zf_complex)  # (2, 256, 256)
        x_gt_chan2 = complex_to_chan2(x_gt_complex)  # (2, 256, 256)

        # 6. Intensity Normalization
        # Normalize by std(x_zf_chan2) so inputs are standard normal scale O(1)
        x_zf_norm, scale = normalize_slice(x_zf_chan2)
        x_gt_norm, _ = normalize_slice(x_gt_chan2, scale=scale)

        # 7. Residual Target (Equation 4.8): Delta_x = x_gt - x_zf
        delta_x_norm = x_gt_norm - x_zf_norm

        return {
            "input": x_zf_norm,            # Shape: (2, 256, 256) -> X_input (Re, Im)
            "target": delta_x_norm,        # Shape: (2, 256, 256) -> Delta_x (residual target)
            "ground_truth": x_gt_norm,     # Shape: (2, 256, 256) -> X_gt (normalized reference)
            "mask": mask.unsqueeze(0),     # Shape: (1, 256, 256) -> binary mask
            "scale": scale.squeeze(),      # Scalar tensor -> for unnormalization
            "subject_id": sub_info["subject_id"],
            "slice_idx": slice_idx
        }

