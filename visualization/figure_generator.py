"""
figure_generator.py
Generates presentation-ready, high-resolution figures for mentor explanations and reports.

Figures generated:
  1. 01_kspace_physics_pipeline.png: Raw k-space components -> IFFT spatial reconstruction
  2. 02_accelerated_mri_problem.png: Acceleration factors (R=1, 2, 4, 8) with masks, aliased zero-filled images, error maps, and PSNR/SSIM metrics
  3. 03_orthogonal_brain_views.png: 3D Multi-Planar Reformatting (Sagittal, Coronal, Axial views)
  4. 04_multislice_montage.png: Whole-brain progression across slices
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .mri_processor import (
    load_subject_kspace,
    reconstruct_slice,
    reconstruct_volume,
    get_orthogonal_views,
    generate_undersampling_mask,
    simulate_undersampling,
    compute_reconstruction_metrics
)


def safe_savefig(fig, output_path: Path, **kwargs):
    """Safely saves a figure to disk on Windows, avoiding momentary file locks."""
    import time
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try direct save first with retries
    for attempt in range(5):
        try:
            fig.savefig(str(output_path), **kwargs)
            return output_path
        except (OSError, PermissionError):
            time.sleep(0.4)
            
    # Fallback to atomic temp file rename
    temp_path = output_path.with_name(f"_tmp_{output_path.name}")
    fig.savefig(str(temp_path), **kwargs)
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass
    temp_path.replace(output_path)
    return output_path


def set_presentation_style():
    """Applies modern clean formatting to Matplotlib plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.titlesize": 15,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "image.interpolation": "nearest"
    })


def generate_physics_pipeline_figure(
    kspace_slice: np.ndarray,
    subject_id: str,
    slice_idx: int,
    output_path: Path
) -> Path:
    """
    Figure 1: Explains MRI physics from raw data to spatial domain.
    Panels: Real, Imaginary, Log-Magnitude k-Space, Reconstructed Magnitude Image.
    """
    set_presentation_style()

    real_part = np.real(kspace_slice)
    imag_part = np.imag(kspace_slice)
    k_shifted = np.fft.fftshift(kspace_slice)
    log_mag = np.log1p(np.abs(k_shifted))
    recon_img = reconstruct_slice(kspace_slice)

    fig = plt.figure(figsize=(16, 5.2), dpi=300, facecolor="#0f1117")
    fig.suptitle(
        f"Calgary-Campinas MR Dataset: Raw k-Space to Spatial Image Pipeline (Subject: {subject_id}, Slice {slice_idx})",
        color="white",
        weight="bold",
        fontsize=13,
        y=0.97
    )

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.15, top=0.82, bottom=0.08, left=0.03, right=0.97)

    panels = [
        ("Real Component\nk-Space (kx, ky)", np.fft.fftshift(real_part), "coolwarm", True),
        ("Imaginary Component\nk-Space (kx, ky)", np.fft.fftshift(imag_part), "coolwarm", True),
        ("Log-Magnitude Spectrum\nlog(1 + |k|)", log_mag, "magma", False),
        ("Spatial Domain Image\n2D Inverse FFT (|IFFT2|)", recon_img, "gray", False),
    ]

    for i, (title, data, cmap, symm) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if symm:
            vmax = np.percentile(np.abs(data), 99)
            vmin = -vmax
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            vmax = np.percentile(data, 99.8)
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)

        ax.set_title(title, color="#e2e8f0", pad=12, weight="semibold", fontsize=11)
        ax.axis("off")

        # Subtle clean border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#334155")

    safe_savefig(fig, output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_accelerated_mri_problem_figure(
    kspace_slice: np.ndarray,
    subject_id: str,
    slice_idx: int,
    output_path: Path
) -> Path:
    """
    Figure 2: The Core Capstone Problem - Accelerated MRI via Undersampling.
    Columns: R=1 (Fully sampled Ground Truth), R=2x, R=4x, R=8x.
    Rows:
      Row 1: Undersampling Masks (1D Gaussian)
      Row 2: Zero-Filled Aliased Reconstructions with PSNR & SSIM
      Row 3: Absolute Error Maps (|GT - ZF| x 5)
    """
    set_presentation_style()

    gt_img = reconstruct_slice(kspace_slice)
    accelerations = [1.0, 2.0, 4.0, 8.0]
    masks = []
    zf_images = []
    metrics_list = []
    error_maps = []

    for r in accelerations:
        if r == 1.0:
            mask = np.ones((256, 256), dtype=np.float32)
            zf = gt_img.copy()
            metrics = {"psnr": float("inf"), "ssim": 1.000}
        else:
            mask = generate_undersampling_mask(
                shape=(256, 256),
                acceleration=r,
                center_fraction=0.08 if r >= 4.0 else 0.16,
                mask_type="gaussian",
                seed=42
            )
            _, zf = simulate_undersampling(kspace_slice, mask)
            metrics = compute_reconstruction_metrics(gt_img, zf)

        error = np.abs(gt_img - zf)
        masks.append(mask)
        zf_images.append(zf)
        metrics_list.append(metrics)
        error_maps.append(error)

    fig = plt.figure(figsize=(15, 11.5), dpi=300, facecolor="#0b0f19")
    fig.suptitle(
        f"The Accelerated MRI Problem: Retrospective Undersampling & Aliasing Artifacts\n(Subject: {subject_id}, Slice: {slice_idx})",
        color="white",
        weight="bold",
        fontsize=13,
        y=0.97
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.32, wspace=0.15, top=0.88, bottom=0.04, left=0.09, right=0.96)

    col_titles = [
        "Fully Sampled (R=1x)\nGround Truth Reference",
        f"Accelerated R=2x\n(Sampled: {masks[1].mean()*100:.1f}%)",
        f"Accelerated R=4x\n(Sampled: {masks[2].mean()*100:.1f}%)",
        f"Accelerated R=8x\n(Sampled: {masks[3].mean()*100:.1f}%)"
    ]

    vmax_img = np.percentile(gt_img, 99.8)
    vmax_err = vmax_img * 0.4  # Sensitive error map threshold

    # Row 1: Masks
    for j, (mask, title) in enumerate(zip(masks, col_titles)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(mask, cmap="Blues_r", vmin=0, vmax=1)
        ax.set_title(title, color="#e2e8f0", pad=10, weight="semibold", fontsize=11)
        ax.axis("off")
        if j == 0:
            ax.text(-0.16, 0.5, "Sampling\nMask (k)", transform=ax.transAxes, color="#38bdf8",
                    va="center", ha="right", weight="bold", fontsize=11)

    # Row 2: Zero-Filled Reconstructions
    for j, (zf, met) in enumerate(zip(zf_images, metrics_list)):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(zf, cmap="gray", vmin=0, vmax=vmax_img)
        ax.axis("off")

        if j == 0:
            sub = "Reference"
            ax.text(-0.16, 0.5, "Zero-Filled\nReconstruction", transform=ax.transAxes, color="#f8fafc",
                    va="center", ha="right", weight="bold", fontsize=11)
        else:
            sub = f"PSNR: {met['psnr']:.2f} dB\nSSIM: {met['ssim']:.4f}"

        # Overlay badge with metrics
        ax.text(
            0.5, 0.05, sub,
            transform=ax.transAxes,
            ha="center", va="bottom",
            color="#38bdf8" if j == 0 else "#f43f5e",
            fontsize=10, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="#334155", alpha=0.9)
        )

    # Row 3: Absolute Error Maps
    for j, err in enumerate(error_maps):
        ax = fig.add_subplot(gs[2, j])
        if j == 0:
            # Blank or perfect 0
            ax.imshow(np.zeros_like(err), cmap="inferno", vmin=0, vmax=vmax_err)
            ax.text(0.5, 0.5, "0.00 Error", transform=ax.transAxes, ha="center", va="center",
                    color="#94a3b8", weight="semibold", fontsize=11)
            ax.text(-0.16, 0.5, "Error Map\n|GT - ZF|", transform=ax.transAxes, color="#fb923c",
                    va="center", ha="right", weight="bold", fontsize=11)
        else:
            im_err = ax.imshow(err, cmap="inferno", vmin=0, vmax=vmax_err)
        ax.axis("off")

    # Add colorbar for error maps
    cbar_ax = fig.add_axes([0.965, 0.06, 0.012, 0.22])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=vmax_err), cmap="inferno"), cax=cbar_ax)
    cb.set_label("Artifact Residual Intensity", color="#94a3b8", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#94a3b8", fontsize=8)

    safe_savefig(fig, output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_orthogonal_views_figure(
    complex_kspace: np.ndarray,
    subject_id: str,
    output_path: Path
) -> Path:
    """
    Figure 3: 3D Multi-Planar Reformatting (Sagittal, Coronal, Axial).
    Demonstrates the full volumetric nature of the CC359 acquisition.
    """
    set_presentation_style()

    volume = reconstruct_volume(complex_kspace)
    sag, cor, axi = get_orthogonal_views(volume)

    fig = plt.figure(figsize=(15, 6.0), dpi=300, facecolor="#090d16")
    fig.suptitle(
        f"3D Multi-Planar Reformatting (MPR) - Whole Brain Volume (Subject: {subject_id})\nAcquisition: 1.0 mm Isotropic T1-Weighted Sagittal 3D Gradient Echo",
        color="white",
        weight="bold",
        fontsize=13,
        y=0.96
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.10, top=0.82, bottom=0.06, left=0.03, right=0.97)

    views = [
        ("Sagittal View (Acquisition Plane)", sag, "Slice 85 / 170 (Mid-Sagittal)", (256, 256)),
        ("Coronal View (Reformatted MPR)", cor, "Slice 128 / 256 (Hippocampal Axis)", (256, 170)),
        ("Axial View (Reformatted MPR)", axi, "Slice 128 / 256 (Lateral Ventricles)", (256, 170))
    ]

    vmax = np.percentile(volume, 99.8)

    for i, (title, img, subtitle, dim) in enumerate(views):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title, color="#e2e8f0", pad=12, weight="bold", fontsize=12)
        ax.text(
            0.5, 0.04, f"{subtitle}\nMatrix: {dim[0]} x {dim[1]}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            color="#38bdf8", fontsize=9, weight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="#334155", alpha=0.85)
        )
        ax.axis("off")

    safe_savefig(fig, output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_multislice_montage_figure(
    complex_kspace: np.ndarray,
    subject_id: str,
    output_path: Path,
    n_slices: int = 12
) -> Path:
    """
    Figure 4: Multi-slice progression across brain anatomy.
    Plots a 3x4 grid of sagittal slices from lateral to medial to contralateral.
    """
    set_presentation_style()

    total_slices = complex_kspace.shape[0]
    # Sample slices evenly between 20% and 80% to capture rich anatomy
    slice_indices = np.linspace(int(total_slices * 0.15), int(total_slices * 0.85), n_slices, dtype=int)

    rows, cols = 3, 4
    fig, axs = plt.subplots(rows, cols, figsize=(14, 10), dpi=300, facecolor="#090d16")
    fig.suptitle(
        f"Whole-Brain Sagittal Slice Series (Subject: {subject_id}, Total: {total_slices} slices)",
        color="white",
        weight="bold",
        fontsize=14,
        y=0.98
    )

    # Global intensity normalization
    sample_img = reconstruct_slice(complex_kspace[total_slices // 2])
    vmax = np.percentile(sample_img, 99.8)

    for idx, (ax, slice_i) in enumerate(zip(axs.flat, slice_indices)):
        img = reconstruct_slice(complex_kspace[slice_i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.text(
            0.05, 0.05, f"Slice #{slice_i}",
            transform=ax.transAxes,
            color="#38bdf8",
            fontsize=9, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#0f172a", ec="#334155", alpha=0.8)
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_savefig(fig, output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return output_path
