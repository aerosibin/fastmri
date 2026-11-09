"""
test_slice.py
Interactive & single-slice testing tool for Residual Complex U-Net.
Allows you to test:
  1. Any specific subject and slice index from the dataset:
     python test_slice.py --subject e14155s3_P69120 --slice 85
  2. Any external custom file (.npy array of shape (N, 256, 256, 2) or (256, 256, 2)):
     python test_slice.py --file path/to/custom_file.npy --slice 0
  3. Interactive GUI window to scrub through slices and test U-Net reconstructions live:
     python test_slice.py --subject e14155s3_P69120 --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    generate_undersampling_mask,
    apply_undersampling,
    complex_to_chan2,
    ifft2_centered,
    normalize_slice,
    unnormalize_slice,
    complex_magnitude
)
from src.models import ResidualComplexUNet, DataConsistency, UnrolledPhysicsNet
from src.metrics import compute_psnr_np, compute_ssim_np, compute_nmse_np
from src.utils import get_device


def load_custom_kspace(
    file_path: Union[str, Path],
    slice_idx: int = 0
) -> Tuple[torch.Tensor, int]:
    """
    Loads k-space from a .npy file.
    Supports shapes:
      - (N_slices, 256, 256, 2)
      - (256, 256, 2)
      - (N_slices, 256, 256) [complex]
      - (256, 256) [complex]
    Returns:
      kspace_slice: Complex tensor of shape (256, 256)
      total_slices: Total slices in file
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    raw = np.load(path, mmap_mode="r")

    if raw.ndim == 4:
        # (N, 256, 256, 2)
        total_slices = raw.shape[0]
        slice_idx = max(0, min(slice_idx, total_slices - 1))
        real = torch.from_numpy(raw[slice_idx, ..., 0].astype(np.float32))
        imag = torch.from_numpy(raw[slice_idx, ..., 1].astype(np.float32))
        kspace_slice = torch.complex(real, imag)
    elif raw.ndim == 3 and raw.shape[-1] == 2:
        # (256, 256, 2)
        total_slices = 1
        real = torch.from_numpy(raw[..., 0].astype(np.float32))
        imag = torch.from_numpy(raw[..., 1].astype(np.float32))
        kspace_slice = torch.complex(real, imag)
    elif raw.ndim == 3:
        # (N, 256, 256) complex
        total_slices = raw.shape[0]
        slice_idx = max(0, min(slice_idx, total_slices - 1))
        kspace_slice = torch.from_numpy(raw[slice_idx].astype(np.complex64))
    elif raw.ndim == 2:
        # (256, 256) complex
        total_slices = 1
        kspace_slice = torch.from_numpy(raw.astype(np.complex64))
    else:
        raise ValueError(f"Unsupported array shape {raw.shape}. Expected 2D or 3D/4D MRI slice volume.")

    return kspace_slice, total_slices


def reconstruct_single_slice(
    kspace_slice: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    acceleration: float = 4.0,
    center_fraction: float = 0.08,
    mask_type: str = "gaussian",
    seed: Optional[int] = 42,
    use_dc: bool = False
) -> dict:
    """Runs full pipeline on a single complex k-space slice."""
    model.eval()

    # 1. Generate mask
    mask_np = generate_undersampling_mask(
        shape=(256, 256),
        acceleration=acceleration,
        center_fraction=center_fraction,
        mask_type=mask_type,
        seed=seed
    )
    mask = torch.from_numpy(mask_np).to(dtype=torch.float32)

    # 2. Retrospectively undersample
    y, _ = apply_undersampling(kspace_slice, mask)

    # 3. Reconstruct Zero-Filled and Ground Truth
    x_zf_complex = ifft2_centered(y)
    x_gt_complex = ifft2_centered(kspace_slice)

    x_zf_chan2 = complex_to_chan2(x_zf_complex)  # (2, 256, 256)
    x_gt_chan2 = complex_to_chan2(x_gt_complex)  # (2, 256, 256)

    # 4. Normalize
    x_zf_norm, scale = normalize_slice(x_zf_chan2)
    x_gt_norm, _ = normalize_slice(x_gt_chan2, scale=scale)

    # 5. Model Inference (Residual Complex U-Net or Unrolled PhysicsNet)
    x_input = x_zf_norm.unsqueeze(0).to(device)  # (1, 2, 256, 256)
    mask_tensor = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        if isinstance(model, UnrolledPhysicsNet):
            x_hat_norm = model(x_input, mask=mask_tensor)
        else:
            x_hat_norm = model(x_input)
            if use_dc:
                dc = DataConsistency().to(device)
                x_hat_norm = dc(x_hat_norm, x_input, mask_tensor)

    # 6. Unnormalize to physical intensity
    scale_val = scale.item()
    x_zf_unnorm = unnormalize_slice(x_zf_norm, scale_val)
    x_gt_unnorm = unnormalize_slice(x_gt_norm, scale_val)
    x_hat_unnorm = unnormalize_slice(x_hat_norm.squeeze(0).cpu(), scale_val)

    # 7. Compute magnitude images
    mag_zf = complex_magnitude(x_zf_unnorm).squeeze().numpy()
    mag_gt = complex_magnitude(x_gt_unnorm).squeeze().numpy()
    mag_hat = complex_magnitude(x_hat_unnorm).squeeze().numpy()

    dr = float(mag_gt.max() - mag_gt.min())
    if dr <= 0:
        dr = 1.0

    # 8. Compute metrics
    metrics_zf = {
        "psnr": compute_psnr_np(mag_gt, mag_zf, data_range=dr),
        "ssim": compute_ssim_np(mag_gt, mag_zf, data_range=dr),
        "nmse": compute_nmse_np(mag_gt, mag_zf)
    }
    metrics_hat = {
        "psnr": compute_psnr_np(mag_gt, mag_hat, data_range=dr),
        "ssim": compute_ssim_np(mag_gt, mag_hat, data_range=dr),
        "nmse": compute_nmse_np(mag_gt, mag_hat)
    }

    return {
        "ground_truth": mag_gt,
        "zero_filled": mag_zf,
        "reconstruction": mag_hat,
        "mask": mask_np,
        "metrics_zf": metrics_zf,
        "metrics_hat": metrics_hat
    }


def save_comparison_plot(
    results: dict,
    title: str,
    output_path: Path
):
    """Saves a 4-panel presentation-quality comparison figure."""
    gt = results["ground_truth"]
    zf = results["zero_filled"]
    hat = results["reconstruction"]
    m_zf = results["metrics_zf"]
    m_hat = results["metrics_hat"]

    err_zf = np.abs(gt - zf)
    err_hat = np.abs(gt - hat)

    vmax_img = np.percentile(gt, 99.8)
    vmax_err = vmax_img * 0.35

    fig, axs = plt.subplots(1, 4, figsize=(18, 5.0), dpi=300, facecolor="#0b0f19")
    fig.suptitle(title, color="white", weight="bold", fontsize=13, y=0.97)

    panels = [
        (axs[0], gt, "Ground Truth (x)\nFully Sampled Reference", "gray", 0, vmax_img),
        (axs[1], zf, f"Zero-Filled Input (x_ZF)\nPSNR: {m_zf['psnr']:.2f} dB | SSIM: {m_zf['ssim']:.4f}", "gray", 0, vmax_img),
        (axs[2], hat, f"Residual U-Net (x_hat)\nPSNR: {m_hat['psnr']:.2f} dB | SSIM: {m_hat['ssim']:.4f}", "gray", 0, vmax_img),
        (axs[3], err_hat, f"Absolute Error |GT - U-Net|\nNMSE: {m_hat['nmse']:.4f}", "inferno", 0, vmax_err)
    ]

    for ax, img, panel_title, cmap, vmin, vmax in panels:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title, color="#e2e8f0", pad=12, fontsize=11, weight="semibold")
        ax.axis("off")

    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.03, right=0.97, wspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved Figure] -> {output_path.resolve()}")


def launch_slice_gui(
    file_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    initial_slice: int = 85,
    initial_accel: float = 4.0
):
    """Launches an interactive GUI to scrub through slices and test reconstructions in real time."""
    raw = np.load(file_path, mmap_mode="r")
    total_slices = raw.shape[0] if raw.ndim in [3, 4] else 1
    current_slice = min(initial_slice, total_slices - 1)
    current_accel = initial_accel

    fig = plt.figure(figsize=(16, 7.5), facecolor="#0b0f19")
    fig.canvas.manager.set_window_title(f"Residual U-Net Slice Tester - {file_path.name}")

    ax_gt = fig.add_axes([0.05, 0.28, 0.28, 0.60])
    ax_zf = fig.add_axes([0.37, 0.28, 0.28, 0.60])
    ax_hat = fig.add_axes([0.69, 0.28, 0.28, 0.60])
    for ax in [ax_gt, ax_zf, ax_hat]:
        ax.axis("off")

    title_text = fig.suptitle("", color="white", weight="bold", fontsize=13, y=0.96)

    # Slider
    ax_slider = fig.add_axes([0.15, 0.12, 0.70, 0.04], facecolor="#1e293b")
    slider = Slider(ax_slider, "Slice Index", 0, total_slices - 1, valinit=current_slice, valstep=1, valfmt="%d", color="#38bdf8")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    # Radio buttons for R
    ax_radio = fig.add_axes([0.05, 0.02, 0.18, 0.08], facecolor="#1e293b")
    radio = RadioButtons(ax_radio, ("R=2x", "R=4x", "R=8x"), active=1, activecolor="#38bdf8")
    for lbl in radio.labels:
        lbl.set_color("white")

    def update_view(val=None):
        nonlocal current_slice, current_accel
        s_idx = int(slider.val)
        accel_map = {"R=2x": 2.0, "R=4x": 4.0, "R=8x": 8.0}
        current_accel = accel_map[radio.value_selected]

        k_slice, _ = load_custom_kspace(file_path, slice_idx=s_idx)
        res = reconstruct_single_slice(k_slice, model, device, acceleration=current_accel)

        gt = res["ground_truth"]
        zf = res["zero_filled"]
        hat = res["reconstruction"]
        m_zf = res["metrics_zf"]
        m_hat = res["metrics_hat"]

        vmax = np.percentile(gt, 99.8)

        ax_gt.clear()
        ax_zf.clear()
        ax_hat.clear()
        for ax in [ax_gt, ax_zf, ax_hat]:
            ax.axis("off")

        ax_gt.imshow(gt, cmap="gray", vmin=0, vmax=vmax)
        ax_gt.set_title("Ground Truth\nFully Sampled", color="#f8fafc", pad=8, weight="bold")

        ax_zf.imshow(zf, cmap="gray", vmin=0, vmax=vmax)
        ax_zf.set_title(f"Zero-Filled (R={current_accel:.0f}x)\nPSNR: {m_zf['psnr']:.2f} dB | SSIM: {m_zf['ssim']:.3f}", color="#f43f5e", pad=8, weight="bold")

        ax_hat.imshow(hat, cmap="gray", vmin=0, vmax=vmax)
        gain = m_hat['psnr'] - m_zf['psnr']
        ax_hat.set_title(f"Residual Complex U-Net\nPSNR: {m_hat['psnr']:.2f} dB ({gain:+4.2f} dB) | SSIM: {m_hat['ssim']:.3f}", color="#38bdf8", pad=8, weight="bold")

        title_text.set_text(f"File: {file_path.name}  |  Slice {s_idx}/{total_slices-1}  |  Acceleration: R={current_accel:.0f}x")
        fig.canvas.draw_idle()

    slider.on_changed(update_view)
    radio.on_clicked(update_view)
    update_view()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test Residual Complex U-Net on custom or specific MRI slices")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--file", type=str, default=None, help="Path to custom .npy file")
    parser.add_argument("--subject", type=str, default="e14155s3_P69120", help="Subject ID from dataset if --file is not given")
    parser.add_argument("--slice", type=int, default=85, help="Slice index to test (default: 85)")
    parser.add_argument("--acceleration", type=float, default=4.0, help="Acceleration factor R (default: 4.0)")
    parser.add_argument("--use_dc", action="store_true", help="Apply Data Consistency layer")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive GUI slider")
    parser.add_argument("--output", type=str, default="reports/custom_tests/test_slice.png", help="Output path for figure")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    args = parser.parse_args()

    device = get_device(args.device)

    # 1. Locate file
    if args.file:
        target_path = Path(args.file)
    else:
        # Search for subject in CC359
        data_root = Path("CC359/Raw-data/Single-channel")
        matches = list(data_root.glob(f"**/*{args.subject}*.npy"))
        if not matches:
            print(f"[Error] Could not find subject matching '{args.subject}' under {data_root.resolve()}")
            sys.exit(1)
        target_path = matches[0]

    print("=" * 70)
    print(" Accelerated MRI: Custom Slice Testing")
    print("=" * 70)
    print(f" Target File  : {target_path}")
    print(f" Slice Index  : {args.slice}")
    print(f" Acceleration : R={args.acceleration:.1f}x")

    # 2. Load trained model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path.resolve()}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("args", {}).get("model", "unet")
    if model_type == "unrolled":
        num_cascades = ckpt.get("args", {}).get("num_cascades", 5)
        model = UnrolledPhysicsNet(
            num_cascades=num_cascades,
            hidden_channels=48,
            num_layers=5,
            share_weights=True
        ).to(device)
    else:
        base_channels = ckpt.get("args", {}).get("base_channels", 32)
        model = ResidualComplexUNet(in_channels=2, out_channels=2, base_channels=base_channels).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f" Loaded Model : {ckpt_path.name} (type: {model_type}, trained {ckpt.get('epoch', '?')} epochs)")

    if args.interactive:
        print("\nLaunching Interactive Slice Explorer...")
        launch_slice_gui(target_path, model, device, initial_slice=args.slice, initial_accel=args.acceleration)
        return

    # 3. Reconstruct single slice
    k_slice, total_slices = load_custom_kspace(target_path, slice_idx=args.slice)
    print(f" Total Slices in Volume: {total_slices}")

    res = reconstruct_single_slice(
        k_slice, model, device,
        acceleration=args.acceleration,
        use_dc=args.use_dc
    )

    m_zf = res["metrics_zf"]
    m_hat = res["metrics_hat"]
    delta_p = m_hat["psnr"] - m_zf["psnr"]
    delta_s = m_hat["ssim"] - m_zf["ssim"]

    print("\n" + "-" * 70)
    print(f" {'Metric':<18} | {'Zero-Filled (x_ZF)':<20} | {'Residual U-Net (x_hat)':<22} | {'Gain':<10}")
    print("-" * 70)
    print(f" {'PSNR (dB)':<18} | {m_zf['psnr']:<20.2f} | {m_hat['psnr']:<22.2f} | {delta_p:+6.2f} dB")
    print(f" {'SSIM':<18} | {m_zf['ssim']:<20.4f} | {m_hat['ssim']:<22.4f} | {delta_s:+6.4f}")
    print(f" {'NMSE':<18} | {m_zf['nmse']:<20.6f} | {m_hat['nmse']:<22.6f} | {m_hat['nmse'] - m_zf['nmse']:+6.6f}")
    print("-" * 70)

    # 4. Save comparison figure
    title = f"Accelerated MRI (R={args.acceleration:.0f}x) - {target_path.name} [Slice {args.slice}]"
    save_comparison_plot(res, title, Path(args.output))


if __name__ == "__main__":
    main()

