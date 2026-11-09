"""
evaluate.py
Validation and Evaluation Pipeline for Accelerated MRI Reconstruction Models on Calgary-Campinas Dataset.
Conforms directly to Capstone Report Section 4.5:
  - Benchmarks Zero-Filled Baseline vs. Trained Neural Reconstruction (Residual Complex U-Net / Unrolled Physics Net).
  - Computes Peak Signal-to-Noise Ratio (PSNR) (Equation 4.11), SSIM, and NMSE across validation subjects.
  - Verifies target PSNR of 32-34 dB for R=4.
  - Saves visual comparison figures (Ground Truth, Zero-Filled, Reconstructed, Error Maps).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CC359Dataset,
    complex_magnitude,
    unnormalize_slice
)
from src.models import ResidualComplexUNet, DataConsistency, UnrolledPhysicsNet
from src.metrics import compute_psnr_np, compute_ssim_np, compute_nmse_np, compute_mae_np
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Accelerated MRI Reconstruction Models on Calgary-Campinas MRI Dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_root", type=str, default="CC359/Raw-data/Single-channel",
                        help="Path to dataset root folder")
    parser.add_argument("--acceleration", type=float, default=4.0,
                        help="Acceleration factor R (4.0 primary, 8.0 stress test)")
    parser.add_argument("--center_fraction", type=float, default=0.08,
                        help="ACS center fraction (default: 0.08)")
    parser.add_argument("--use_dc", action="store_true",
                        help="Apply Data Consistency layer after U-Net prediction")
    parser.add_argument("--output_dir", type=str, default="reports/evaluation_results",
                        help="Directory to save evaluation reports and figures")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run inference on ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--num_visualizations", type=int, default=5,
                        help="Number of slice comparison figures to save")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of validation slices to evaluate (default: all)")
    return parser.parse_args()


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset: CC359Dataset,
    device: torch.device,
    use_dc: bool = False,
    output_dir: Optional[Path] = None,
    num_visualizations: int = 5,
    model_label: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    dc_layer = DataConsistency().to(device) if use_dc else None

    if model_label is None:
        if isinstance(model, UnrolledPhysicsNet):
            model_label = f"Unrolled Physics Net (T={getattr(model, 'num_cascades', 5)})"
        elif use_dc:
            model_label = "Residual Complex U-Net + DC"
        else:
            model_label = "Residual Complex U-Net"

    zf_psnr_list, zf_ssim_list, zf_nmse_list = [], [], []
    unet_psnr_list, unet_ssim_list, unet_nmse_list = [], [], []

    saved_figures = 0
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    num_eval = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    print(f"\nEvaluating {num_eval} validation slices on {device}...")
    for idx in tqdm(range(num_eval), desc="Inference"):
        sample = dataset[idx]

        x_input = sample["input"].unsqueeze(0).to(device)        # (1, 2, 256, 256)
        x_gt = sample["ground_truth"].unsqueeze(0).to(device)   # (1, 2, 256, 256)
        scale = sample["scale"].item()
        sub_id = sample["subject_id"]
        slice_idx = sample["slice_idx"]

        mask = sample["mask"].unsqueeze(0).to(device)
        if isinstance(model, UnrolledPhysicsNet):
            x_hat = model(x_input, mask=mask)
        else:
            x_hat = model(x_input)
            if dc_layer is not None:
                x_hat = dc_layer(x_hat, x_input, mask)

        # Unnormalize to physical intensity
        x_input_unnorm = unnormalize_slice(x_input, scale)
        x_hat_unnorm = unnormalize_slice(x_hat, scale)
        x_gt_unnorm = unnormalize_slice(x_gt, scale)

        # Compute magnitude images
        mag_zf = complex_magnitude(x_input_unnorm).squeeze().cpu().numpy()
        mag_unet = complex_magnitude(x_hat_unnorm).squeeze().cpu().numpy()
        mag_gt = complex_magnitude(x_gt_unnorm).squeeze().cpu().numpy()

        dr = float(mag_gt.max() - mag_gt.min())
        if dr <= 0:
            dr = 1.0

        # Zero-Filled Baseline Metrics
        zf_p = compute_psnr_np(mag_gt, mag_zf, data_range=dr)
        zf_s = compute_ssim_np(mag_gt, mag_zf, data_range=dr)
        zf_n = compute_nmse_np(mag_gt, mag_zf)

        # Reconstruction Model Metrics
        unet_p = compute_psnr_np(mag_gt, mag_unet, data_range=dr)
        unet_s = compute_ssim_np(mag_gt, mag_unet, data_range=dr)
        unet_n = compute_nmse_np(mag_gt, mag_unet)

        zf_psnr_list.append(zf_p)
        zf_ssim_list.append(zf_s)
        zf_nmse_list.append(zf_n)

        unet_psnr_list.append(unet_p)
        unet_ssim_list.append(unet_s)
        unet_nmse_list.append(unet_n)

        # Save comparison figures for key representative slices
        save_this_slice = (slice_idx in [70, 80, 85, 90, 100]) if max_samples is None else True
        if output_dir and saved_figures < num_visualizations and save_this_slice:
            fig_path = output_dir / f"recon_{sub_id}_slice{slice_idx:03d}.png"
            err_zf = np.abs(mag_gt - mag_zf)
            err_unet = np.abs(mag_gt - mag_unet)
            vmax_img = np.percentile(mag_gt, 99.8)
            vmax_err = vmax_img * 0.35

            fig, axs = plt.subplots(1, 5, figsize=(20, 4.8), dpi=250, facecolor="#0b0f19")
            fig.suptitle(
                f"Accelerated MRI Reconstruction (R={dataset.acceleration:.0f}x) - Subject: {sub_id}, Slice {slice_idx}",
                color="white", weight="bold", fontsize=13, y=0.97
            )

            panels = [
                (axs[0], mag_gt, "Ground Truth (x)\nReference", "gray", 0, vmax_img),
                (axs[1], mag_zf, f"Zero-Filled (x_ZF)\nPSNR: {zf_p:.2f} dB | SSIM: {zf_s:.4f}", "gray", 0, vmax_img),
                (axs[2], mag_unet, f"{model_label} (x_hat)\nPSNR: {unet_p:.2f} dB | SSIM: {unet_s:.4f}", "gray", 0, vmax_img),
                (axs[3], err_zf, "Zero-Filled Error Map\n|GT - ZF|", "inferno", 0, vmax_err),
                (axs[4], err_unet, f"{model_label} Error Map\n|GT - Recon|", "inferno", 0, vmax_err)
            ]

            for ax, img, title, cmap, vmin, vmax in panels:
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title, color="#e2e8f0", pad=12, fontsize=10, weight="semibold")
                ax.axis("off")

            fig.subplots_adjust(top=0.80, bottom=0.08, left=0.02, right=0.98, wspace=0.12)
            plt.savefig(str(fig_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            saved_figures += 1

    zf_summary = {
        "psnr": float(np.mean(zf_psnr_list)),
        "ssim": float(np.mean(zf_ssim_list)),
        "nmse": float(np.mean(zf_nmse_list))
    }
    unet_summary = {
        "psnr": float(np.mean(unet_psnr_list)),
        "ssim": float(np.mean(unet_ssim_list)),
        "nmse": float(np.mean(unet_nmse_list))
    }
    return zf_summary, unet_summary


def main():
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)

    print("=" * 75)
    print(" Accelerated MRI Reconstruction: Performance Evaluation")
    print(f" Capstone Phase 1 - Review 3 (Dr. S. Vidhusha)")
    print("=" * 75)
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Device     : {device}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f" [Error] Checkpoint not found at: {ckpt_path.resolve()}")
        print(f" Run training first: python train.py --epochs 30")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("args", {}).get("model", "unet")
    use_dc = args.use_dc or ckpt.get("args", {}).get("use_dc", False)

    if model_type == "unrolled":
        num_cascades = ckpt.get("args", {}).get("num_cascades", 5)
        model = UnrolledPhysicsNet(
            num_cascades=num_cascades,
            hidden_channels=48,
            num_layers=5,
            share_weights=True
        ).to(device)
        model_label = f"Unrolled Physics Net (T={num_cascades})"
    else:
        base_channels = ckpt.get("args", {}).get("base_channels", 32)
        model = ResidualComplexUNet(
            in_channels=2,
            out_channels=2,
            base_channels=base_channels,
            bilinear=True
        ).to(device)
        if use_dc:
            model_label = "Residual Complex U-Net + DC"
        else:
            model_label = "Residual Complex U-Net"

    print(f" Detected Architecture: {model_label}")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f" Loaded model trained for {ckpt.get('epoch', '?')} epochs. Best train Val PSNR: {ckpt.get('best_val_psnr', 0):.2f} dB")

    # Load validation dataset
    val_dataset = CC359Dataset(
        data_root=args.data_root,
        split="val",
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        mask_type="gaussian",
        seed=42
    )

    zf_res, unet_res = evaluate_dataset(
        model=model,
        dataset=val_dataset,
        device=device,
        use_dc=use_dc,
        output_dir=output_dir,
        num_visualizations=args.num_visualizations,
        model_label=model_label,
        max_samples=args.max_samples
    )

    delta_psnr = unet_res["psnr"] - zf_res["psnr"]
    delta_ssim = unet_res["ssim"] - zf_res["ssim"]

    col_w = max(len(model_label), 24)
    table_w = max(75, 20 + 3 + 22 + 3 + col_w + 3 + 10)

    print("\n" + "=" * table_w)
    print(f" FINAL BENCHMARK RESULTS (Acceleration R={args.acceleration:.0f}x, 10 Validation Subjects)")
    print("=" * table_w)
    print(f" {'Metric':<20} | {'Zero-Filled (Baseline)':<22} | {model_label:<{col_w}} | {'Gain':<10}")
    print("-" * table_w)
    print(f" {'PSNR (dB)':<20} | {zf_res['psnr']:<22.2f} | {unet_res['psnr']:<{col_w}.2f} | {delta_psnr:+6.2f} dB")
    print(f" {'SSIM':<20} | {zf_res['ssim']:<22.4f} | {unet_res['ssim']:<{col_w}.4f} | {delta_ssim:+6.4f}")
    print(f" {'NMSE':<20} | {zf_res['nmse']:<22.6f} | {unet_res['nmse']:<{col_w}.6f} | {unet_res['nmse'] - zf_res['nmse']:+6.6f}")
    print("=" * table_w)
    print(f" Visual results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
