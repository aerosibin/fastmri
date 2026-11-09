"""
train.py
Training Pipeline for Residual Complex U-Net on Calgary-Campinas Brain MR Dataset.
Conforms directly to Capstone Report Chapter 4:
  - 4.2: Data Ingestion (Train: 25 subjects, Val: 10 subjects, memory-mapped streaming).
  - 4.3.2: Input: Two-channel Zero-Filled MRI [Re, Im] in R^{B x 2 x 256 x 256}.
  - 4.3.3: Residual Complex U-Net predicting Delta_x.
  - 4.4: Supervised training with AdamW (lr=10^-4), Cosine Annealing scheduler,
         and Composite Loss L_total = alpha * L_L1 + beta * L_SSIM + gamma * L_freq.
  - 4.5: Quantitative validation logging for PSNR (target 32-34 dB for R=4) and SSIM.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CC359Dataset, complex_magnitude, unnormalize_slice
from src.models import ResidualComplexUNet, DataConsistency, UnrolledPhysicsNet
from src.losses import CompositeReconstructionLoss
from src.metrics import evaluate_batch_metrics
from src.utils import set_seed, get_device, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Residual Complex U-Net for Accelerated MRI Reconstruction")

    # Dataset & Physics arguments
    parser.add_argument("--data_root", type=str, default="CC359/Raw-data/Single-channel",
                        help="Path to single-channel root directory")
    parser.add_argument("--acceleration", type=float, default=4.0,
                        help="Acceleration factor R (4.0 primary, 8.0 stress test)")
    parser.add_argument("--center_fraction", type=float, default=0.08,
                        help="ACS center fraction preserved in k-space (default: 0.08)")
    parser.add_argument("--mask_type", type=str, default="gaussian", choices=["gaussian", "random", "equispaced"],
                        help="Retrospective undersampling mask type")

    # Error simulation flags (Section 4.3.1)
    parser.add_argument("--simulate_noise", action="store_true", help="Add simulated Gaussian measurement noise")
    parser.add_argument("--noise_snr_db", type=float, default=30.0, help="Noise SNR in dB")
    parser.add_argument("--simulate_phase_error", action="store_true", help="Add simulated phase perturbations")
    parser.add_argument("--simulate_motion", action="store_true", help="Add simulated motion artifacts")

    # Architecture & Training arguments
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unrolled"],
                        help="Reconstruction model: 'unet' (Residual Complex U-Net) or 'unrolled' (Physics-Informed Deep Cascade / MoDL)")
    parser.add_argument("--use_dc", action="store_true",
                        help="Train with end-to-end physics Data Consistency (DC) layer")
    parser.add_argument("--num_cascades", type=int, default=5,
                        help="Number of unrolled cascade iterations T for physics net (default: 5)")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Number of initial feature channels in U-Net (default: 32)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Mini-batch size (default: 8, optimal for RTX 3050 Laptop GPU)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate for AdamW (default: 1e-4 as in report)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"],
                        help="Learning rate scheduler: 'plateau' (ReduceLROnPlateau) or 'cosine' (CosineAnnealingLR)")
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="Patience (epochs without validation improvement) before decaying LR (default: 3)")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="Multiplicative decay factor for ReduceLROnPlateau (default: 0.5)")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate threshold (default: 1e-6)")

    # Loss weights (Equation 4.10)
    parser.add_argument("--alpha", type=float, default=1.0, help="L1 loss weight alpha")
    parser.add_argument("--beta", type=float, default=0.5, help="SSIM loss weight beta")
    parser.add_argument("--gamma", type=float, default=0.1, help="Frequency k-space loss weight gamma")

    # Compute & Checkpoint
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes (0 on Windows)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quick_run", action="store_true", help="Quick debug mode on a small subset of slices")

    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CompositeReconstructionLoss,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    dc_layer: Optional[DataConsistency] = None
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0
    total_freq_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for batch in pbar:
        x_input = batch["input"].to(device, non_blocking=True)       # (B, 2, 256, 256)
        x_gt = batch["ground_truth"].to(device, non_blocking=True)  # (B, 2, 256, 256)
        mask = batch["mask"].to(device, non_blocking=True)           # (B, 1, 256, 256)

        optimizer.zero_grad(set_to_none=True)

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=device_type, enabled=(use_amp and device.type == "cuda")):
            if isinstance(model, UnrolledPhysicsNet):
                x_hat = model(x_input, mask=mask)
            else:
                x_hat = model(x_input)
                if dc_layer is not None:
                    x_hat = dc_layer(x_hat, x_input, mask)

            loss, loss_dict = criterion(x_hat, x_gt)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss_dict["loss_total"]
        total_l1 += loss_dict["loss_l1"]
        total_ssim_loss += loss_dict["loss_ssim"]
        total_freq_loss += loss_dict["loss_freq"]

        pbar.set_postfix({"loss": f"{loss_dict['loss_total']:.4f}", "ssim": f"{loss_dict['ssim']:.3f}"})

    n = len(loader)
    return {
        "train_loss": total_loss / n,
        "train_l1": total_l1 / n,
        "train_ssim_loss": total_ssim_loss / n,
        "train_freq": total_freq_loss / n
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompositeReconstructionLoss,
    device: torch.device,
    dc_layer: Optional[DataConsistency] = None
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    psnr_list = []
    ssim_list = []
    nmse_list = []

    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch in pbar:
        x_input = batch["input"].to(device, non_blocking=True)
        x_gt = batch["ground_truth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        scale = batch["scale"].to(device, non_blocking=True)

        if isinstance(model, UnrolledPhysicsNet):
            x_hat = model(x_input, mask=mask)
        else:
            x_hat = model(x_input)
            if dc_layer is not None:
                x_hat = dc_layer(x_hat, x_input, mask)

        _, loss_dict = criterion(x_hat, x_gt)
        total_loss += loss_dict["loss_total"]

        # Unnormalize to original physical signal scale for accurate clinical metric evaluation
        scale_expanded = scale.view(-1, 1, 1, 1)
        x_hat_unnorm = unnormalize_slice(x_hat, scale_expanded)
        x_gt_unnorm = unnormalize_slice(x_gt, scale_expanded)

        # Compute magnitude images
        mag_hat = complex_magnitude(x_hat_unnorm)
        mag_gt = complex_magnitude(x_gt_unnorm)

        metrics = evaluate_batch_metrics(mag_gt, mag_hat)
        psnr_list.append(metrics["psnr"])
        ssim_list.append(metrics["ssim"])
        nmse_list.append(metrics["nmse"])

    n = max(1, len(loader))
    return {
        "val_loss": total_loss / n,
        "val_psnr": sum(psnr_list) / len(psnr_list),
        "val_ssim": sum(ssim_list) / len(ssim_list),
        "val_nmse": sum(nmse_list) / len(nmse_list)
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print("=" * 75)
    print(" Accelerated MRI Reconstruction: Residual Complex U-Net Training")
    print(f" Capstone Phase 1 ")
    print("=" * 75)
    print(f" Hardware Device     : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Acceleration Factor : R={args.acceleration:.1f}x (ACS={args.center_fraction*100:.1f}%, Mask={args.mask_type})")
    print(f" Batch Size / Epochs : {args.batch_size} / {args.epochs}")
    print(f" Optimizer / LR      : AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f" Loss Weights        : alpha(L1)={args.alpha}, beta(SSIM)={args.beta}, gamma(freq)={args.gamma}")

    # 1. Datasets & DataLoaders
    slice_filter = (15, 155) if args.quick_run else None
    train_dataset = CC359Dataset(
        data_root=args.data_root,
        split="train",
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        mask_type=args.mask_type,
        slice_range=slice_filter,
        simulate_noise=args.simulate_noise,
        noise_snr_db=args.noise_snr_db,
        simulate_phase_error=args.simulate_phase_error,
        simulate_motion=args.simulate_motion,
        seed=args.seed
    )

    val_dataset = CC359Dataset(
        data_root=args.data_root,
        split="val",
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        mask_type=args.mask_type,
        slice_range=slice_filter,
        simulate_noise=args.simulate_noise,
        noise_snr_db=args.noise_snr_db,
        simulate_phase_error=args.simulate_phase_error,
        simulate_motion=args.simulate_motion,
        seed=args.seed
    )

    if args.quick_run:
        # Use small subsets for rapid test
        indices_train = list(range(0, min(100, len(train_dataset))))
        indices_val = list(range(0, min(40, len(val_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices_train)
        val_dataset = torch.utils.data.Subset(val_dataset, indices_val)
        print(f" [Quick Run Mode] Subsampled to {len(train_dataset)} train and {len(val_dataset)} val slices.")
    else:
        print(f" Training Samples    : {len(train_dataset)} slices (25 subjects)")
        print(f" Validation Samples  : {len(val_dataset)} slices (10 subjects)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Model Initialization
    if args.model == "unrolled":
        model = UnrolledPhysicsNet(
            num_cascades=args.num_cascades,
            hidden_channels=48,
            num_layers=5,
            share_weights=True
        ).to(device)
        dc_layer = None
        print(f" Model Architecture  : Unrolled Physics-Informed Cascade Net (MoDL, T={args.num_cascades})")
    else:
        model = ResidualComplexUNet(
            in_channels=2,
            out_channels=2,
            base_channels=args.base_channels,
            bilinear=True
        ).to(device)
        dc_layer = DataConsistency().to(device) if args.use_dc else None
        print(f" Model Architecture  : Residual Complex U-Net {'+ End-to-End DC' if args.use_dc else ''}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Trainable Parameters: {num_params:,}")

    # 3. Loss, Optimizer & Scheduler
    criterion = CompositeReconstructionLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    scaler = GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    start_epoch = 1
    best_val_psnr = -float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if requested
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f" Resuming checkpoint from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_psnr = ckpt.get("best_val_psnr", -float("inf"))

    print("\n" + "-" * 85)
    print(f" {'Epoch':<7} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PSNR (dB)':<15} | {'Val SSIM':<10} | {'LR':<10} | {'Time':<8}")
    print("-" * 85)

    # 4. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=args.amp,
            epoch=epoch,
            dc_layer=dc_layer
        )

        val_res = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            dc_layer=dc_layer
        )

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_res["val_loss"])
        else:
            scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - t0

        is_best = val_res["val_psnr"] > best_val_psnr
        if is_best:
            best_val_psnr = val_res["val_psnr"]

        print(
            f" {epoch:02d}/{args.epochs:02d}   | "
            f"{train_res['train_loss']:<12.4f} | "
            f"{val_res['val_loss']:<10.4f} | "
            f"{val_res['val_psnr']:<15.2f} | "
            f"{val_res['val_ssim']:<10.4f} | "
            f"{current_lr:<10.2e} | "
            f"{epoch_time:<6.1f}s"
            f"{' [*BEST*]' if is_best else ''}"
        )

        if new_lr < current_lr:
            print(f"       >>> [ReduceLROnPlateau] Validation loss plateaued. Learning rate decayed: {current_lr:.2e} -> {new_lr:.2e}")

        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_res": train_res,
            "val_res": val_res,
            "best_val_psnr": best_val_psnr,
            "args": vars(args)
        }
        save_checkpoint(checkpoint_data, is_best=is_best, checkpoint_dir=checkpoint_dir)

    print("-" * 75)
    print(f" Training Complete! Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f" Model saved at: {checkpoint_dir.resolve() / 'best_model.pt'}")
    print("=" * 75)


if __name__ == "__main__":
    main()

