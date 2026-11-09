"""
test_pipeline.py
End-to-end verification script for the Accelerated MRI Reconstruction codebase.
Tests:
  1. Mask generation (Gaussian R=4, R=8) and error models.
  2. Complex tensor transforms and normalization.
  3. CC359Dataset slice loading and shapes.
  4. Residual Complex U-Net architecture and residual addition.
  5. Physics Data Consistency layer.
  6. Differentiable Composite Loss and gradient backpropagation.
  7. Quantitative metrics calculation (PSNR, SSIM, NMSE).
  8. One complete training step with AdamW optimizer.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CC359Dataset,
    generate_undersampling_mask,
    apply_undersampling,
    add_gaussian_noise,
    add_phase_errors,
    add_motion_artifacts,
    complex_to_chan2,
    chan2_to_complex,
    complex_magnitude,
    ifft2_centered,
    fft2_centered,
    normalize_slice,
    unnormalize_slice
)
from src.models import ResidualComplexUNet, DataConsistency
from src.losses import CompositeReconstructionLoss
from src.metrics import compute_psnr_np, compute_ssim_np, compute_nmse_np, evaluate_batch_metrics
from src.utils import get_device


def run_tests():
    print("=" * 70)
    print(" Running Accelerated MRI Reconstruction Verification Suite")
    print("=" * 70)
    device = get_device("auto")
    print(f" Compute Device: {device}\n")

    # Test 1: Mask Generator & Errors
    print("[1/8] Testing Mask Generation & Error Models...")
    m4 = generate_undersampling_mask(shape=(256, 256), acceleration=4.0, center_fraction=0.08, mask_type="gaussian", seed=42)
    m8 = generate_undersampling_mask(shape=(256, 256), acceleration=8.0, center_fraction=0.08, mask_type="gaussian", seed=42)
    assert m4.shape == (256, 256) and m8.shape == (256, 256)
    assert np.isclose(1.0 / m4.mean(), 4.0, atol=0.25)
    assert np.isclose(1.0 / m8.mean(), 8.0, atol=0.5)

    k_dummy = torch.randn(256, 256, dtype=torch.complex64)
    k_noisy = add_gaussian_noise(k_dummy, snr_db=30.0, seed=42)
    k_phase = add_phase_errors(k_dummy, max_phase_rad=0.2, seed=42)
    k_motion = add_motion_artifacts(k_dummy, affected_lines_fraction=0.08, seed=42)
    assert k_noisy.shape == k_dummy.shape
    assert k_phase.shape == k_dummy.shape
    assert k_motion.shape == k_dummy.shape
    print("      -> Mask R=4, R=8, noise, phase, and motion tests PASSED.")

    # Test 2: Transforms & Normalization
    print("[2/8] Testing Complex Transforms & Normalization...")
    c_img = torch.randn(2, 256, 256, dtype=torch.complex64)
    t_chan2 = complex_to_chan2(c_img)
    assert t_chan2.shape == (2, 2, 256, 256)
    c_recon = chan2_to_complex(t_chan2)
    assert torch.allclose(c_img, c_recon)

    k_fwd = fft2_centered(c_img)
    c_inv = ifft2_centered(k_fwd)
    assert torch.allclose(c_img, c_inv, atol=1e-5)

    t_norm, scale = normalize_slice(t_chan2)
    t_unnorm = unnormalize_slice(t_norm, scale)
    assert torch.allclose(t_chan2, t_unnorm, atol=1e-5)
    print("      -> FFT/IFFT invertibility and normalization tests PASSED.")

    # Test 3: Dataset Loading
    print("[3/8] Testing CC359Dataset Loading...")
    train_ds = CC359Dataset(split="train", acceleration=4.0)
    val_ds = CC359Dataset(split="val", acceleration=4.0)
    assert len(train_ds) == 4254, f"Expected 4254 train slices, got {len(train_ds)}"
    assert len(val_ds) == 1700, f"Expected 1700 val slices, got {len(val_ds)}"

    sample = train_ds[50]
    assert sample["input"].shape == (2, 256, 256)
    assert sample["ground_truth"].shape == (2, 256, 256)
    assert sample["target"].shape == (2, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    print(f"      -> Dataset: {len(train_ds)} train slices, {len(val_ds)} val slices indexed & loaded.")

    # Test 4: Residual Complex U-Net
    print("[4/8] Testing Residual Complex U-Net Forward Pass...")
    model = ResidualComplexUNet(in_channels=2, out_channels=2, base_channels=32).to(device)
    x_in = sample["input"].unsqueeze(0).to(device)
    x_hat, delta_x = model(x_in, return_residual=True)
    assert x_hat.shape == (1, 2, 256, 256)
    assert delta_x.shape == (1, 2, 256, 256)
    assert torch.allclose(x_hat, x_in + delta_x), "Residual addition must equal x_in + delta_x!"
    print("      -> U-Net forward pass & exact residual formulation PASSED.")

    # Test 5: Data Consistency
    print("[5/8] Testing Data Consistency Layer...")
    dc = DataConsistency().to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    x_dc = dc(x_hat, x_in, mask)
    assert x_dc.shape == (1, 2, 256, 256)
    print("      -> Data Consistency projection PASSED.")

    # Test 6: Composite Loss & Gradient Flow
    print("[6/8] Testing Differentiable Composite Loss & Backpropagation...")
    criterion = CompositeReconstructionLoss(alpha=1.0, beta=0.5, gamma=0.1).to(device)
    x_gt = sample["ground_truth"].unsqueeze(0).to(device)

    loss, loss_dict = criterion(x_hat, x_gt)
    assert loss.requires_grad
    loss.backward()

    # Verify gradients exist across model weights
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads, "Gradients failed to propagate to model parameters!"
    print(f"      -> Loss: {loss.item():.4f} (L1: {loss_dict['loss_l1']:.4f}, SSIM: {loss_dict['loss_ssim']:.4f}, Freq: {loss_dict['loss_freq']:.4f})")
    print("      -> Gradient backpropagation PASSED.")

    # Test 7: Metrics
    print("[7/8] Testing Metric Functions (PSNR, SSIM, NMSE)...")
    mag_gt = complex_magnitude(sample["ground_truth"]).squeeze().numpy()
    mag_in = complex_magnitude(sample["input"]).squeeze().numpy()
    metrics = evaluate_batch_metrics(mag_gt, mag_in)
    assert 10.0 <= metrics["psnr"] <= 50.0
    assert 0.0 <= metrics["ssim"] <= 1.0
    print(f"      -> ZF Baseline Metrics: PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}, NMSE={metrics['nmse']:.4f}")

    # Test 8: Optimizer Step
    print("[8/8] Testing AdamW Optimization Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    print("      -> Optimizer step PASSED.")

    print("\n" + "=" * 70)
    print(" ALL 8 SYSTEM TESTS PASSED SUCCESSFULLY! Codebase is ready for training.")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

