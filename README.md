# FastMRI: Accelerated MRI Reconstruction
---
Trained Models:
1. [Residual Complex U-Net R=5x](https://drive.google.com/drive/folders/1kNydNMfaqs2RE7FIj1VGPgIJd3UXDGDj?usp=sharing)
2. [Unrolled Physics-Informed Cascade Net](https://drive.google.com/drive/folders/1kNydNMfaqs2RE7FIj1VGPgIJd3UXDGDj?usp=sharing)

Datset Used: [CC359 dataset](https://drive.google.com/drive/folders/1TJIUYGfPznnj_ZLUtVXY8LdA9e6SJCmE?usp=sharing)

## 1. System Overview & Mathematics

This codebase directly implements the system architecture specified in **Chapters 3 & 4 of the Capstone Report**:

### 1.1 Data Ingestion & Decoupling
- **Dataset**: Calgary-Campinas Brain MR Dataset (CC359), 3.0T GE Discovery MR750, $T_1$-weighted sagittal scans ($1.0\text{ mm}^3$ isotropic).
- **Format**: NumPy (`.npy`) complex arrays of shape $(N_{\text{slices}}, 256, 256, 2)$ (Real, Imaginary), with unshifted DC at index `[0, 0]`.
- **Memory-Mapped Streaming**: `src.data.CC359Dataset` uses `numpy.memmap` mode=`'r'`, indexing all 4,254 training slices (25 subjects) and 1,700 validation slices (10 subjects) with zero RAM bloat.

### 1.2 Forward Acquisition & Retrospective Undersampling
$$y = M \odot \mathcal{F}(x) + \epsilon \quad (4.2)$$
- $x \in \mathbb{C}^{256 \times 256}$: Fully sampled reference brain image.
- $M \in \{0, 1\}^{256 \times 256}$: Variable-density Gaussian sampling mask along phase-encoding lines:
  - $R = 4\times$ (primary configuration)
  - $R = 8\times$ (higher acceleration stress test)
  - Central 8% Auto-Calibration Signal (ACS) preserved.
- $\epsilon = \epsilon_r + j\epsilon_i$: Complex Gaussian measurement noise (optional simulation).
- $y$: Observed undersampled $k$-space.

### 1.3 Zero-Filled Input Preparation
$$x_{\text{ZF}} = \mathcal{F}^{-1}(y) \quad (4.5)$$
$$X_{\text{input}} = \begin{bmatrix} \text{Re}(x_{\text{ZF}}) \\ \text{Im}(x_{\text{ZF}}) \end{bmatrix} \in \mathbb{R}^{B \times 2 \times 256 \times 256} \quad (4.6, 4.7)$$

### 1.4 Residual Complex U-Net
- 4-level encoder-decoder architecture with skip connections.
- Predicts real and imaginary residual corrections:
  $$\Delta x = x - x_{\text{ZF}} \quad (4.8)$$
  $$\hat{x} = x_{\text{ZF}} + \Delta x_{\text{pred}} \quad (4.9)$$

### 1.5 Composite Loss Function
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{L_1} + \beta \mathcal{L}_{\text{SSIM}} + \gamma \mathcal{L}_{\text{freq}} \quad (4.10)$$
- **$\mathcal{L}_{L_1}$**: Pixel-wise $L_1$ reconstruction loss on complex channels and magnitude.
- **$\mathcal{L}_{\text{SSIM}}$**: Structural Similarity loss ($1 - \text{SSIM}$) via differentiable Gaussian smoothing.
- **$\mathcal{L}_{\text{freq}}$**: Frequency domain penalty $\|\mathcal{F}(\hat{x}) - \mathcal{F}(x)\|_1$ in $k$-space.

---

## 2. Directory Structure

```
c:/SNUC/Capstone/calgary-campinas/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mask.py             # Gaussian R=4/R=8 masks + noise/phase/motion errors
│   │   ├── transforms.py       # Complex 2D IFFT/FFT, 2-channel stacking, normalization
│   │   └── dataset.py          # Fast memory-mapped PyTorch Dataset (4,254 train, 1,700 val)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py             # 4-level Residual Complex U-Net (4.32M parameters)
│   │   └── data_consistency.py # Physics-based Data Consistency (DC) projection layer
│   ├── losses/
│   │   ├── __init__.py
│   │   └── composite_loss.py   # Differentiable composite loss (L1 + SSIM + Freq)
│   ├── metrics.py              # PSNR, SSIM, NMSE, and MAE evaluation functions
│   └── utils.py                # Seeding, device setup, checkpoint handlers
├── train.py                    # Master training pipeline (AdamW, Cosine Annealing, AMP)
├── evaluate.py                 # Full dataset benchmark & visualization generator
├── test_pipeline.py            # End-to-end unit test suite
└── checkpoints/                # Saved weights (best_model.pt)
```

---

## 3. How to Run

### Step 1: Run the Verification Suite
Runs all 8 system checks (mask generation, IFFT invertibility, dataset loading, U-Net forward pass, DC layer, loss backprop, metrics, and optimizer step):
```powershell
python test_pipeline.py
```

### Step 2: Train the Residual Complex U-Net
Train on the full 25-subject dataset (4,254 slices) for 30 epochs:
```powershell
python train.py --acceleration 4.0 --epochs 30 --batch_size 8 --lr 1e-4 --amp
```
*To quickly test the training loop for 1 epoch on a small subset:*
```powershell
python train.py --quick_run --epochs 1
```

### Step 3: Evaluate on Validation Set
Evaluates all 1,700 validation slices across 10 subjects, reports PSNR, SSIM, and NMSE gains over the Zero-Filled baseline, and generates comparison images:
```powershell
python evaluate.py --checkpoint checkpoints/best_model.pt --acceleration 4.0
```
Outputs are saved into [`reports/evaluation_results/`](file:///c:/SNUC/Capstone/calgary-campinas/reports/evaluation_results/).

