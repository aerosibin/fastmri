# Calgary-Campinas Brain MRI (CC359) Visualization Suite

This module provides an end-to-end visualization, exploration, and presentation framework for the single-channel complex $k$-space MRI dataset from the Calgary-Campinas Public Brain MR Dataset (CC359).

---

## Quick Start Commands

### 1. Generate All Presentation Figures (High-Res 300 DPI)
```powershell
python run_visualization.py
```
Outputs are saved into [`reports/figures/`](file:///c:/SNUC/Capstone/calgary-campinas/reports/figures/):
- `01_kspace_physics_pipeline.png`: Raw $k$-space components (Real, Imag, Log-Mag) $\to$ Reconstructed Image.
- `02_accelerated_mri_problem.png`: The core Capstone problem ($R=1\times, 2\times, 4\times, 8\times$ masks, zero-filled aliased images, and error maps with PSNR/SSIM).
- `03_orthogonal_brain_views.png`: 3D Multi-Planar Reformatting (MPR) showing Sagittal, Coronal, and Axial planes.
- `04_multislice_montage.png`: Whole-brain sagittal slice progression.

### 2. Launch Interactive Live GUI Viewer
```powershell
python run_visualization.py --interactive
```
- **Slice Slider**: Scrub through all 170 slices in real time (or use Left/Right arrow keys).
- **Acceleration Toggle**: Switch between $R=2\times$, $R=4\times$, and $R=8\times$.
- **Mask Selector**: Toggle between Gaussian variable density, Uniform random, and Equispaced masks.
- **Dynamic Metrics**: Instant real-time updates for PSNR (dB) and SSIM.

### 3. Inspect Any Subject or Slice
```powershell
# List all 35 available subjects (25 Train + 10 Val):
python run_visualization.py --list-subjects

# Visualize a specific subject and slice:
python run_visualization.py --subject e14155s3_P69120 --slice 85
```

---

## Important Points

### 1. Data Representation & MRI Acquisition Physics
- **Complex $k$-Space**: The scanner records raw radiofrequency signals in the spatial frequency domain ($k$-space). Each acquisition matrix is shape `(N_slices, 256, 256, 2)` where the last dimension holds **Real** and **Imaginary** components:
  $$S(k_x, k_y) = \text{Real} + i \cdot \text{Imag}$$
- **Reconstruction Operator**: By MRI physics, the spatial image $m(x, y)$ is the 2D Inverse Fourier Transform ($\mathcal{F}^{-1}$) of $k$-space:
  $$m(x, y) = |\mathcal{F}^{-1}\{S(k_x, k_y)\}|$$
- In CC359 single-channel data, DC is natively at `(0, 0)`, so direct 2D IFFT (`np.fft.ifft2`) reconstructs properly centered spatial domain brain slices.

### 2. The Core Capstone Problem: Accelerated MRI
- **Clinical Motivation**: Full MRI scans take long acquisition times (e.g. 5–10 minutes per sequence), causing patient discomfort, high costs, and motion artifacts.
- **Acceleration via Retrospective Undersampling**:
  - We skip lines in $k$-space during acquisition by a factor of $R$ (e.g., $R=4\times$ or $R=8\times$), reducing scan time by $4\times$ or $8\times$.
  - We preserve the central Auto-Calibration Signal (ACS) region (8% to 16% of low-frequency lines) because that is where ~80%+ of image energy and contrast resides.
- **The Aliasing Dilemma (Zero-Filled Reconstruction)**:
  - If we simply set missing lines to zero and take the IFFT (Zero-Filled reconstruction), Nyquist-Shannon sampling is violated.
  - This results in severe coherent aliasing, streaking, and blurring artifacts:
    - Ground Truth ($R=1\times$): Reference $\infty\text{ dB}$, $\text{SSIM} = 1.000$
    - $R=2\times$: $\approx 31.6\text{ dB}$, $\text{SSIM} \approx 0.86$
    - $R=4\times$: $\approx 26.3\text{ dB}$, $\text{SSIM} \approx 0.71$
    - $R=8\times$: $\approx 24.4\text{ dB}$, $\text{SSIM} \approx 0.62$
- **Where Deep Learning Enters**:
  - The objective of your Capstone model is to invert this ill-posed inverse problem:
    $$\min_{\theta} \mathcal{L}(f_\theta(y, M), x)$$
    where $y$ is undersampled $k$-space, $M$ is the mask, and $x$ is the fully sampled ground truth image.

### 3. Volumetric & Anatomical Coverage
- The 170 continuous 1.0 mm slices form a full 3D isotropic volume ($170 \times 256 \times 256$).
- Multi-Planar Reformatting (MPR) enables clean reconstruction along Sagittal, Coronal (hippocampus, ventricles), and Axial (cortex, basal ganglia) orientations.

---

## Code Architecture

- [`visualization/mri_processor.py`](file:///c:/SNUC/Capstone/calgary-campinas/visualization/mri_processor.py): Core mathematical transforms (complex loading, 2D IFFT, mask generation, metric calculations).
- [`visualization/figure_generator.py`](file:///c:/SNUC/Capstone/calgary-campinas/visualization/figure_generator.py): Production scripts for publication-quality figures with safe disk saving.
- [`visualization/interactive_viewer.py`](file:///c:/SNUC/Capstone/calgary-campinas/visualization/interactive_viewer.py): Interactive slice explorer and live accelerator GUI.
- [`run_visualization.py`](file:///c:/SNUC/Capstone/calgary-campinas/run_visualization.py): Master CLI entry point.
