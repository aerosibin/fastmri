"""
interactive_viewer.py
Interactive Matplotlib GUI for live demonstration to mentors and researchers.
Features:
  - Interactive slice scrubbing slider
  - Real-time acceleration factor toggle (R=2x, 4x, 8x)
  - Real-time mask type selection (Gaussian, Random, Equispaced)
  - Live metric computation (PSNR in dB, SSIM)
  - Subject selector dropdown / buttons
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

from .mri_processor import (
    load_subject_kspace,
    reconstruct_slice,
    generate_undersampling_mask,
    simulate_undersampling,
    compute_reconstruction_metrics
)


class MRIInteractiveViewer:
    def __init__(self, filepath: Path, subject_id: Optional[str] = None):
        self.filepath = Path(filepath)
        self.subject_id = subject_id or self.filepath.stem.split(".")[0]

        print(f"Loading k-space volume for {self.subject_id}...")
        self.complex_kspace = load_subject_kspace(self.filepath, mmap=True)
        self.total_slices = self.complex_kspace.shape[0]

        # State variables
        self.current_slice = self.total_slices // 2
        self.current_accel = 4.0
        self.current_mask_type = "gaussian"

        # Setup figure
        plt.rcParams.update({"font.family": "sans-serif", "image.interpolation": "nearest"})
        self.fig = plt.figure(figsize=(15, 8), facecolor="#0f172a")
        self.fig.canvas.manager.set_window_title(f"Calgary-Campinas MRI Interactive Explorer - {self.subject_id}")

        # Header title
        self.title_text = self.fig.suptitle(
            f"Subject: {self.subject_id}  |  Slice: {self.current_slice}/{self.total_slices - 1}  |  Acceleration: R={self.current_accel:.0f}x",
            color="white", weight="bold", fontsize=14, y=0.96
        )

        # 4 Main display axes
        # [0]: Sampling Mask, [1]: Ground Truth, [2]: Zero-Filled Aliased, [3]: Error Map
        self.ax_mask = self.fig.add_axes([0.05, 0.28, 0.20, 0.60])
        self.ax_gt = self.fig.add_axes([0.28, 0.28, 0.20, 0.60])
        self.ax_zf = self.fig.add_axes([0.51, 0.28, 0.20, 0.60])
        self.ax_err = self.fig.add_axes([0.74, 0.28, 0.20, 0.60])

        for ax in [self.ax_mask, self.ax_gt, self.ax_zf, self.ax_err]:
            ax.axis("off")

        # Colorbar axis for error map
        self.cbar_ax = self.fig.add_axes([0.95, 0.28, 0.015, 0.60])

        # Interactive controls (bottom)
        # 1. Slice Slider
        self.ax_slider = self.fig.add_axes([0.15, 0.14, 0.70, 0.04], facecolor="#1e293b")
        self.slider = Slider(
            self.ax_slider,
            "Slice Index",
            0,
            self.total_slices - 1,
            valinit=self.current_slice,
            valstep=1,
            valfmt="%d",
            color="#38bdf8"
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self.on_slice_change)

        # 2. Acceleration Factor Radio Buttons
        self.ax_radio_accel = self.fig.add_axes([0.05, 0.02, 0.18, 0.09], facecolor="#1e293b")
        self.radio_accel = RadioButtons(
            self.ax_radio_accel,
            ("R=2x", "R=4x", "R=8x"),
            active=1,
            activecolor="#38bdf8"
        )
        for label in self.radio_accel.labels:
            label.set_color("white")
            label.set_fontsize(9)
        self.radio_accel.on_clicked(self.on_accel_change)

        # 3. Mask Type Radio Buttons
        self.ax_radio_mask = self.fig.add_axes([0.28, 0.02, 0.22, 0.09], facecolor="#1e293b")
        self.radio_mask = RadioButtons(
            self.ax_radio_mask,
            ("Gaussian (Variable Density)", "Uniform Random", "Equispaced"),
            active=0,
            activecolor="#38bdf8"
        )
        for label in self.radio_mask.labels:
            label.set_color("white")
            label.set_fontsize(8)
        self.radio_mask.on_clicked(self.on_mask_type_change)

        # Precompute initial slice
        self.im_mask = None
        self.im_gt = None
        self.im_zf = None
        self.im_err = None
        self.txt_gt = None
        self.txt_zf = None
        self.cb = None

        self.update_display(full_redraw=True)

        # Keyboard navigation
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def update_display(self, full_redraw: bool = False):
        kspace_slice = self.complex_kspace[self.current_slice]
        gt_img = reconstruct_slice(kspace_slice)

        # Generate mask
        center_frac = 0.08 if self.current_accel >= 4.0 else 0.16
        mask = generate_undersampling_mask(
            shape=(256, 256),
            acceleration=self.current_accel,
            center_fraction=center_frac,
            mask_type=self.current_mask_type,
            seed=42
        )

        # Undersampled Zero-Filled reconstruction
        _, zf_img = simulate_undersampling(kspace_slice, mask)
        err_img = np.abs(gt_img - zf_img)

        # Metrics
        metrics = compute_reconstruction_metrics(gt_img, zf_img)

        vmax_img = np.percentile(gt_img, 99.8)
        vmax_err = vmax_img * 0.4

        if full_redraw or self.im_mask is None:
            self.ax_mask.clear()
            self.ax_gt.clear()
            self.ax_zf.clear()
            self.ax_err.clear()
            for ax in [self.ax_mask, self.ax_gt, self.ax_zf, self.ax_err]:
                ax.axis("off")

            self.im_mask = self.ax_mask.imshow(mask, cmap="Blues_r", vmin=0, vmax=1)
            self.ax_mask.set_title(f"Sampling Mask (k)\nR={self.current_accel:.0f}x ({mask.mean()*100:.1f}%)",
                                  color="#38bdf8", pad=8, weight="bold")

            self.im_gt = self.ax_gt.imshow(gt_img, cmap="gray", vmin=0, vmax=vmax_img)
            self.ax_gt.set_title("Ground Truth\nFully Sampled (2D IFFT)",
                                color="#f8fafc", pad=8, weight="bold")

            self.im_zf = self.ax_zf.imshow(zf_img, cmap="gray", vmin=0, vmax=vmax_img)
            self.ax_zf.set_title(f"Zero-Filled Aliased\nPSNR: {metrics['psnr']:.1f} dB | SSIM: {metrics['ssim']:.3f}",
                                color="#f43f5e", pad=8, weight="bold")

            self.im_err = self.ax_err.imshow(err_img, cmap="inferno", vmin=0, vmax=vmax_err)
            self.ax_err.set_title("Absolute Error Map\n|Ground Truth - Zero-Filled|",
                                 color="#fb923c", pad=8, weight="bold")

            if self.cb is None:
                self.cb = self.fig.colorbar(self.im_err, cax=self.cbar_ax)
                self.cb.set_label("Artifact Residual", color="#94a3b8", fontsize=9)
                self.cb.ax.yaxis.set_tick_params(color="#94a3b8")
                plt.setp(plt.getp(self.cb.ax.axes, "yticklabels"), color="#94a3b8", fontsize=8)
        else:
            self.im_mask.set_data(mask)
            self.ax_mask.set_title(f"Sampling Mask (k)\nR={self.current_accel:.0f}x ({mask.mean()*100:.1f}%)",
                                  color="#38bdf8", pad=8, weight="bold")

            self.im_gt.set_data(gt_img)
            self.im_gt.set_clim(0, vmax_img)

            self.im_zf.set_data(zf_img)
            self.im_zf.set_clim(0, vmax_img)
            self.ax_zf.set_title(f"Zero-Filled Aliased\nPSNR: {metrics['psnr']:.1f} dB | SSIM: {metrics['ssim']:.3f}",
                                color="#f43f5e", pad=8, weight="bold")

            self.im_err.set_data(err_img)
            self.im_err.set_clim(0, vmax_err)

        self.title_text.set_text(
            f"Subject: {self.subject_id}  |  Slice: {self.current_slice}/{self.total_slices - 1}  |  Acceleration: R={self.current_accel:.0f}x ({self.current_mask_type.title()})"
        )
        self.fig.canvas.draw_idle()

    def on_slice_change(self, val):
        self.current_slice = int(val)
        self.update_display(full_redraw=False)

    def on_accel_change(self, label):
        mapping = {"R=2x": 2.0, "R=4x": 4.0, "R=8x": 8.0}
        self.current_accel = mapping[label]
        self.update_display(full_redraw=False)

    def on_mask_type_change(self, label):
        if "Gaussian" in label:
            self.current_mask_type = "gaussian"
        elif "Uniform" in label:
            self.current_mask_type = "random"
        else:
            self.current_mask_type = "equispaced"
        self.update_display(full_redraw=False)

    def on_key_press(self, event):
        if event.key in ["right", "up"]:
            new_slice = min(self.total_slices - 1, self.current_slice + 1)
            self.slider.set_val(new_slice)
        elif event.key in ["left", "down"]:
            new_slice = max(0, self.current_slice - 1)
            self.slider.set_val(new_slice)


def launch_viewer(filepath: Path, subject_id: Optional[str] = None):
    viewer = MRIInteractiveViewer(filepath=filepath, subject_id=subject_id)
    plt.show()
