"""
src/models/unet.py
Residual Complex U-Net Architecture for Accelerated MRI Reconstruction.
Conforms directly to Capstone Report Section 4.3.3:
  - 4-level U-Net encoder-decoder architecture with skip connections.
  - Two-channel input X_input in R^{B x 2 x 256 x 256} (Real & Imaginary).
  - Residual learning (Equations 4.8 & 4.9):
        Delta_x = x - x_ZF
        x_hat = x_ZF + Delta_x_pred
  - Predicts 2-channel residual corrections to remove aliasing artifacts.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms import complex_magnitude


class DoubleConv(nn.Module):
    """
    [Conv2d -> InstanceNorm2d -> LeakyReLU] x 2
    Uses InstanceNorm2d for robust normalization across varying scanner gains and small batch sizes.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling via bilinear interpolation or ConvTranspose2d, then DoubleConv with skip connection."""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle potential padding differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Concatenate skip connection along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResidualComplexUNet(nn.Module):
    """
    4-Level Residual Complex U-Net for accelerated MRI reconstruction.
    Input:
        x_input: (B, 2, 256, 256) - Zero-filled reconstruction (Real, Imag)
    Output:
        x_hat: (B, 2, 256, 256) - Final reconstructed complex image x_ZF + Delta_x_pred
        delta_x: (B, 2, 256, 256) - Predicted residual artifact correction
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        bilinear: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        c = base_channels  # e.g., 32

        # 4-Level Encoder
        self.inc = DoubleConv(in_channels, c)            # Level 1: 256 x 256 (c)
        self.down1 = Down(c, c * 2)                       # Level 2: 128 x 128 (2c)
        self.down2 = Down(c * 2, c * 4)                   # Level 3: 64 x 64   (4c)
        self.down3 = Down(c * 4, c * 8)                   # Level 4: 32 x 32   (8c)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.bottleneck = Down(c * 8, c * 16 // factor)   # Bottleneck: 16 x 16 (16c or 8c)

        # 4-Level Decoder with skip connections
        self.up1 = Up(c * 16, c * 8 // factor, bilinear)  # 32 x 32
        self.up2 = Up(c * 8, c * 4 // factor, bilinear)   # 64 x 64
        self.up3 = Up(c * 4, c * 2 // factor, bilinear)   # 128 x 128
        self.up4 = Up(c * 2, c, bilinear)                 # 256 x 256

        # Output projection head (1x1 Conv) predicting residual correction Delta_x
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(
        self,
        x_input: torch.Tensor,
        return_residual: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with residual addition:
            Delta_x_pred = U-Net(x_input)
            x_hat = x_input + Delta_x_pred
        
        Args:
            x_input: Tensor of shape (B, 2, 256, 256) representing [Re(x_ZF), Im(x_ZF)]
            return_residual: If True, returns (x_hat, delta_x_pred)
            
        Returns:
            x_hat: Reconstructed complex image tensor of shape (B, 2, 256, 256)
        """
        # Encoder forward pass
        x1 = self.inc(x_input)       # Level 1 skip
        x2 = self.down1(x1)          # Level 2 skip
        x3 = self.down2(x2)          # Level 3 skip
        x4 = self.down3(x3)          # Level 4 skip
        x5 = self.bottleneck(x4)     # Bottleneck

        # Decoder forward pass with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Predict residual correction Delta_x
        delta_x_pred = self.outc(x)

        # Residual reconstruction (Equation 4.9: x_hat = x_ZF + Delta_x_pred)
        x_hat = x_input + delta_x_pred

        if return_residual:
            return x_hat, delta_x_pred
        return x_hat

    def get_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to compute magnitude image |x| from 2-channel tensor (B, 2, H, W)."""
        return complex_magnitude(x)

