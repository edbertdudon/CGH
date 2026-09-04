"""
phase_net.py

A small UNet that predicts a phase-only SLM pattern directly from a
target image, in a single forward pass -- no 200-iteration back-and-forth
like Gerchberg-Saxton. This is the "Network" box from the training-loop
diagram a while back.

The downsampling path shrinks the image while building up feature
channels; the upsampling path rebuilds resolution while mixing in the
matching downsampling features (the skip connections), which is what
lets the network use both coarse, global structure and fine, local
detail when deciding each pixel's phase.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class PhaseNet(nn.Module):
    """
    Input:  (B, in_channels, H, W) target amplitude image
    Output: (B, 1, H, W) predicted phase, in radians, unbounded --
            wrap it with torch.remainder(phase, 2*pi) before use, exactly
            like the phase-wrapping discussion: extra full cycles don't
            change anything physically, so there's no harm in letting the
            raw output range freely and wrapping it afterward.

    H and W need to be divisible by 8 (three 2x downsampling steps).
    """

    def __init__(self, base_channels=32, in_channels=1):
        super().__init__()
        c = base_channels

        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.bottleneck = conv_block(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dec3 = conv_block(c * 8 + c * 4, c * 4)
        self.dec2 = conv_block(c * 4 + c * 2, c * 2)
        self.dec1 = conv_block(c * 2 + c, c)

        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                   # (B, c,   H,   W)
        e2 = self.enc2(self.pool(e1))       # (B, 2c,  H/2, W/2)
        e3 = self.enc3(self.pool(e2))       # (B, 4c,  H/4, W/4)
        b = self.bottleneck(self.pool(e3))  # (B, 8c,  H/8, W/8)

        d3 = self.up(b)                            # (B, 8c, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B, 4c, H/4, W/4)

        d2 = self.up(d3)                            # (B, 4c, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 2c, H/2, W/2)

        d1 = self.up(d2)                            # (B, 2c, H, W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, c,  H, W)

        phase = self.out_conv(d1)  # (B, 1, H, W)
        return phase
