"""
Lightweight FPN Neck.

Fuses C3 (112ch), C4 (224ch), C5 (448ch) from RepViT-M1.0
into a single 64-channel feature map at stride-8 resolution.

Input  (360x640):
  C3: (B, 112, 45, 80)
  C4: (B, 224, 23, 40)
  C5: (B, 448, 11, 20)

Output:
  (B, 64, 45, 80)   <-- used by LaneATT head

All ops: Conv2d + BN + ReLU + Upsample  (no grid_sample, fully INT8 quantizable)
"""

import torch.nn as nn
import torch.nn.functional as F


class FPNNeck(nn.Module):
    def __init__(
        self,
        in_channels=(112, 224, 448),   # C3, C4, C5 from RepViT-M1.0
        out_channels=64,
    ):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels

        # lateral 1x1 convs — reduce each level to out_channels
        self.lat_c5 = self._lateral(c5_in, out_channels)
        self.lat_c4 = self._lateral(c4_in, out_channels)
        self.lat_c3 = self._lateral(c3_in, out_channels)

        # output 3x3 conv to smooth fused features
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _lateral(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, c3, c4, c5):
        # top-down pathway
        p5 = self.lat_c5(c5)                                          # (B,64,11,20)
        p4 = self.lat_c4(c4) + F.interpolate(                         # (B,64,23,40)
            p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lat_c3(c3) + F.interpolate(                         # (B,64,45,80)
            p4, size=c3.shape[2:], mode='nearest')

        return self.out_conv(p3)                                       # (B,64,45,80)
