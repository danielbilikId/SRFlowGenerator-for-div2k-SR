import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simpler residual block to replace heavy RDBs."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class SRFlowGenerator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4, upscale_factor=4):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.res_blocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(nf, out_nc * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.res_blocks(fea))
        fea = fea + trunk
        out = self.upsample(fea)
        return out

