import torch
import torch.nn as nn

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        '''Input shape: (N, C, H, W)
        Output shape: (N, C / (block_size^2), H * block_size, W * block_size)'''
        return torch.pixel_shuffle(x, self.block_size)

class SRModel(nn.Module):
    def __init__(self, upscale_factor=4, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding='same')
        self.depth_to_space = DepthToSpace(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        outputs = self.depth_to_space(x)
        return outputs