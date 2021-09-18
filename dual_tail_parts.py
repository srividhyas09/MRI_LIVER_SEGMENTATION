""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownScale(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, return_indices=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x, idx = self.maxpool(x)
        return x, idx


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.conv2 = DoubleConv(out_channels*3, out_channels, in_channels // 2)

    def forward(self, x1, x2, x3=None):

        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if x3 is None:
            x = torch.cat([x2, x1], dim=1)
            x = self.conv1(x)
        else:
            x = torch.cat([x3, x2, x1], dim=1)
            x = self.conv2(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MaxUnpool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.MaxUnpool = nn.MaxUnpool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, idx, orig_shape):
        x = self.MaxUnpool(x, idx, orig_shape)
        x = self.conv(x)
        return x
