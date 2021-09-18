""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from dual_tail_parts import *


class Dual(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Dual, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownScale(64, 128)
        self.down2 = DownScale(128, 256)
        self.down3 = DownScale(256, 512)
        self.down4 = DownScale(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.max1 = MaxUnpool(512, 256)
        self.max2 = MaxUnpool(256, 128)
        self.max3 = MaxUnpool(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        x2, idx2 = self.down1(x1)
        x3, idx3 = self.down2(x2)
        x4, idx4 = self.down3(x3)
        x5, idx5 = self.down4(x4)

        d1 = self.max1(x4, idx4, x3.shape)
        d2 = self.max2(d1, idx3, x2.shape)
        d3 = self.max3(d2, idx2, x1.shape)

        x = self.up1(x5, x4)
        x = self.up2(x, d1, x3)
        x = self.up3(x, d2, x2)
        x = self.up4(x, d3, x1)

        logits = self.outc(x)

        return torch.sigmoid(logits)
