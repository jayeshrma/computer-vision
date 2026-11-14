import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class DoubleConv(nn.Module):
    """Standard U-Net Double Convolution Block (dilation=1)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=1, dilation=1),
            ConvBNReLU(out_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        )
    def forward(self, x):
        return self.block(x)

class ParallelDilatedBottleneck(nn.Module):
    """
    Task 3 Controlled Mechanism Change: Multi-Rate Dilated Bottleneck (d = 1, 2, 4, 8).
    Splits out_ch (512) into 4 parallel dilated branches of 128 channels each,
    concatenates them back to 512, and applies a 1x1 projection.
    """
    def __init__(self, in_ch, out_ch, rates=(1, 2, 4, 8)):
        super().__init__()
        self.stem = ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        branch_ch = out_ch // len(rates)
        self.branches = nn.ModuleList([
            ConvBNReLU(out_ch, branch_ch, kernel_size=3, padding=r, dilation=r)
            for r in rates
        ])
        self.project = ConvBNReLU(out_ch, out_ch, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x = self.stem(x)
        feat = torch.cat([b(x) for b in self.branches], dim=1)
        return self.project(feat)

class SegmentationUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, base_ch=32, use_dilated_bottleneck=False):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        # Shared 4-Stage Encoder
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck: Baseline (Standard DoubleConv) vs Controlled Change (Parallel Dilated)
        if use_dilated_bottleneck:
            self.bottleneck = ParallelDilatedBottleneck(c4, c5, rates=(1, 2, 4, 8))
        else:
            self.bottleneck = DoubleConv(c4, c5)

        # Shared 4-Stage Decoder with Concatenation Skip Connections
        self.up4  = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)

        self.up3  = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)

        self.up2  = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)

        self.up1  = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        # 1x1 Pixel Classification Head
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return self.head(d1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
