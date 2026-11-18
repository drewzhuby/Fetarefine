"""Shared DeepLabV3+ with ResNet-50 backbone and low-level feature decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18), out_channels=256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet-50 encoder and low-level skip connection."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        pretrained: bool = True,
        output_stride: int = 16,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        dilation_config = [False, False, True] if output_stride == 16 else [False, True, True]
        self.backbone = resnet50(weights=weights, replace_stride_with_dilation=dilation_config)

        # Adapt first convolution if in_channels != 3
        if in_channels != 3:
            orig_conv = self.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.zero_()
                new_conv.weight[:, :3].copy_(orig_conv.weight)
            self.backbone.conv1 = new_conv

        self.aspp = ASPP(2048, atrous_rates=(6, 12, 18) if output_stride == 16 else (12, 24, 36), out_channels=256)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        # Encoder forward
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x0 = self.backbone.maxpool(x0)

        c1 = self.backbone.layer1(x0)  # Low-level features: 1/4 res, 256 channels
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)  # High-level features: 1/16 res, 2048 channels

        aspp_features = self.aspp(c4)
        aspp_up = F.interpolate(aspp_features, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        low_level_features = self.low_level_proj(c1)

        fused = torch.cat([aspp_up, low_level_features], dim=1)
        decoded = self.decoder(fused)
        logits = self.head(decoded)

        # Upsample back to input image resolution
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
