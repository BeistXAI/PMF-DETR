from __future__ import annotations

from torch import Tensor, nn

from .modules import ConvBNAct, MEEA


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride, "relu")
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, activation=None)
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2, ceil_mode=True),
                ConvBNAct(in_channels, out_channels, 1, activation=None),
            )
        elif in_channels != out_channels:
            self.shortcut = ConvBNAct(in_channels, out_channels, 1, activation=None)
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.shortcut(x))


class MEEAResNet18(nn.Module):
    out_channels = (64, 128, 256, 512)
    strides = (4, 8, 16, 32)

    def __init__(self, input_channels: int = 3, meea: bool = True,
                 meea_dilations=(1, 2, 5), meea_sigma: float = 1.0,
                 meea_wavelength: float = 2.0, meea_gamma: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(input_channels, 32, 3, 2, "relu"),
            ConvBNAct(32, 32, 3, 1, "relu"),
            ConvBNAct(32, 64, 3, 1, "relu"), nn.MaxPool2d(3, 2, 1),
        )
        # MEEA follows the stem/max-pool and precedes the first BasicBlock stage.
        self.edge = MEEA(64, meea_dilations, meea_sigma, meea_wavelength, meea_gamma) if meea else nn.Identity()
        self.stages = nn.ModuleList()
        in_channels = 64
        for index, channels in enumerate(self.out_channels):
            self.stages.append(nn.Sequential(
                BasicBlock(in_channels, channels, 1 if index == 0 else 2),
                BasicBlock(channels, channels),
            ))
            in_channels = channels

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.edge(self.stem(x))
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs
