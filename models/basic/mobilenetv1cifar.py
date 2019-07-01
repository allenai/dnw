"""
MobileNet in PyTorch.

Borrowed from https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
"""

import torch.nn as nn
import torch.nn.functional as F

from genutil.config import FLAGS


class Block(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNetV1CIFAR(nn.Module):
    cfg = [
        64,
        (128, 2),
        128,
        (256, 2),
        256,
        (512, 2),
        512,
        512,
        512,
        512,
        512,
        (1024, 2),
        1024,
    ]

    def __init__(self, num_classes=FLAGS.output_size):
        super(MobileNetV1CIFAR, self).__init__()
        self.width_mult = 1 if not hasattr(FLAGS, 'width_mult') else FLAGS.width_mult
        self.conv1 = nn.Conv2d(3, int(32 * self.width_mult), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * self.width_mult))
        self.layers = self._make_layers(in_planes=int(32 * self.width_mult))
        self.linear = nn.Linear(int(1024 * self.width_mult), num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            out_planes = int(self.width_mult * out_planes)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = out.mean(dim=[2,3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
