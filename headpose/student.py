"""Student head-pose network, trained from scratch (no ImageNet or teacher weights).

MobileNetV3-Small backbone (randomly initialised) on 64x64 head crops, with two heads:
  * rot6d : 6D rotation, turned into a rotation matrix exactly like SixDRepNet
  * look  : logit for "looking at the ego-vehicle" (trained in stage 2 on JAAD 'look' labels)
About 1.1 M parameters, versus 26.8 M for the RepVGG-A2 teacher.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

from .rotation import compute_rotation


class HeadStudent(nn.Module):
    def __init__(self, width=1.0, dropout=0.2):
        super().__init__()
        self.features = mobilenet_v3_small(weights=None, width_mult=width).features  # from scratch
        for m in self.features.modules():          # torchvision uses BN momentum 0.01, tuned for long
            if isinstance(m, nn.BatchNorm2d):      # ImageNet runs; 0.1 lets the running statistics
                m.momentum = 0.1                   # follow our shorter training
        c = self.features[-1].out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(c, 256), nn.Hardswish(), nn.Dropout(dropout))
        self.rot6d = nn.Linear(256, 6)
        self.look = nn.Linear(256, 1)

    def forward(self, x):
        h = self.neck(torch.flatten(self.pool(self.features(x)), 1))
        return compute_rotation(self.rot6d(h)), self.look(h).squeeze(1)


# ImageNet statistics, same normalisation as the teacher
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(batch_uint8):
    """(N, H, W, 3) uint8 RGB numpy -> normalised float tensor (N, 3, H, W)."""
    x = torch.as_tensor(batch_uint8).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD
