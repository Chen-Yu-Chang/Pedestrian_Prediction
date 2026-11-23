"""Frozen SixDRepNet teacher (https://github.com/Shohruh72/SixDRepNet).

The release checkpoints (e.g. best_a2.pt) store the whole pickled model, whose classes live in the
repo's `nets.nn` module. Instead of putting that repo on sys.path (its `utils` package would clash
with other code), we register two small compatible classes under the module name `nets.nn` and
load the checkpoint with torch's restricted `weights_only=True` unpickler, so only an explicit
allow-list of classes can be created.

Download:  https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_a2.pt
"""
import collections
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .rotation import compute_euler, compute_rotation


class RepVGGBlock(nn.Module):
    """Forward pass of SixDRepNet's RepVGGBlock (training-time and re-parameterised forms)."""

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(x)))
        out = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x) + out))


class HPE(nn.Module):
    """Forward pass of SixDRepNet's HPE: RepVGG backbone -> 6D -> rotation matrix."""

    def forward(self, x):
        for layer in (self.layer0, self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        x = torch.flatten(self.gap(x), 1)
        return compute_rotation(self.linear_reg(x))


def _register_compat_module():
    nets = sys.modules.setdefault('nets', types.ModuleType('nets'))
    mod = types.ModuleType('nets.nn')
    mod.HPE, mod.RepVGGBlock = HPE, RepVGGBlock
    HPE.__module__ = RepVGGBlock.__module__ = 'nets.nn'
    sys.modules['nets.nn'] = nets.nn = mod


def load_teacher(path, device='cpu'):
    _register_compat_module()
    allowed = [HPE, RepVGGBlock, set, collections.OrderedDict, nn.ReLU, nn.BatchNorm2d, nn.Sequential,
               nn.Conv2d, nn.Identity, nn.Linear, nn.AdaptiveAvgPool2d]
    with torch.serialization.safe_globals(allowed):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model = model.float().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# Same as SixDRepNet get_transforms(for_training=False)
TEACHER_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


@torch.no_grad()
def teacher_predict(model, images, device='cpu', batch=64):
    """images: list of RGB uint8 arrays or PIL images (head crops). Returns rotations (N, 3, 3) and
    euler angles in degrees (N, 3) as (pitch, yaw, roll)."""
    rots = []
    for i in range(0, len(images), batch):
        x = torch.stack([TEACHER_TF(im if isinstance(im, Image.Image) else Image.fromarray(im))
                         for im in images[i:i + batch]]).to(device)
        rots.append(model(x).float().cpu())
    R = torch.cat(rots) if rots else torch.zeros(0, 3, 3)
    return R.numpy(), (compute_euler(R) * 180 / np.pi).numpy()
