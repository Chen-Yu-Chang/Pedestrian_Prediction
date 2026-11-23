"""Rotation helpers. compute_rotation / compute_euler follow SixDRepNet (utils/util.py) so the
student's outputs are on exactly the same convention as the teacher's."""
import numpy as np
import torch


def compute_rotation(x6):
    """6D representation (B, 6) -> rotation matrix (B, 3, 3) via Gram-Schmidt (SixDRepNet)."""
    x = torch.nn.functional.normalize(x6[:, 0:3], dim=1, eps=1e-8)
    z = torch.nn.functional.normalize(torch.cross(x, x6[:, 3:6], dim=1), dim=1, eps=1e-8)
    y = torch.cross(z, x, dim=1)
    return torch.stack((x, y, z), dim=2)


def compute_euler(R):
    """Rotation matrix (B, 3, 3) -> (pitch, yaw, roll) in radians, SixDRepNet convention."""
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = (sy < 1e-6).float()
    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    out = torch.stack([x * (1 - singular) + xs * singular,
                       y,
                       z * (1 - singular)], dim=1)
    return out


def euler_to_rotation(pitch, yaw, roll):
    """Same construction as SixDRepNet Datasets.get_rotation: Rz(roll) @ Ry(yaw) @ Rx(pitch)."""
    cx, sx, cy, sy, cz, sz = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


_FLIP = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))


def flip_rotation(R):
    """Rotation of the horizontally mirrored image: F R F with F = diag(-1, 1, 1).
    Equivalent to negating yaw and roll, as SixDRepNet does for flipped training images."""
    F = _FLIP.to(R)
    return F @ R @ F


def geodesic_loss(R_pred, R_true, eps=1e-7):
    """Per-sample angle (radians) between two batches of rotations (SixDRepNet ComputeLoss).
    Take .mean() for a loss; the angle in degrees is a readable error measure."""
    m = torch.bmm(R_true, R_pred.transpose(1, 2))
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    return torch.acos(torch.clamp(cos, -1 + eps, 1 - eps))
