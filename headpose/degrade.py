"""Make clear face crops look like JAAD pedestrian heads.

JAAD heads are small (the stored frames are 640x640, stretched from 1920x1080, so a 40 px tall head in
the original video is about 13 x 24 px in the stored frame) and suffer from blur, motion blur, JPEG
artefacts and uneven light. `degrade` reproduces these effects on a clean crop and returns an image
at the student's input size, so training and test inputs go through the same final resize.
"""
import cv2
import numpy as np

from .config import STUDENT_SIZE


def _motion_kernel(length, angle):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    return k / max(k.sum(), 1e-6)


def degrade(img, rng, min_px=12, max_px=48, out=STUDENT_SIZE, strength=1.0):
    """img: RGB uint8 (H, W, 3) clean head crop. rng: np.random.Generator.
    Returns (degraded RGB uint8 (out, out, 3), simulated head size in pixels)."""
    # 1) resolution loss: shrink to a JAAD-like size, with a non-square aspect like the stretched frames
    px = int(rng.integers(min_px, max_px + 1))
    aspect = rng.uniform(0.55, 1.0) if rng.random() < 0.5 else 1.0   # width shrink from 1920->640 stretch
    small = cv2.resize(img, (max(int(px * aspect), 4), px),
                       interpolation=[cv2.INTER_AREA, cv2.INTER_LINEAR][int(rng.integers(2))])

    # 2) optical / motion blur at the small scale
    if rng.random() < 0.5 * strength:
        small = cv2.GaussianBlur(small, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3 * strength:
        small = cv2.filter2D(small, -1, _motion_kernel(int(rng.integers(2, 4)) * 2 - 1, rng.uniform(0, 180)))

    # 3) light: brightness, contrast, gamma, optional CLAHE as in the JAAD pipeline
    f = small.astype(np.float32) / 255.0
    if rng.random() < 0.8 * strength:
        f = (f - 0.5) * rng.uniform(0.6, 1.3) + 0.5 + rng.uniform(-0.25, 0.2)
        f = np.clip(f, 0, 1) ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.5 * strength:
        f = f + rng.normal(0, rng.uniform(0.005, 0.04), f.shape)
    small = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.5:
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2)).apply(lab[..., 0])
        small = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 4) compression
    if rng.random() < 0.7 * strength:
        ok, enc = cv2.imencode('.jpg', small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        if ok:
            small = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]

    # 5) back to the student input size (same final step as for real JAAD crops)
    return cv2.resize(np.ascontiguousarray(small), (out, out), interpolation=cv2.INTER_LINEAR), px


def to_student_size(img, out=STUDENT_SIZE):
    """Final resize for real JAAD crops (no synthetic degradation)."""
    return cv2.resize(img, (out, out), interpolation=cv2.INTER_LINEAR)
