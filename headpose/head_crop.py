"""Locate and crop the head of a pedestrian from YOLO26-pose keypoints.

Keypoints and boxes in pose_{split}.pkl are in 640x640 detector coordinates (frames stretched from
1920x1080). The head box is computed in original 1920x1080 coordinates, where heads are square, and
then cut out of whatever frame is on disk (the stored 640x640 frame or a full-resolution frame).

COCO keypoints used: 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear, 5/6 shoulders.
"""
import cv2
import numpy as np

from .config import DET_SIZE, KP_CONF, ORIG_H, ORIG_W

HEAD_KP, SHOULDERS = [0, 1, 2, 3, 4], [5, 6]
SRC_KEYPOINTS, SRC_PARTIAL, SRC_BOX = 2, 1, 0   # how the head box was found
CONTEXT = 1.3                                   # crop side = CONTEXT x head size (hair, chin)


def head_box(box, kx, ky, kc, conf=KP_CONF):
    """box: (x1, y1, x2, y2), kx/ky/kc: (17,), all in 640x640 coords.
    Returns (cx, cy, size, source) in original 1920x1080 coords; size ~ head height."""
    sx, sy = ORIG_W / DET_SIZE, ORIG_H / DET_SIZE
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    bh = max(y2 - y1, 1.0)
    X, Y, vis = np.asarray(kx) * sx, np.asarray(ky) * sy, np.asarray(kc) >= conf
    lo, hi = 0.08 * bh, 0.25 * bh                 # a head is roughly 1/8 to 1/4 of body height
    hv = [i for i in HEAD_KP if vis[i]]
    if len(hv) >= 2:
        spread = max(np.ptp(X[hv]), np.ptp(Y[hv]))
        size = float(np.clip(max(2.0 * spread, 0.12 * bh), lo, hi))
        cx, cy = float(X[hv].mean()), float(Y[hv].mean()) - 0.05 * size
        return cx, cy, size, SRC_KEYPOINTS
    size = 0.14 * bh
    if len(hv) == 1:
        return float(X[hv[0]]), float(Y[hv[0]]), size, SRC_PARTIAL
    if vis[5] and vis[6]:
        sw = abs(X[5] - X[6])
        return float((X[5] + X[6]) / 2), float((Y[5] + Y[6]) / 2 - max(0.6 * sw, 0.5 * size)), size, SRC_PARTIAL
    return (x1 + x2) / 2, y1 + 0.08 * bh, size, SRC_BOX


def crop_head(frame, cx, cy, size, context=CONTEXT):
    """Cut the square head region (original coords) out of `frame` (any resolution of the 16:9 video,
    stretched or not). Out-of-frame parts are filled by edge replication.
    Returns (crop as stored on disk, effective resolution in stored pixels = min side)."""
    h_img, w_img = frame.shape[:2]
    sx, sy = w_img / ORIG_W, h_img / ORIG_H
    half = context * size / 2
    x0, x1 = int(round((cx - half) * sx)), int(round((cx + half) * sx))
    y0, y1 = int(round((cy - half) * sy)), int(round((cy + half) * sy))
    x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
    pad = max(0, -x0, -y0, x1 - w_img, y1 - h_img)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, x1, y0, y1 = x0 + pad, x1 + pad, y0 + pad, y1 + pad
    crop = frame[y0:y1, x0:x1]
    return crop, min(size * sx, size * sy)


def square(crop, out):
    """Resize a crop to out x out. For crops from stretched frames this also undoes the stretch."""
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_LINEAR if crop.shape[0] < out else cv2.INTER_AREA)
