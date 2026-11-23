"""Step 1 - crop 300W-LP / AFLW2000 faces and label them with the frozen SixDRepNet teacher.

For every face image we keep
  * a clean head crop (CROP_STORE x CROP_STORE, uint8) that hp_03 later degrades for the student
  * the teacher's rotation matrix on the clean, full-resolution crop  (distillation target)
  * the ground-truth rotation and Euler angles from the .mat file   (only used for evaluation, or
    mixed in with --gt_weight in hp_03)
Crops follow SixDRepNet's test-time crop (landmark box enlarged with k = 0.2); faces with any
|angle| > 99 deg are dropped, as in SixDRepNet.

Expected layout (SixDRepNet's):  data/HPE/300W_LP/**/*.jpg + .mat,  data/HPE/AFLW2000/*.jpg + .mat

Usage: python hp_01_label_faces.py [--sets 300W_LP AFLW2000] [--limit N] [--device cuda]
Output: outputs_head/faces_{set}.npz
"""
import argparse
import os

import cv2

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from tqdm import tqdm

from headpose.config import HEAD_DIR, HPE_DATA, TEACHER_WEIGHTS
from headpose.rotation import euler_to_rotation
from headpose.teacher import load_teacher, teacher_predict

VALID_ONLY = True   # set by --raw_crop: False reproduces SixDRepNet's original crop exactly
CROP_STORE = 80   # stored clean crop size; degradations go down to 12-48 px, so 80 keeps enough detail


def list_faces(root):
    out = []
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith('.jpg') and os.path.exists(os.path.join(d, f[:-4] + '.mat')):
                out.append(os.path.join(d, f[:-4]))
    return sorted(out)


def load_face(stem):
    """SixDRepNet's test-time crop box (landmark box enlarged with k = 0.2). By default two fixes: AFLW2000 uses its
    68-point landmarks like 300W-LP (and never the invisible ones), and areas outside the photo repeat the
    edge instead of being black. --raw_crop reproduces SixDRepNet exactly. Returns (RGB PIL crop, pitch, yaw, roll)."""
    image = Image.open(stem + '.jpg').convert('RGB')
    mat = sio.loadmat(stem + '.mat')
    pt2d = mat['pt2d']
    if VALID_ONLY and 'pt3d_68' in mat:   # AFLW2000: use its 68-point landmarks, the same points 300W-LP uses
        pt2d = mat['pt3d_68'][:2]
    if VALID_ONLY:   # AFLW2000 marks invisible landmarks with negative coordinates; they would stretch the box
        ok = (pt2d[0] >= 0) & (pt2d[1] >= 0)
        if ok.sum() >= 3:
            pt2d = pt2d[:, ok]
    x_min, y_min, x_max, y_max = pt2d[0].min(), pt2d[1].min(), pt2d[0].max(), pt2d[1].max()
    k = 0.20
    x_min -= 2 * k * abs(x_max - x_min)
    y_min -= 2 * k * abs(y_max - y_min)
    x_max += 2 * k * abs(x_max - x_min)
    y_max += 0.6 * k * abs(y_max - y_min)
    if VALID_ONLY:   # fill parts outside the photo by repeating the edge, not with black (JAAD crops do the same)
        arr = np.asarray(image)
        x0, y0, x1, y1 = int(x_min), int(y_min), int(x_max), int(y_max)
        pad = max(0, -x0, -y0, x1 - arr.shape[1], y1 - arr.shape[0])
        if pad:
            arr = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crop = Image.fromarray(arr[y0 + pad:y1 + pad, x0 + pad:x1 + pad])
    else:
        crop = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    pitch, yaw, roll = mat['Pose_Para'][0][:3]
    return crop, float(pitch), float(yaw), float(roll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['300W_LP', 'AFLW2000'])
    ap.add_argument('--data', default=HPE_DATA)
    ap.add_argument('--weights', default=TEACHER_WEIGHTS)
    ap.add_argument('--limit', type=int, default=0, help='random subset size per set (0 = all)')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--raw_crop', action='store_true',
                    help="use all landmarks for the crop box, like SixDRepNet's evaluation (default: ignore negative ones)")
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    global VALID_ONLY
    VALID_ONLY = not args.raw_crop
    teacher = load_teacher(args.weights, args.device)
    rng = np.random.default_rng(0)
    for name in args.sets:
        stems = list_faces(os.path.join(args.data, name))
        if args.limit and len(stems) > args.limit:
            stems = sorted(rng.choice(stems, args.limit, replace=False).tolist())
        imgs, gt_e, gt_R, t_R, t_e, kept = [], [], [], [], [], []
        for i in tqdm(range(0, len(stems), args.batch), desc=name):
            crops = []
            for s in stems[i:i + args.batch]:
                crop, p, y, r = load_face(s)
                if max(abs(np.degrees([p, y, r]))) > 99:
                    continue
                crops.append(crop)
                imgs.append(np.asarray(crop.resize((CROP_STORE, CROP_STORE), Image.BICUBIC)))
                gt_e.append(np.degrees([p, y, r]))
                gt_R.append(euler_to_rotation(p, y, r))
                kept.append(os.path.relpath(s, args.data))
            if crops:
                R, e = teacher_predict(teacher, crops, args.device)
                t_R.append(R); t_e.append(e)
        t_R, t_e, gt_e = np.concatenate(t_R), np.concatenate(t_e), np.array(gt_e)
        mae = np.abs(t_e - gt_e).mean(0)
        print(f'{name}: {len(imgs)} faces | teacher vs GT on clean crops, MAE pitch/yaw/roll = '
              f'{mae[0]:.2f} / {mae[1]:.2f} / {mae[2]:.2f} deg', flush=True)
        np.savez_compressed(os.path.join(HEAD_DIR, f'faces_{name}.npz'), img=np.stack(imgs).astype(np.uint8),
                            teacher_R=t_R.astype(np.float32), teacher_euler=t_e.astype(np.float32),
                            gt_R=np.array(gt_R, np.float32), gt_euler=gt_e.astype(np.float32), stem=np.array(kept))


if __name__ == '__main__':
    main()
