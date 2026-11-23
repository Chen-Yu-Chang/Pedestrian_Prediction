"""Step 2 - crop every pedestrian head in JAAD and label it.

Input : outputs_pose/pose_{split}.pkl (from pose_extract.py: one row per video, pedestrian, frame with the
        detector box and 17 YOLO26 keypoints in 640x640 coords) and the stored JAAD frames.
For each row:
  1. head box, in original 1920x1080 coords (headpose/head_crop.py). By default it is placed with the nose /
     eye / ear keypoints already in pose_{split}.pkl; these keypoints only say WHERE the head is and are not
     fed to the crossing model. --head_from box places it at the top of the pedestrian box instead.
  2. crop it from the (CLAHE-enhanced, like the rest of the pipeline) frame and resize to 64x64
  3. JAAD 'look' label for that pedestrian and frame (1 looking, 0 not-looking, -1 not annotated)
  4. teacher (SixDRepNet) angles on the native crop; rows whose head is at least TEACHER_MIN_PX stored
     pixels are flagged `teacher_ok` and can be used as pseudo-labels for the student

Usage: python hp_02_extract_jaad_heads.py [--splits train val test] [--frames_dir DIR] [--head_from keypoints|box] [--no_clahe]
  --frames_dir  folder holding images/{split}/video_XXXX_fYYYY.jpg at any 16:9 resolution, e.g. full-res
                1920x1080 frames extracted from the JAAD clips (sharper heads). Default: the stored frames.
Output: outputs_head/heads_{split}.npz (img: N x 64 x 64 x 3, aligned with the rows of heads_{split}.pkl)
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

from headpose.config import (BASE_PATH, HEAD_DIR, POSE_DIR, STUDENT_SIZE, TEACHER_MIN_PX, TEACHER_WEIGHTS,
                             enhance_image)
from headpose.head_crop import crop_head, head_box, square
from headpose.jaad_look import load_look_labels
from headpose.teacher import load_teacher, teacher_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--frames_dir', default=BASE_PATH)
    ap.add_argument('--no_clahe', action='store_true')
    ap.add_argument('--head_from', choices=['keypoints', 'box'], default='keypoints',
                    help='locate the head with face keypoints (more accurate) or from the top of the box only')
    ap.add_argument('--weights', default=TEACHER_WEIGHTS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    teacher = load_teacher(args.weights, args.device)

    for split in args.splits:
        df = pd.read_pickle(os.path.join(POSE_DIR, f'pose_{split}.pkl')).reset_index(drop=True)
        img_dir = os.path.join(args.frames_dir, 'images', split)
        files = {}
        for f in os.listdir(img_dir):
            if f.startswith('video_') and f.count('_') >= 2:
                files[(f.split('_')[1], int(f.split('_')[2][1:5]))] = f
        crops = np.zeros((len(df), STUDENT_SIZE, STUDENT_SIZE, 3), np.uint8)
        meta = {k: np.zeros(len(df), t) for k, t in
                [('eff_px', np.float32), ('src', np.int8), ('look', np.int8), ('found', bool)]}
        t_euler = np.full((len(df), 3), np.nan, np.float32)
        t_R = np.full((len(df), 9), np.nan, np.float32)
        t0 = time.time()
        for vi, (v_id, g) in enumerate(df.groupby('video_id', sort=False)):
            looks = load_look_labels(v_id)
            for frame_no, gf in g.groupby('frame', sort=False):
                fname = files.get((v_id, int(frame_no)))
                if fname is None:
                    continue
                img = cv2.imread(os.path.join(img_dir, fname))
                if img is None:
                    continue
                img = img if args.no_clahe else enhance_image(img)
                rgb = img[..., ::-1]
                native, idx = [], []
                for i, r in gf.iterrows():
                    kc = r.kc if args.head_from == 'keypoints' else np.zeros(17)   # zeros -> box-only placement
                    cx, cy, size, src = head_box((r.x1, r.y1, r.x2, r.y2), r.kx, r.ky, kc)
                    crop, eff = crop_head(rgb, cx, cy, size)
                    crops[i] = square(np.ascontiguousarray(crop), STUDENT_SIZE)
                    meta['eff_px'][i], meta['src'][i], meta['found'][i] = eff, src, True
                    meta['look'][i] = looks.get((r.ped_id, int(frame_no)), -1)
                    native.append(np.ascontiguousarray(crop)); idx.append(i)
                R, e = teacher_predict(teacher, native, args.device)
                t_R[idx], t_euler[idx] = R.reshape(-1, 9), e
            if (vi + 1) % 20 == 0:
                print(f'  {split}: {vi + 1} videos, {time.time() - t0:.0f}s', flush=True)

        out = df[['video_id', 'ped_id', 'frame', 'label', 'dp']].copy()
        for k, v in meta.items():
            out[k] = v
        out['head_from'] = args.head_from
        out['teacher_ok'] = out.found & (out.eff_px >= TEACHER_MIN_PX)
        out[['t_pitch', 't_yaw', 't_roll']] = t_euler
        out['t_R'] = list(t_R)
        out.to_pickle(os.path.join(HEAD_DIR, f'heads_{split}.pkl'))
        np.savez_compressed(os.path.join(HEAD_DIR, f'heads_{split}.npz'), img=crops)
        lk = out.look.values
        print(f'Saved heads_{split}: {out.found.sum()}/{len(out)} crops | median head {out.eff_px.median():.1f} px '
              f'(stored) | from keypoints {np.mean(out.src == 2):.0%} | look labels {np.mean(lk >= 0):.0%} '
              f'(looking {np.mean(lk[lk >= 0] == 1) if (lk >= 0).any() else 0:.0%}) | teacher_ok {out.teacher_ok.mean():.0%}',
              flush=True)


if __name__ == '__main__':
    main()
