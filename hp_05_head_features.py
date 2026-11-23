"""Step 5 - per-frame head features for the crossing model.

Runs the trained student on every JAAD head crop and writes outputs_head/head_{split}.pkl with one row per
row of pose_{split}.pkl (same video_id, ped_id, frame keys):
  h_yaw, h_pitch, h_roll : student angles in degrees (0 when the head is not usable)
  h_look                 : student P(looking)        (stage-2 checkpoint; NaN with a stage-1 checkpoint)
  h_ok                   : 1 if the head is at least --min_px stored pixels and (with keypoint placement)
                           the head box came from face keypoints
  h_px                   : head size in stored pixels

Usage: python hp_05_head_features.py [--ckpt outputs_head/student_stage2.pt] [--min_px 4]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from headpose.config import HEAD_DIR
from headpose.head_crop import SRC_KEYPOINTS
from headpose.student import HeadStudent
from hp_04_eval_heads import student_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(HEAD_DIR, 'student_stage2.pt'))
    ap.add_argument('--min_px', type=float, default=4.0)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = HeadStudent().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    has_look = 'stage2' in os.path.basename(args.ckpt)

    for split in args.splits:
        meta = pd.read_pickle(os.path.join(HEAD_DIR, f'heads_{split}.pkl'))
        imgs = np.load(os.path.join(HEAD_DIR, f'heads_{split}.npz'))['img']
        e, p = student_predict(model, imgs, dev)
        placed = (meta.src.values == SRC_KEYPOINTS) | np.asarray(meta.get('head_from', 'keypoints') == 'box')
        ok = meta.found.values & placed & (meta.eff_px.values >= args.min_px)
        out = meta[['video_id', 'ped_id', 'frame']].copy()
        out['h_pitch'], out['h_yaw'], out['h_roll'] = (np.where(ok, e[:, j], 0.0) for j in range(3))
        out['h_look'] = np.where(ok, p, 0.0) if has_look else np.nan
        out['h_ok'] = ok.astype(np.float32)
        out['h_px'] = meta.eff_px.values
        out.to_pickle(os.path.join(HEAD_DIR, f'head_{split}.pkl'))
        print(f'head_{split}.pkl: {len(out)} rows, usable heads {ok.mean():.0%}, '
              f'mean P(look) {np.nanmean(out.h_look[ok]) if has_look and ok.any() else float("nan"):.2f}')


if __name__ == '__main__':
    main()
