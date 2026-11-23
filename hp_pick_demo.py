"""Find good JAAD pedestrians for the demo video.

Ranks pedestrians in a split (default: test, so the student never trained on them) by how well they show
the head-pose module: big enough heads, a moment where they look at the car, and a student that agrees
with JAAD's own 'looking' labels.

Prints two short lists:
  CROSSES  - pedestrians who cross (label 1), ideally looking before they step out
  STAYS    - pedestrians who do not cross (label 0) but still look, the hard case
Then render one with:  python hp_demo_video.py --video XXXX --ped YYYY

Usage: python hp_pick_demo.py [--split test] [--top 8]
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from headpose.config import HEAD_DIR
from headpose.head_crop import SRC_KEYPOINTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--min_px', type=float, default=20, help='minimum median head size (pixels in the frames used)')
    args = ap.parse_args()

    meta = pd.read_pickle(os.path.join(HEAD_DIR, f'heads_{args.split}.pkl'))
    feat = pd.read_pickle(os.path.join(HEAD_DIR, f'head_{args.split}.pkl'))
    df = meta.merge(feat[['video_id', 'ped_id', 'frame', 'h_look', 'h_yaw', 'h_ok']], on=['video_id', 'ped_id', 'frame'])
    rows = []
    for (v, p), g in df.groupby(['video_id', 'ped_id']):
        g = g.sort_values('frame')
        lab = g.look.values
        before = g[g.frame <= g.dp.iat[0]]
        good = lab >= 0
        auc = roc_auc_score(lab[good], g.h_look.values[good]) if good.any() and len(np.unique(lab[good])) > 1 else np.nan
        acc = np.mean((g.h_look.values[good] > 0.5) == (lab[good] == 1)) if good.any() else np.nan
        rows.append({'video': v, 'ped': p, 'crosses': int(g.label.iat[0]), 'frames': f'{g.frame.min()}-{g.frame.max()}',
                     'seconds': round((g.frame.max() - g.frame.min()) / 30, 1), 'decision_frame': int(g.dp.iat[0]),
                     'head_px': round(float(g.eff_px.median()), 1),
                     'usable': round(float(((g.src == SRC_KEYPOINTS) & g.found).mean()), 2),
                     'looks_before_dp': round(float((before.look == 1).mean()) if len(before) else 0.0, 2),
                     'look_frac': round(float((lab == 1).mean()), 2), 'student_auc': round(auc, 2),
                     'student_acc': round(acc, 2)})
    r = pd.DataFrame(rows)
    ok = (r.head_px >= args.min_px) & (r.seconds >= 3) & (r.usable >= 0.6) & (r.look_frac > 0) & (r.look_frac < 1)
    r = r[ok].copy()
    # prefer: looks before the decision point, student agrees with JAAD, bigger heads
    r['score'] = (2 * r.looks_before_dp + r.student_acc.fillna(0) + r.student_auc.fillna(0.5)
                  + np.clip(r.head_px / 60, 0, 1))
    cols = ['video', 'ped', 'frames', 'seconds', 'decision_frame', 'head_px', 'looks_before_dp', 'student_acc', 'student_auc']
    for name, lab in [('CROSSES (label 1)', 1), ('STAYS (label 0) but looks', 0)]:
        print(f'\n{name}:')
        print(r[r.crosses == lab].sort_values('score', ascending=False).head(args.top)[cols].to_string(index=False))
    print('\nRender one:  python hp_demo_video.py --video <video> --ped <ped>')


if __name__ == '__main__':
    main()
