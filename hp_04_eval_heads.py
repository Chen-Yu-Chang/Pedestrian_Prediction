"""Step 4 - evaluate the head-pose student against the teacher (report Experiment 2).

A) Pose accuracy vs head size: AFLW2000 ground truth, faces degraded to 12 / 16 / 24 / 32 / 48 px.
   MAE (deg) of pitch / yaw / roll for the teacher and for the student checkpoints.
B) 'Is the pedestrian looking?' on JAAD test, split by head size (stored pixels):
     teacher yaw rule  (score = -|yaw|, i.e. more frontal = more likely looking)
     student yaw rule  (same rule on the student's yaw)
     student look head (stage 2)
   AUC per size bin and overall.

Usage: python hp_04_eval_heads.py [--device cuda]
Output: outputs_head/eval_aflw_by_size.csv, outputs_head/eval_jaad_look.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from headpose.config import HEAD_DIR, TEACHER_WEIGHTS
from headpose.degrade import degrade
from headpose.rotation import compute_euler
from headpose.student import HeadStudent, to_tensor
from headpose.teacher import load_teacher, teacher_predict

SIZES = [12, 16, 24, 32, 48]
BINS = [(0, 8), (8, 16), (16, 32), (32, 1e9)]


@torch.no_grad()
def student_predict(model, imgs, device, batch=512):
    model.eval()
    R, L = [], []
    for i in range(0, len(imgs), batch):
        r, l = model(to_tensor(imgs[i:i + batch]).to(device))
        R.append(r.cpu()); L.append(l.cpu())
    R = torch.cat(R)
    return (compute_euler(R) * 180 / np.pi).numpy(), torch.sigmoid(torch.cat(L)).numpy()


def load_students(device):
    out = {}
    for stage in (1, 2):
        p = os.path.join(HEAD_DIR, f'student_stage{stage}.pt')
        if os.path.exists(p):
            m = HeadStudent().to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            out[f'student_stage{stage}'] = m
    return out


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=TEACHER_WEIGHTS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)
    teacher, students = load_teacher(args.weights, dev), load_students(dev)

    # ---- A) AFLW2000 by size ----
    d = np.load(os.path.join(HEAD_DIR, 'faces_AFLW2000.npz'))
    gt, rows = d['gt_euler'], []
    for s in ['clean'] + SIZES:
        rng = np.random.default_rng(123)
        imgs = d['img'] if s == 'clean' else np.stack([degrade(im, rng, s, s, strength=0.5)[0] for im in d['img']])
        preds = {'teacher': teacher_predict(teacher, list(imgs), args.device)[1]}
        preds.update({k: student_predict(m, imgs, dev)[0] for k, m in students.items()})
        for k, e in preds.items():
            mae = np.abs(e - gt).mean(0)
            rows.append({'size_px': s, 'model': k, 'pitch': mae[0], 'yaw': mae[1], 'roll': mae[2], 'mean': mae.mean()})
    a = pd.DataFrame(rows)
    a.to_csv(os.path.join(HEAD_DIR, 'eval_aflw_by_size.csv'), index=False)
    print('\nAFLW2000 MAE (deg, mean of pitch/yaw/roll) by head size:')
    print(a.pivot(index='size_px', columns='model', values='mean').round(2).to_string())

    # ---- B) JAAD looking by size ----
    if not os.path.exists(os.path.join(HEAD_DIR, 'heads_test.pkl')):
        print('\nheads_test.pkl not found (JAAD heads not cropped yet): skipping the JAAD looking evaluation')
        return
    meta = pd.read_pickle(os.path.join(HEAD_DIR, 'heads_test.pkl'))
    imgs = np.load(os.path.join(HEAD_DIR, 'heads_test.npz'))['img']
    keep = np.where((meta.look.values >= 0) & meta.found.values)[0]
    m, imgs = meta.iloc[keep].reset_index(drop=True), imgs[keep]
    scores = {'teacher_yaw_rule': -np.abs(m.t_yaw.values)}
    for k, model in students.items():
        e, p = student_predict(model, imgs, dev)
        scores[f'{k}_yaw_rule'] = -np.abs(e[:, 1])
        if k == 'student_stage2':
            scores['student_stage2_look_head'] = p
    rows = []
    for lo, hi in BINS + [(0, 1e9)]:
        sel = (m.eff_px.values >= lo) & (m.eff_px.values < hi)
        name = 'all' if (lo, hi) == (0, 1e9) else (f'>={lo}px' if hi > 1e8 else f'{lo}-{hi}px')
        for k, sc in scores.items():
            rows.append({'head_size': name, 'n': int(sel.sum()), 'looking': float(m.look.values[sel].mean()) if sel.any() else np.nan,
                         'method': k, 'auc': safe_auc(m.look.values[sel], sc[sel]) if sel.any() else np.nan})
    b = pd.DataFrame(rows)
    order = list(dict.fromkeys(b.head_size))                      # keep bins in size order
    b['head_size'] = pd.Categorical(b.head_size, categories=order, ordered=True)
    b.to_csv(os.path.join(HEAD_DIR, 'eval_jaad_look.csv'), index=False)
    print('\nJAAD test, AUC for "looking" by head size (stored pixels):')
    print(b.pivot_table(index='head_size', columns='method', values='auc', observed=False, dropna=False).round(3).to_string())
    print(b.drop_duplicates('head_size')[['head_size', 'n', 'looking']].to_string(index=False))


if __name__ == '__main__':
    main()
