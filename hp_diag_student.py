"""Diagnose the stage-1 student: is it reading the face, or predicting roughly the same pose for everyone?

Prints, for AFLW2000 (real faces, never trained on) and for held-out 300W-LP identities (same kind of
images as training, but people the student never saw):
  * MAE of the student, of the teacher, and of a 'constant' predictor that always outputs the median
    300W-LP pose (the error you get without looking at the image)
  * spread (std) of the student's predicted yaw vs the true yaw spread
  * correlation between predicted and true yaw (0 = ignores the face, 1 = perfect ordering)

Usage: python hp_diag_student.py [--ckpt outputs_head/student_stage1.pt] [--size 32]
"""
import argparse
import os

import numpy as np
import torch

from headpose.config import HEAD_DIR, TEACHER_WEIGHTS
from headpose.degrade import degrade
from headpose.student import HeadStudent
from headpose.teacher import load_teacher, teacher_predict
from hp_03_train_student import identity_split
from hp_04_eval_heads import student_predict


def report(name, gt, preds):
    print(f'\n{name}  (n={len(gt)}; true yaw std {gt[:, 1].std():.1f} deg)')
    print(f"  {'model':<12}{'pitch':>8}{'yaw':>8}{'roll':>8}{'yaw std':>10}{'yaw corr':>10}")
    for k, e in preds.items():
        mae = np.abs(e - gt).mean(0)
        corr = np.corrcoef(e[:, 1], gt[:, 1])[0, 1] if e[:, 1].std() > 0.1 else 0.0
        print(f'  {k:<12}{mae[0]:8.1f}{mae[1]:8.1f}{mae[2]:8.1f}{e[:, 1].std():10.1f}{corr:10.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(HEAD_DIR, 'student_stage1.pt'))
    ap.add_argument('--size', type=int, default=32, help='simulated head size in pixels')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)
    student = HeadStudent().to(dev)
    student.load_state_dict(torch.load(args.ckpt, map_location=dev))
    teacher = load_teacher(TEACHER_WEIGHTS, dev)

    lp_npz = os.path.join(HEAD_DIR, 'faces_300W_LP.npz')
    lp = np.load(lp_npz)
    const = np.median(lp['gt_euler'], 0)
    _, ho = identity_split(lp_npz)
    sets = {'AFLW2000 (real faces)': (np.load(os.path.join(HEAD_DIR, 'faces_AFLW2000.npz')), None),
            '300W-LP held-out identities': (lp, ho[:args.n])}
    for name, (d, idx) in sets.items():
        all_img = d['img']                                   # read once
        idx = np.arange(min(len(all_img), args.n)) if idx is None else idx
        clean, gt = all_img[idx], d['gt_euler'][idx]
        rng = np.random.default_rng(0)
        small = np.stack([degrade(im, rng, args.size, args.size, strength=0.5)[0] for im in clean])
        for tag, imgs in [('clean', clean), (f'{args.size}px', small)]:
            preds = {'student': student_predict(student, imgs, dev)[0],
                     'teacher': teacher_predict(teacher, list(imgs), args.device)[1],
                     'constant': np.tile(const, (len(gt), 1))}
            report(f'{name}, {tag}', gt, preds)

    print('\nHow to read this:')
    print('  student ~ constant and yaw corr near 0  -> the student ignores the face (collapsed)')
    print('  good on 300W-LP held-out, bad on AFLW2000 -> it learned 300W-LP-specific cues (domain gap)')
    print('  bad on both                               -> it is under-trained or the task is too hard at this size')


if __name__ == '__main__':
    main()
