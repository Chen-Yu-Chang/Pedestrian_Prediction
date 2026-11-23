"""Look at the face crops the student is trained and tested on.

Saves outputs_head/crops_check.png: rows of 12 random crops each
  1) 300W-LP training crops   2) the same at 32 px   3) AFLW2000 test crops   4) the same at 32 px
and prints how many AFLW2000 .mat files contain negative (invisible) landmarks, which make the crop box
too large if they are not ignored.

Usage: python hp_show_crops.py
"""
import glob
import os

import cv2
import numpy as np
import scipy.io as sio

from headpose.config import HEAD_DIR, HPE_DATA
from headpose.degrade import degrade


def main():
    rng = np.random.default_rng(0)
    rows = []
    for name in ['300W_LP', 'AFLW2000']:
        img = np.load(os.path.join(HEAD_DIR, f'faces_{name}.npz'))['img']     # read once
        idx = rng.choice(len(img), 12, replace=False)
        clean = [cv2.resize(img[i], (96, 96)) for i in idx]
        small = [cv2.resize(degrade(img[i], rng, 32, 32, strength=0.5)[0], (96, 96)) for i in idx]
        rows += [np.hstack(clean), np.hstack(small)]
    out = os.path.join(HEAD_DIR, 'crops_check.png')
    cv2.imwrite(out, np.vstack(rows)[..., ::-1])
    print('saved', out, '(rows: 300W-LP, 300W-LP at 32px, AFLW2000, AFLW2000 at 32px)')

    mats = sorted(glob.glob(os.path.join(HPE_DATA, 'AFLW2000', '*.mat')))
    neg = sum(int((sio.loadmat(m)['pt2d'] < 0).any()) for m in mats)
    print(f'AFLW2000: {neg} of {len(mats)} files have negative (invisible) landmarks in pt2d')


if __name__ == '__main__':
    main()
