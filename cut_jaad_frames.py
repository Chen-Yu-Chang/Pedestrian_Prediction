"""Cut full-resolution JAAD frames (1920x1080) from the original clips, for sharper head crops.

The stored frames used by the team are 640x640, stretched from 1920x1080, so a head is only about 1/3 of
its real width. This script extracts the SAME frames at full resolution and names them the way the rest
of the pipeline expects:   <out>/images/{split}/video_XXXX_fYYYY.jpg   (YYYY = original frame number)

Mode 1 (default): only the frames listed in outputs_pose/pose_{split}.pkl  -> small, exactly what hp_02 needs
Mode 2 (--every 5): every 5th frame of every video in a JAAD split file    -> e.g. to rerun pose_extract.py

Get the clips first (about 3 GB), inside the JAAD repo:  bash download_clips.sh   (creates JAAD_clips/*.mp4)

Usage:
  python cut_jaad_frames.py --clips /path/to/JAAD/JAAD_clips
  python cut_jaad_frames.py --clips /path/to/JAAD/JAAD_clips --every 5 --split_dir /path/to/JAAD/split_ids/default
Then:   FRAMES_DIR=data/jaad_fullres bash run_all.sh     (or hp_02 --frames_dir data/jaad_fullres)
"""
import argparse
import os
from collections import defaultdict

import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def wanted_from_pkl(pose_dir, splits):
    want = defaultdict(lambda: defaultdict(set))          # split -> video_id -> {frames}
    for s in splits:
        df = pd.read_pickle(os.path.join(pose_dir, f'pose_{s}.pkl'))
        for v, f in zip(df.video_id, df.frame):
            want[s][str(v)].add(int(f))
    return want


def wanted_every(split_dir, splits, every):
    want = defaultdict(lambda: defaultdict(set))
    for s in splits:
        for line in open(os.path.join(split_dir, f'{s}.txt')):
            if line.strip():
                want[s][line.strip().replace('video_', '')] = None   # None = every `every` frames
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True, help='folder with video_XXXX.mp4 (JAAD_clips)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'jaad_fullres'))
    ap.add_argument('--pose_dir', default=os.path.join(ROOT, 'outputs_pose'))
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--every', type=int, default=0, help='cut every Nth frame of each video instead of the pkl frames')
    ap.add_argument('--split_dir', default=None, help='JAAD split_ids/default (needed with --every)')
    ap.add_argument('--quality', type=int, default=95)
    args = ap.parse_args()

    want = wanted_every(args.split_dir, args.splits, args.every) if args.every else wanted_from_pkl(args.pose_dir, args.splits)
    for s in args.splits:
        out_dir = os.path.join(args.out, 'images', s)
        os.makedirs(out_dir, exist_ok=True)
        n_saved, missing = 0, []
        for v_id, frames in sorted(want[s].items()):
            path = os.path.join(args.clips, f'video_{v_id}.mp4')
            if not os.path.exists(path):
                missing.append(v_id); continue
            cap, idx = cv2.VideoCapture(path), 0
            last = max(frames) if frames else None
            while True:
                ok, img = cap.read()                 # read sequentially: frame numbers stay exact
                if not ok or (last is not None and idx > last):
                    break
                if (frames is None and idx % args.every == 0) or (frames is not None and idx in frames):
                    cv2.imwrite(os.path.join(out_dir, f'video_{v_id}_f{idx:04d}.jpg'), img,
                                [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    n_saved += 1
                idx += 1
            cap.release()
        print(f'{s}: saved {n_saved} frames to {out_dir}' + (f' | clips not found: {missing[:5]}...' if missing else ''),
              flush=True)


if __name__ == '__main__':
    main()
