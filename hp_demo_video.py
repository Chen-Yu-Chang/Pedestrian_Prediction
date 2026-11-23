"""Render a demo video of the head-pose module on one JAAD pedestrian.

What the video shows (every frame of the original 30 fps clip):
  * the pedestrian box, green while the student says "looking", grey otherwise
  * an arrow from the head showing where the face points (student; teacher too with --teacher)
  * top-left: the 64x64 crop the student actually sees, and the head size in the video
  * top-right: P(looking) bar, JAAD's own label, and the time to the crossing decision
  * bottom: timeline of P(looking) with JAAD 'looking' frames shaded and the decision point marked

Head boxes come from the keypoints in pose_{split}.pkl (every 5th frame) and are interpolated in between;
the student and teacher run on every frame of the full-resolution clip.

Usage:
  python hp_demo_video.py --video 0123 --ped 0_123_456b [--teacher] [--split test]
Output: demo/demo_<video>_<ped>.mp4 plus a few PNG stills for slides
"""
import argparse
import os
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch

from headpose.config import DET_SIZE, HEAD_DIR, ORIG_H, ORIG_W, POSE_DIR, ROOT, STUDENT_SIZE, TEACHER_WEIGHTS, enhance_image
from headpose.head_crop import SRC_KEYPOINTS, crop_head, head_box, square
from headpose.jaad_look import load_look_labels
from headpose.student import HeadStudent
from headpose.teacher import load_teacher, teacher_predict
from hp_04_eval_heads import student_predict

W, H, STRIP = 1280, 720, 150          # output size: video area + timeline strip
GREEN, GREY, CYAN, ORANGE, RED, WHITE = (80, 200, 80), (170, 170, 170), (255, 200, 0), (0, 150, 255), (60, 60, 230), (255, 255, 255)


def face_arrow(img, x, y, pitch, yaw, size, color, thick=3):
    """Arrow along the face direction (the z axis of the usual head-pose drawing)."""
    p, yw = np.radians(pitch), np.radians(yaw)   # checked against SixDRepNet's own pose cube on real faces
    x2 = x + size * np.sin(yw)
    y2 = y - size * np.cos(yw) * np.sin(p)
    cv2.arrowedLine(img, (int(x), int(y)), (int(x2), int(y2)), color, thick, cv2.LINE_AA, tipLength=0.3)


def text(img, s, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def ema(x, a=0.3):
    out, m = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        m = a * v + (1 - a) * m
        out[i] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='e.g. 0123')
    ap.add_argument('--ped', required=True, help='pedestrian id, as listed by hp_pick_demo.py')
    ap.add_argument('--split', default='test')
    ap.add_argument('--clips', default=os.path.join(ROOT, 'data', 'JAAD_repo', 'JAAD_clips'))
    ap.add_argument('--ckpt', default=os.path.join(HEAD_DIR, 'student_stage2.pt'))
    ap.add_argument('--teacher', action='store_true', help='also draw the teacher arrow (orange)')
    ap.add_argument('--speed', type=float, default=1.0, help='0.5 = half speed')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)

    # ---- the pedestrian's sampled frames: boxes + keypoints ----
    pose = pd.read_pickle(os.path.join(POSE_DIR, f'pose_{args.split}.pkl'))
    g = pose[(pose.video_id.astype(str) == args.video) & (pose.ped_id.astype(str) == args.ped)].sort_values('frame')
    if g.empty:
        raise SystemExit(f'pedestrian {args.ped} not found in video {args.video} ({args.split} split)')
    F = g.frame.values.astype(int)
    sx, sy = ORIG_W / DET_SIZE, ORIG_H / DET_SIZE
    hb = np.array([head_box((r.x1, r.y1, r.x2, r.y2), r.kx, r.ky, r.kc)[:4] for r in g.itertuples()], float)
    pb = np.c_[g.x1 * sx, g.y1 * sy, g.x2 * sx, g.y2 * sy]
    frames = np.arange(F[0], F[-1] + 1)
    interp = lambda a: np.stack([np.interp(frames, F, a[:, j]) for j in range(a.shape[1])], 1)
    head, person = interp(hb[:, :3]), interp(pb)
    usable = np.interp(frames, F, (hb[:, 3] == SRC_KEYPOINTS).astype(float)) > 0.5
    dp, crosses = int(g.dp.iat[0]), int(g.label.iat[0])
    gt = load_look_labels(args.video)
    gt_look = np.array([gt.get((args.ped, int(f)), -1) for f in frames])

    # ---- read the clip, crop heads, run the models ----
    cap = cv2.VideoCapture(os.path.join(args.clips, f'video_{args.video}.mp4'))
    raw, crops, native, px, idx = [], [], [], [], 0
    while True:
        ok, img = cap.read()
        if not ok or idx > frames[-1]:
            break
        if idx >= frames[0]:
            k = idx - frames[0]
            enh = enhance_image(img)[..., ::-1]
            c, eff = crop_head(np.ascontiguousarray(enh), *head[k])
            c = np.ascontiguousarray(c)
            raw.append(img); native.append(c); crops.append(square(c, STUDENT_SIZE)); px.append(eff)
        idx += 1
    cap.release()
    if not raw:
        raise SystemExit(f'could not read frames from video_{args.video}.mp4 in {args.clips}')
    n = len(raw)
    frames, head, person, usable, gt_look = frames[:n], head[:n], person[:n], usable[:n], gt_look[:n]
    student = HeadStudent().to(dev)
    student.load_state_dict(torch.load(args.ckpt, map_location=dev))
    s_e, s_p = student_predict(student, np.stack(crops), dev)
    s_e, s_p = np.c_[ema(s_e[:, 0]), ema(s_e[:, 1]), ema(s_e[:, 2])], ema(s_p)
    t_e = None
    if args.teacher:
        t_e = teacher_predict(load_teacher(TEACHER_WEIGHTS, dev), native, args.device)[1]
        t_e = np.c_[ema(t_e[:, 0]), ema(t_e[:, 1])]

    # ---- render ----
    os.makedirs(os.path.join(ROOT, 'demo'), exist_ok=True)
    stem = os.path.join(ROOT, 'demo', f'demo_{args.video}_{args.ped}')
    fps = 30 * args.speed
    vw = cv2.VideoWriter(stem + '_raw.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H + STRIP))
    scale = W / raw[0].shape[1]
    t_x = lambda k: int(40 + (W - 80) * k / max(n - 1, 1))
    stills = sorted({int(n * q) for q in (0.2, 0.5, 0.8)} | ({int(dp - frames[0] - 30)} if frames[0] < dp - 30 < frames[-1] else set()))
    for k in range(n):
        canvas = np.zeros((H + STRIP, W, 3), np.uint8)
        fr = cv2.resize(raw[k], (W, int(raw[k].shape[0] * scale)))[:H]
        canvas[:fr.shape[0]] = fr
        looking = s_p[k] > 0.5 and usable[k]
        x1, y1, x2, y2 = (person[k] * scale * raw[k].shape[1] / ORIG_W).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN if looking else GREY, 3)
        cx, cy, sz = head[k] * scale * raw[k].shape[1] / ORIG_W
        cv2.rectangle(canvas, (int(cx - sz / 2), int(cy - sz / 2)), (int(cx + sz / 2), int(cy + sz / 2)), (0, 220, 255), 1)
        alen = max(40, (y2 - y1) * 0.35)
        if usable[k]:
            face_arrow(canvas, cx, cy, s_e[k, 0], s_e[k, 1], alen, CYAN, 3)
            if t_e is not None:
                face_arrow(canvas, cx, cy, t_e[k, 0], t_e[k, 1], alen, ORANGE, 2)
        # dark panels behind the text so it stays readable on any background
        for (a, b, c2, d) in [(10, 10, 180, 222), (W - 345, 10, W - 10, 198)]:
            roi = canvas[b:d, a:c2]
            canvas[b:d, a:c2] = (roi * 0.35).astype(np.uint8)
        # inset: what the model sees
        inset = cv2.resize(crops[k][..., ::-1], (150, 150), interpolation=cv2.INTER_NEAREST)
        canvas[40:190, 20:170] = inset
        cv2.rectangle(canvas, (20, 40), (170, 190), WHITE, 1)
        text(canvas, 'what the model sees', (20, 30), 0.55)
        text(canvas, f'head ~{px[k]:.0f} px in video', (20, 212), 0.5)
        # right panel
        bx = W - 330
        text(canvas, f'P(looking at car): {s_p[k]:.2f}', (bx, 40), 0.65, GREEN if looking else WHITE, 2)
        cv2.rectangle(canvas, (bx, 52), (bx + 300, 72), WHITE, 1)
        cv2.rectangle(canvas, (bx, 52), (bx + int(300 * s_p[k]), 72), GREEN if looking else GREY, -1)
        lab = {1: 'looking', 0: 'not looking', -1: 'n/a'}[int(gt_look[k])]
        text(canvas, f'JAAD label: {lab}', (bx, 100), 0.6)
        dt = (dp - frames[k]) / 30
        text(canvas, (f'decision in {dt:.1f} s' if dt > 0 else f'decision {-dt:.1f} s ago') +
             (' (crosses)' if crosses else ' (does not cross)'), (bx, 128), 0.6, RED if dt <= 0 else WHITE)
        text(canvas, 'arrow: student', (bx, 156), 0.55, CYAN)
        if t_e is not None:
            text(canvas, 'teacher', (bx + 160, 156), 0.55, ORANGE)
        if not usable[k]:
            text(canvas, 'face not visible', (bx, 184), 0.55, GREY)
        # timeline strip
        y0, y1s = H + 20, H + STRIP - 30
        bw = max(1, (W - 80) // max(n, 1) + 1)
        for j in range(n):
            if gt_look[j] == 1:
                cv2.rectangle(canvas, (t_x(j), y0), (t_x(j) + bw, y1s), (40, 110, 40), -1)
        pts = np.array([[t_x(j), int(y1s - (y1s - y0) * s_p[j])] for j in range(k + 1)], np.int32)
        cv2.polylines(canvas, [pts], False, CYAN, 2, cv2.LINE_AA)
        if frames[0] <= dp <= frames[-1]:
            xd = t_x(dp - frames[0])
            for yy in range(y0, y1s, 8):
                cv2.line(canvas, (xd, yy), (xd, min(yy + 4, y1s)), RED, 2)
            text(canvas, 'decision point', (xd + 6, y0 + 12), 0.45, RED)
        cv2.line(canvas, (t_x(k), y0 - 5), (t_x(k), y1s + 5), WHITE, 1)
        text(canvas, 'P(looking) over time    green = JAAD says looking', (40, H + STRIP - 8), 0.5)
        vw.write(canvas)
        if k in stills:
            cv2.imwrite(f'{stem}_still_{k:04d}.png', canvas)
    vw.release()

    # H.264 so the video plays in PowerPoint / Keynote / Google Slides
    ff = shutil.which('ffmpeg')
    if ff is None:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if ff:
        subprocess.run([ff, '-y', '-loglevel', 'error', '-i', stem + '_raw.mp4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '20', stem + '.mp4'], check=True)
        os.remove(stem + '_raw.mp4')
        print('saved', stem + '.mp4')
    else:
        print('saved', stem + '_raw.mp4', '(install ffmpeg or "pip install imageio-ffmpeg" for a slide-friendly H.264 copy)')
    print(f'{n} frames ({n / 30:.1f} s of video), stills: {stem}_still_*.png')


if __name__ == '__main__':
    main()
