"""Step 3 - train the head-pose student from scratch by distilling the SixDRepNet teacher.

Stage 1 (pose distillation)
  input  : 300W-LP clean crops, degraded on the fly to 12-48 px JAAD-like heads (headpose/degrade.py)
           + optionally JAAD heads large enough for the teacher (--jaad_pseudo), used as they are
  target : the teacher's rotation on the *clean* crop      loss = geodesic angle(student, teacher)
           (+ --gt_weight x geodesic angle to the 300W-LP ground truth, off by default)
  val    : AFLW2000 ground truth, degraded to 16 / 24 / 32 / 48 px  -> MAE in degrees, best epoch kept

Stage 2 (looking fine-tune), starts from the stage-1 weights
  input  : JAAD train head crops with a 'look' label
  loss   : class-weighted BCE on look  + rot_weight x geodesic to the teacher where teacher_ok
           + replay_weight x stage-1 distillation loss on a 300W-LP batch (so pose is not forgotten)
  val    : AUC of the look head on JAAD val, best epoch kept

Usage:
  python hp_03_train_student.py --stage 1 [--epochs 40] [--jaad_pseudo]
  python hp_03_train_student.py --stage 2 [--epochs 15]
Output: outputs_head/student_stage{1,2}.pt, outputs_head/train_stage{1,2}.csv
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from headpose.config import HEAD_DIR, STUDENT_SIZE
from headpose.degrade import degrade
from headpose.rotation import compute_euler, flip_rotation, geodesic_loss
from headpose.student import HeadStudent, to_tensor

VAL_SIZES = [16, 24, 32, 48]


def _rng():
    return np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))


class FaceDistill(Dataset):
    """300W-LP clean crops -> degraded student input; target = teacher rotation (and GT)."""

    def __init__(self, npz, idx=None):
        d = np.load(npz)
        self.img, self.R_t, self.R_gt = d['img'], torch.tensor(d['teacher_R']), torch.tensor(d['gt_R'])
        self.idx = np.arange(len(self.img)) if idx is None else np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        rng, i = _rng(), self.idx[k]
        img, R_t, R_gt = self.img[i], self.R_t[i], self.R_gt[i]
        # random zoom in / out and shift, so the student copes with loose or tight head boxes (JAAD heads
        # are placed from keypoints, so their framing varies)
        n = img.shape[0]
        c = int(n * rng.uniform(0.8, 1.3))
        if c > n:
            p = c - n
            img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE)
            n = img.shape[0]
        y0, x0 = int(rng.integers(0, n - c + 1)), int(rng.integers(0, n - c + 1))
        img = img[y0:y0 + c, x0:x0 + c]
        if rng.random() < 0.5:
            img, R_t, R_gt = img[:, ::-1], flip_rotation(R_t), flip_rotation(R_gt)
        # mostly JAAD-like small heads; 20% larger, sharper ones so clear inputs are not out of distribution
        img, _ = degrade(np.ascontiguousarray(img), rng, max_px=80 if rng.random() < 0.2 else 48)
        return img, R_t, R_gt


class JaadHeads(Dataset):
    """JAAD head crops (already small and blurry). Returns image, teacher rotation, teacher mask, look."""

    def __init__(self, split, rows=None):
        self.meta = pd.read_pickle(os.path.join(HEAD_DIR, f'heads_{split}.pkl'))
        self.img = np.load(os.path.join(HEAD_DIR, f'heads_{split}.npz'))['img']
        self.idx = np.arange(len(self.meta)) if rows is None else np.asarray(rows)
        R = np.stack(self.meta.t_R.values).reshape(-1, 3, 3)
        self.R = torch.tensor(np.nan_to_num(R), dtype=torch.float32)
        self.ok = torch.tensor(self.meta.teacher_ok.values, dtype=torch.bool)
        self.look = torch.tensor(self.meta.look.values, dtype=torch.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        rng, i = _rng(), self.idx[k]
        img, R = self.img[i], self.R[i]
        if rng.random() < 0.5:
            img, R = img[:, ::-1], flip_rotation(R)
        f = img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)   # light photometric jitter
        return np.clip(f, 0, 255).astype(np.uint8), R, self.ok[i], self.look[i]


def collate(batch):
    cols = list(zip(*batch))
    return [to_tensor(np.stack(cols[0]))] + [torch.stack([torch.as_tensor(c) for c in col]) for col in cols[1:]]


def fixed_val_set(npz, seed=123, idx=None):
    """Faces degraded once per size with a fixed seed, so every epoch sees the same images."""
    d = np.load(npz)
    # read each array ONCE: indexing d['img'] inside the loop would unpack the whole file every time
    all_img, all_gt = d['img'], d['gt_euler']
    idx = np.arange(len(all_img)) if idx is None else np.asarray(idx)
    rng = np.random.default_rng(seed)
    sets = {}
    for s in VAL_SIZES:
        imgs = np.stack([degrade(all_img[i], rng, min_px=s, max_px=s, strength=0.5)[0] for i in idx])
        sets[s] = to_tensor(imgs)
    return sets, all_gt[idx]


def identity_split(npz, frac=0.05, seed=0):
    """Split 300W-LP by face identity (all rotated / flipped copies of one photo stay together), so the
    held-out faces are people the student never saw. Returns (train_idx, holdout_idx)."""
    stems = np.load(npz)['stem']
    ident = np.array([os.path.dirname(s).replace('_Flip', '') + '/' + os.path.basename(s).rsplit('_', 1)[0]
                      for s in stems])
    uniq = np.unique(ident)
    held = set(np.random.default_rng(seed).choice(uniq, max(1, int(len(uniq) * frac)), replace=False))
    mask = np.array([i in held for i in ident])
    return np.where(~mask)[0], np.where(mask)[0]


@torch.no_grad()
def predict(model, x, device, batch=512):
    model.eval()
    R, L = [], []
    for i in range(0, len(x), batch):
        r, l = model(x[i:i + batch].to(device))
        R.append(r.cpu()); L.append(l.cpu())
    return torch.cat(R), torch.cat(L)


def aflw_mae(model, val_sets, gt_euler, device):
    res = {}
    for s, x in val_sets.items():
        e = (compute_euler(predict(model, x, device)[0]) * 180 / np.pi).numpy()
        res[s] = np.abs(e - gt_euler).mean(0)          # pitch, yaw, roll
    return res


def stage1(args, device):
    npz = os.path.join(HEAD_DIR, 'faces_300W_LP.npz')
    print('loading 300W-LP faces ...', flush=True)
    tr_idx, ho_idx = identity_split(npz)
    print(f'300W-LP: {len(tr_idx)} training faces, {len(ho_idx)} held-out faces (unseen identities)', flush=True)
    train = FaceDistill(npz, tr_idx)
    loader = DataLoader(train, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                        drop_last=True, persistent_workers=args.workers > 0)
    jaad = None
    if args.jaad_pseudo:
        meta = pd.read_pickle(os.path.join(HEAD_DIR, 'heads_train.pkl'))
        rows = np.where(meta.teacher_ok.values)[0]
        if len(rows):
            jaad = DataLoader(JaadHeads('train', rows), args.batch // 4, shuffle=True, num_workers=args.workers,
                              collate_fn=collate, drop_last=len(rows) >= args.batch // 4)
            print(f'JAAD pseudo-labelled heads: {len(rows)}')
    val_sets, gt = fixed_val_set(os.path.join(HEAD_DIR, 'faces_AFLW2000.npz'))
    ho_sets, ho_gt = fixed_val_set(npz, idx=ho_idx[:3000])
    print('validation sets ready, training ...', flush=True)

    model = HeadStudent().to(device)
    print(f'student parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M')
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loader), pct_start=0.1)
    best, log = float('inf'), []
    jaad_iter = iter(jaad) if jaad else None
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, R_t, R_gt in loader:
            x, R_t, R_gt = x.to(device), R_t.to(device), R_gt.to(device)
            R = model(x)[0]
            loss = geodesic_loss(R, R_t).mean()
            if args.gt_weight:
                loss = loss + args.gt_weight * geodesic_loss(R, R_gt).mean()
            if jaad_iter is not None:
                try:
                    xj, Rj, okj, _ = next(jaad_iter)
                except StopIteration:
                    jaad_iter = iter(jaad)
                    xj, Rj, okj, _ = next(jaad_iter)
                loss = loss + args.jaad_weight * geodesic_loss(model(xj.to(device))[0], Rj.to(device)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        mae = aflw_mae(model, val_sets, gt, device)
        ho = aflw_mae(model, ho_sets, ho_gt, device)
        aflw_score = float(np.mean([m.mean() for m in mae.values()]))
        ho_score = float(np.mean([m.mean() for m in ho.values()]))
        # pick the epoch on held-out 300W-LP faces; AFLW2000 stays an untouched test set
        score = aflw_score if args.select == 'aflw' else ho_score
        row = {'epoch': ep + 1, 'loss_deg': np.degrees(tot / len(loader)), 'holdout_mae_mean': ho_score,
               'aflw_mae_mean': aflw_score,
               **{f'aflw_{s}px_{a}': m[j] for s, m in mae.items() for j, a in enumerate(['pitch', 'yaw', 'roll'])},
               **{f'holdout_{s}px_{a}': m[j] for s, m in ho.items() for j, a in enumerate(['pitch', 'yaw', 'roll'])}}
        log.append(row)
        print(f"ep {ep + 1:3d} | train {row['loss_deg']:.2f} deg | held-out 300W-LP MAE {ho_score:.1f} | AFLW2000 MAE " +
              ' '.join(f'{s}px:{m.mean():.1f}' for s, m in mae.items()) + f' | {time.time() - t0:.0f}s', flush=True)
        if score < best:
            best = score
            torch.save(model.state_dict(), os.path.join(HEAD_DIR, 'student_stage1.pt'))
        pd.DataFrame(log).to_csv(os.path.join(HEAD_DIR, 'train_stage1.csv'), index=False)
    print(f'best {args.select} MAE {best:.2f} deg -> student_stage1.pt')


def stage2(args, device):
    meta = pd.read_pickle(os.path.join(HEAD_DIR, 'heads_train.pkl'))
    rows = np.where((meta.look.values >= 0) & meta.found.values)[0]
    look = meta.look.values[rows]
    print(f'JAAD train heads with look label: {len(rows)} (looking {look.mean():.0%})')
    loader = DataLoader(JaadHeads('train', rows), args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=len(rows) > args.batch)
    faces = DataLoader(FaceDistill(os.path.join(HEAD_DIR, 'faces_300W_LP.npz')), args.batch, shuffle=True,
                       num_workers=args.workers, collate_fn=collate, drop_last=True)
    vmeta = pd.read_pickle(os.path.join(HEAD_DIR, 'heads_val.pkl'))
    vrows = np.where((vmeta.look.values >= 0) & vmeta.found.values)[0]
    vx = to_tensor(np.load(os.path.join(HEAD_DIR, 'heads_val.npz'))['img'][vrows])
    vy = vmeta.look.values[vrows]

    model = HeadStudent().to(device)
    model.load_state_dict(torch.load(os.path.join(HEAD_DIR, 'student_stage1.pt'), map_location=device))
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader))
    pos_weight = torch.tensor((1 - look.mean()) / max(look.mean(), 1e-6), device=device)
    best, log, face_iter = -1, [], iter(faces)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, R_t, ok, y in loader:
            x, R_t, ok, y = x.to(device), R_t.to(device), ok.to(device), y.to(device)
            R, logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            if ok.any():
                loss = loss + args.rot_weight * geodesic_loss(R[ok], R_t[ok]).mean()
            try:
                xf, Rf, _ = next(face_iter)
            except StopIteration:
                face_iter = iter(faces)
                xf, Rf, _ = next(face_iter)
            loss = loss + args.replay_weight * geodesic_loss(model(xf.to(device))[0], Rf.to(device)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        auc = roc_auc_score(vy, predict(model, vx, device)[1].numpy()) if len(np.unique(vy)) > 1 else float('nan')
        log.append({'epoch': ep + 1, 'loss': tot / len(loader), 'val_look_auc': auc})
        print(f'ep {ep + 1:3d} | loss {tot / len(loader):.3f} | JAAD val look AUC {auc:.3f}', flush=True)
        if auc > best or np.isnan(auc):
            best = auc
            torch.save(model.state_dict(), os.path.join(HEAD_DIR, 'student_stage2.pt'))
        pd.DataFrame(log).to_csv(os.path.join(HEAD_DIR, 'train_stage2.csv'), index=False)
    print(f'best val look AUC {best:.3f} -> student_stage2.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=[1, 2], required=True)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--gt_weight', type=float, default=0.0, help='stage 1: weight of the 300W-LP GT loss')
    ap.add_argument('--jaad_pseudo', action='store_true', help='stage 1: add teacher-labelled JAAD heads')
    ap.add_argument('--jaad_weight', type=float, default=0.5)
    ap.add_argument('--rot_weight', type=float, default=0.5, help='stage 2: pose loss on teacher_ok JAAD heads')
    ap.add_argument('--replay_weight', type=float, default=0.5, help='stage 2: 300W-LP distillation replay')
    ap.add_argument('--select', choices=['holdout', 'aflw'], default='holdout',
                    help='stage 1: choose the best epoch on held-out 300W-LP identities (default) or on AFLW2000')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.stage == 1:
        args.epochs, args.lr = args.epochs or 40, args.lr or 2e-3
        stage1(args, torch.device(args.device))
    else:
        args.epochs, args.lr = args.epochs or 15, args.lr or 5e-4
        stage2(args, torch.device(args.device))


if __name__ == '__main__':
    main()
