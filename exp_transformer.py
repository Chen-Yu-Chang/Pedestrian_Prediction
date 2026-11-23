"""Transformer crossing-intention model: box (B1 baseline) + head orientation.

Main comparison (default):  box only (B1)  vs  box + head pose.
Separate, optional test:    box + body pose (`--sets bbox+pose`), and head alone (`--sets head`).

The network is the Transformer of exp_classifiers.py (64-d tokens, learned positional embedding,
2 encoder layers, 4 heads, feed-forward 128, dropout 0.1, mean pooling, linear output), with one change:
each cue group gets its own input projection and the projections are summed into the frame token,

    token_t = W_box x_box,t + W_pose x_pose,t + W_head x_head,t + b .

With only the box group this is exactly the original model (a sum of linear maps is one linear map), and
the parameters are created in the same order, so `--sets bbox` reproduces the Transformer row of the
classifier comparison for the same seeds. With more groups, `--mod_drop p` zeroes each non-box group's
projection per sample with probability p during training (modality dropout), so the model still works
when a cue is missing (e.g. face turned away) and cannot lean on a single cue.

Cue groups (per frame):
  bbox : 6  - cx, cy, vel_x, vel_y, delta_area, aspect_ratio             (exp_classifiers.py)
  head : 6  - yaw, pitch, roll (/90), P(looking), yaw change, head usable flag   (hp_05_head_features.py)
  pose : 85 - 17 keypoints (x, y, conf) normalised to the box + velocity (exp_classifiers.py; separate test only)

Evaluation is the same as exp_classifiers.py (5-fold CV grouped by video, inner validation videos for early
stopping and threshold, 3 seeds), plus window AUC by time to the decision point for `before_dp`.

Usage:
  python exp_transformer.py                         # main test: box vs box + head, 5-fold CV
  python exp_transformer.py --sets bbox+pose        # separate test: box + body pose (17 keypoints)
  python exp_transformer.py --official              # official train / val / test split instead of CV
Output: outputs_pose/results_transformer_{cv|official}_{sets}.csv, e.g. results_transformer_cv_bbox_bbox+head.csv
"""
import argparse
import os
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from exp_classifiers import (BBOX, FRAME_STRIDE, OUT, POSE, SEQ_LEN, SEEDS, device, frame_features, metrics,
                             predict_deep, val_threshold)

HEAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs_head')
HEAD = ['h_yaw_n', 'h_pitch_n', 'h_roll_n', 'h_look', 'h_dyaw', 'h_ok']
GROUPS = {'bbox': BBOX, 'pose': POSE, 'head': HEAD}
TTE_BINS = [(0, 30), (30, 60), (60, 90), (90, 10 ** 6)]   # frames before the decision point (30 fps)


# ---------------- data ----------------
def add_head_features(df, split):
    """Merge per-frame head features (hp_05) into the frame table and derive the 6 head inputs."""
    path = os.path.join(HEAD_DIR, f'head_{split}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found - run hp_01..hp_05 first, or use --sets without head')
    h = pd.read_pickle(path).drop_duplicates(['video_id', 'ped_id', 'frame'])
    df = df.merge(h, on=['video_id', 'ped_id', 'frame'], how='left')
    df['h_ok'] = df.h_ok.fillna(0.0)
    for a in ('yaw', 'pitch', 'roll'):
        df[f'h_{a}_n'] = df[f'h_{a}'].fillna(0.0) / 90.0
    df['h_look'] = df.h_look.fillna(0.0) if df.h_look.notna().any() else 0.0
    g = df.groupby(['video_id', 'ped_id'])
    cont = (g.frame.diff() == FRAME_STRIDE).values
    both = (df.h_ok.values > 0) & (g.h_ok.shift().fillna(0).values > 0) & cont
    df['h_dyaw'] = np.where(both, df.h_yaw_n - g.h_yaw_n.shift(), 0.0)
    return df


def load_data(need_head):
    data = {}
    for s in ['train', 'val', 'test']:
        d = frame_features(pd.read_pickle(os.path.join(OUT, f'pose_{s}.pkl')))
        data[s] = add_head_features(d, s) if need_head else d
    return data


def make_windows(df, cols):
    """Same windows as exp_classifiers.make_windows, plus each window's frames-to-decision-point."""
    X, y, grp, tte = [], [], [], []
    for (v, p, _), g in df.groupby(['video_id', 'ped_id', 'streak'], sort=False):
        if len(g) < SEQ_LEN:
            continue
        f, fr, dp = g[cols].values.astype(np.float32), g.frame.values, g.dp.iat[0]
        for i in range(len(g) - SEQ_LEN + 1):
            X.append(f[i:i + SEQ_LEN]); y.append(g.label.iat[0]); grp.append(f'{v}/{p}')
            tte.append(dp - fr[i + SEQ_LEN - 1])
    return (np.array(X).reshape(-1, SEQ_LEN, len(cols)), np.array(y), np.array(grp), np.array(tte))


# ---------------- model ----------------
class CueTransformer(nn.Module):
    def __init__(self, dims, mod_drop=0.0, group_mlp=False):
        super().__init__()
        self.dims, self.mod_drop = list(dims), mod_drop
        mk = (lambda d: nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 64))) if group_mlp \
            else (lambda d: nn.Linear(d, 64))
        self.inp = nn.ModuleList([mk(d) for d in self.dims])
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        tok = 0
        for k, (proj, xg) in enumerate(zip(self.inp, torch.split(x, self.dims, dim=-1))):
            e = proj(xg)
            if k > 0 and self.training and self.mod_drop > 0:      # never drop the box group
                e = e * (torch.rand(e.shape[0], 1, 1, device=e.device) >= self.mod_drop)
            tok = tok + e
        return self.fc(self.enc(tok + self.pos).mean(1))


def train_transformer(dims, Xtr, ytr, Xva, yva, seed, args):
    """Same optimisation as exp_classifiers.train_deep (Adam 1e-3, wd 1e-4, batch 256, <=40 epochs,
    class-balanced BCE, early stopping on validation AUC with patience 10)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = CueTransformer(dims, args.mod_drop, args.group_mlp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w = np.where(ytr == 1, len(ytr) / (2 * ytr.sum()), len(ytr) / (2 * (1 - ytr).sum())).astype(np.float32)
    Xt, yt, wt = (torch.tensor(a).to(device) for a in (Xtr, ytr.astype(np.float32), w))
    Xv = torch.tensor(Xva).to(device)
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xt[b]).squeeze(1), yt[b], weight=wt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sv = predict_deep(model, Xv)
        auc = roc_auc_score(yva, sv) if len(np.unique(yva)) > 1 else ep
        if auc > best:
            best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_state)
    return model


# ---------------- experiment ----------------
def tte_aucs(y, s, tte):
    out = {}
    for lo, hi in TTE_BINS:
        sel = (tte >= lo) & (tte < hi)
        key = f'auc_tte_{lo // 30}-{hi // 30}s' if hi < 10 ** 6 else f'auc_tte_{lo // 30}s+'
        out[key] = roc_auc_score(y[sel], s[sel]) if sel.any() and len(np.unique(y[sel])) > 1 else np.nan
    return out


def run_one(tr, va, te, fs, setting, rows, tag, args):
    groups = fs.split('+')
    cols = [c for g in groups for c in GROUPS[g]]
    dims = [len(GROUPS[g]) for g in groups]
    if setting == 'before_dp':
        tr, va, te = (d[d.frame <= d.dp] for d in (tr, va, te))
    sc = StandardScaler().fit(tr[cols].values)
    norm = lambda d: d.assign(**dict(zip(cols, sc.transform(d[cols].values).T)))
    (Xtr, ytr, _, _), (Xva, yva, _, _), (Xte, yte, gte, tte) = (make_windows(norm(d), cols) for d in (tr, va, te))
    for seed in args.seeds:
        t0 = time.time()
        model = train_transformer(dims, Xtr, ytr, Xva, yva, seed, args)
        sv, st = predict_deep(model, Xva), predict_deep(model, Xte)
        row = {'split': tag, 'features': fs, 'setting': setting, 'model': 'Transformer', 'seed': seed,
               'mod_drop': args.mod_drop if len(groups) > 1 else 0.0,
               **metrics(yte, st, gte, val_threshold(yva, sv)), **tte_aucs(yte, st, tte), 'sec': time.time() - t0}
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)


def summarize(res):
    cols = ['seq_auc', 'seq_bacc', 'ped_auc'] + [c for c in res.columns if c.startswith('auc_tte')]
    g = res.groupby(['setting', 'features'])[cols]
    table = g.mean().round(3).astype(str) + ' ± ' + g.std().round(3).astype(str)
    print('\nmean ± std over folds and seeds\n' + table.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['bbox', 'bbox+head'])
    ap.add_argument('--settings', nargs='+', default=['all', 'before_dp'])
    ap.add_argument('--official', action='store_true', help='official split instead of 5-fold CV')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--mod_drop', type=float, default=0.15)
    ap.add_argument('--group_mlp', action='store_true', help='2-layer MLP per cue group instead of a linear map')
    args = ap.parse_args()
    for fs in args.sets:
        assert all(g in GROUPS for g in fs.split('+')), f'unknown cue group in {fs}'

    data = load_data(any('head' in fs for fs in args.sets))
    rows = []
    out = os.path.join(OUT, f"results_transformer_{'official' if args.official else 'cv'}_{'_'.join(args.sets)}.csv")
    for fs, setting in product(args.sets, args.settings):
        if args.official:
            run_one(data['train'], data['val'], data['test'], fs, setting, rows, 'official', args)
        else:   # identical folds to exp_classifiers.py --cv
            full = pd.concat(data.values(), ignore_index=True)
            for k, (tri, tei) in enumerate(GroupKFold(args.folds).split(full, groups=full.video_id.values)):
                trd, ted = full.iloc[tri], full.iloc[tei]
                tv = np.unique(trd.video_id.values)
                vv = set(np.random.RandomState(k).choice(tv, len(tv) // 5, replace=False))
                run_one(trd[~trd.video_id.isin(vv)], trd[trd.video_id.isin(vv)], ted, fs, setting, rows, f'fold{k}', args)
        pd.DataFrame(rows).to_csv(out, index=False)
    summarize(pd.DataFrame(rows))
    print('saved', out)


if __name__ == '__main__':
    main()
