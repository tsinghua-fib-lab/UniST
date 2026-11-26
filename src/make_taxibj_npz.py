"""
make_taxibj_npz.py -- rebuild a UniST-style short-term TaxiBJ dataset as .npz (memory friendly).

usage:
  python make_taxibj_npz.py <TaxiBJ folder> <out.npz> [--channel 0] [--stride-train 4] [--stride-eval 1]

Layout (derived from probing TaxiNYC_short.json):
  X_<split>  (N, 12, 32, 32)   one flow channel, raw counts, windows slide one step
  P_<split>  (N, 3, 32, 32)    periodic reference frames: the slot right AFTER the window
                               (window start + 12), 3, 2 and 1 days earlier, OLDEST FIRST.
                               In the JSON these are repeated over the 12 window positions; the
                               DataLoader patch repeats them, so they are stored once here.
  T_<split>  (N, 12) strings 'YYYY-MM-DD HH:MM:SS'  (slot 01 of a day = 00:00)

Windows are built inside each period (the periods are months apart), only from complete
runs of consecutive half-hour slots, split 70/15/15 chronologically inside each period,
then concatenated. Train windows are thinned by --stride-train to keep memory small.
"""
import argparse
import datetime as dt
import glob
import os

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('folder')
ap.add_argument('out')
ap.add_argument('--channel', type=int, default=0, help='0 = inflow, 1 = outflow')
ap.add_argument('--seq-len', type=int, default=12)
ap.add_argument('--n-days', type=int, default=3)
ap.add_argument('--stride-train', type=int, default=4)
ap.add_argument('--stride-eval', type=int, default=1)
a = ap.parse_args()
L, D, STEP = a.seq_len, 48, dt.timedelta(minutes=30)


def slot_time(label):
    return dt.datetime.strptime(label[:8], '%Y%m%d') + STEP * (int(label[8:]) - 1)


splits = {k: {'X': [], 'P': [], 'T': []} for k in ('train', 'val', 'test')}
for path in sorted(glob.glob(os.path.join(a.folder, 'BJ*_M32x32_T30_InOut.h5'))):
    with h5py.File(path, 'r') as f:
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in f['date'][:]]
        flow = f['data'][:, a.channel].astype(np.float32)
    times = [slot_time(s) for s in labels]
    t0 = times[0]
    idx = np.array([int((t - t0) / STEP) for t in times])
    n = int((times[-1] - t0) / STEP) + 1
    full = np.zeros((n,) + flow.shape[1:], np.float32)
    present = np.zeros(n, bool)
    full[idx], present[idx] = flow, True

    starts = []
    for s in range(n - L + 1):
        if not present[s:s + L].all():
            continue
        ref = s + L - D * np.arange(a.n_days, 0, -1)          # oldest first
        if ref.min() >= 0 and present[ref].all():
            starts.append(s)
    starts = np.array(starts)
    cut1, cut2 = int(0.70 * len(starts)), int(0.85 * len(starts))
    parts = {'train': starts[:cut1][::a.stride_train], 'val': starts[cut1:cut2][::a.stride_eval],
             'test': starts[cut2:][::a.stride_eval]}
    print(f'{os.path.basename(path)}: {n} slots, {len(starts)} valid windows -> ' +
          ', '.join(f'{k} {len(v)}' for k, v in parts.items()))
    steps = np.arange(L)[None, :]
    for name, s in parts.items():
        if len(s) == 0:
            continue
        s = s[:, None]
        splits[name]['X'].append(full[s + steps])                                          # (n, L, H, W)
        ref = s[:, 0][:, None] + L - D * np.arange(a.n_days, 0, -1)[None, :]
        splits[name]['P'].append(full[ref])                                                # (n, days, H, W)
        base = np.datetime64(t0) + (s + steps) * np.timedelta64(30, 'm')
        splits[name]['T'].append(np.char.replace(np.datetime_as_string(base, unit='s'), 'T', ' '))

out = {}
for name, parts in splits.items():
    for key in 'XPT':
        out[f'{key}_{name}'] = np.concatenate(parts[key], axis=0)
np.savez(a.out, **out)
print('\nwrote', a.out)
for k, v in out.items():
    print(f'  {k}: {v.shape} {v.dtype}')