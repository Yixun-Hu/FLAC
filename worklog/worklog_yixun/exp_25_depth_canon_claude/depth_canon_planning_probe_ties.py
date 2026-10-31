#!/usr/bin/env python3
"""exp_25 planning probe (run 2026-10-23 before the plan was written; results in
depth_canon_worklog.md).  Two questions the canonicalization design depends on:

1. How often is the "deepest pixel" of an AR depth panorama ambiguous (exact ties
   at the global maximum, in different columns)?
2. Can per-pixel depth be recovered from the metadata point cloud ([3,H,W] camera
   coordinates) in a way that is BITWISE invariant under the reviewed yaw operator
   rotate_scene_metadata, so that ties and comparisons do not flip under rotation?

Read-only: loads depth maps from the dataset symlink, writes nothing.
Run from the repo root with the `flac` env:
    python worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_planning_probe_ties.py
Planner-written (Claude Fable 5.1); reviewed in the first batched Codex round.
"""
import glob
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from src.configs.dataset_configs.custom_metadata.AR_md import convert_equirect_to_camera_coord  # noqa: E402
from src.data.yaw_rotation import offsets_to_radians, rotate_scene_metadata  # noqa: E402

H, W = 256, 512
N_SAMPLE = 400
N_ROT = 60

allf = sorted(glob.glob("AcousticRooms/depth_map/*/*/*.npy"))
assert len(allf) >= N_SAMPLE, f"only {len(allf)} depth maps found; need >= {N_SAMPLE}"
random.seed(1)
fs = random.sample(allf, N_SAMPLE)

# ---- 1. tie structure at the global maximum ---------------------------------
same_col = cross_col = rows_same = 0
dists, examples = [], []
gaps = []
for f in fs:
    d = np.load(f)
    assert d.shape == (H, W) and d.dtype == np.float64 and np.isfinite(d).all()
    m = d.max()
    idx = np.argwhere(d == m)
    cm = np.sort(d.max(axis=0))[::-1]
    gaps.append(cm[0] - cm[1])
    if len(idx) > 1:
        cols = sorted(set(idx[:, 1].tolist()))
        rws = set(idx[:, 0].tolist())
        if len(cols) == 1:
            same_col += 1
        else:
            cross_col += 1
            dists.append(max(min(abs(a - b), W - abs(a - b)) for a in cols for b in cols if a != b))
            rows_same += int(len(rws) == 1)
            if len(examples) < 6:
                examples.append((os.path.relpath(f, "AcousticRooms/depth_map"), len(idx), cols[:8], sorted(rws)[:4], round(float(m), 4)))
gaps = np.array(gaps)
print(f"maps sampled {N_SAMPLE} of {len(allf)}")
print(f"exact ties at the global max: same-column-only {same_col}, cross-column {cross_col}; "
      f"cross-column circular separations {sorted(dists)}; all on one row: {rows_same}")
print(f"top1-top2 column-max gap: min {gaps.min():.3e} p1 {np.percentile(gaps, 1):.3e} "
      f"median {np.median(gaps):.3e}; n<1e-6 {int((gaps < 1e-6).sum())}; n<1e-3 {int((gaps < 1e-3).sum())}")
for e in examples:
    print("  example:", e)

# ---- 2. exact, rotation-invariant depth recovery from the z channel ---------
phi = torch.arange(H)
phi_map = (phi + 0.5) * math.pi / H - math.pi / 2          # float32, as in AR_md
s = torch.sin(phi_map)
print(f"min |sin phi| over rows: {float(s.abs().min()):.5f} (never zero)")
nbad = norm_bad = ntot = 0
for f in fs[:N_ROT]:
    d = np.load(f)
    pc = convert_equirect_to_camera_coord(torch.from_numpy(d), H, W).permute(2, 0, 1)   # [3,H,W] float64
    rec = pc[2] / (-s.to(pc.dtype))[:, None]
    md = {"depth": pc, "source": torch.randn(3), "source_vit": torch.randn(1, 3),
          "context_poses": torch.randn(8, 3), "context_poses_vit": torch.randn(8, 3)}
    for dj in (1, 64, 128, 300, 511):
        r = rotate_scene_metadata(md, offsets_to_radians([dj], W)[0], W)
        rec_r = torch.roll(r["depth"][2] / (-s.to(pc.dtype))[:, None], shifts=-dj, dims=1)
        nrm_r = torch.roll(r["depth"].norm(dim=0), shifts=-dj, dims=1)
        ntot += 1
        nbad += int(not torch.equal(rec_r, rec))
        norm_bad += int(not torch.equal(nrm_r, pc.norm(dim=0)))
print(f"z-channel recovery bitwise-invariant under rotate_scene_metadata: {nbad}/{ntot} mismatches; "
      f"L2-norm route: {norm_bad}/{ntot} mismatches")
d = np.load(fs[0])
pc = convert_equirect_to_camera_coord(torch.from_numpy(d), H, W).permute(2, 0, 1)
rec = pc[2] / (-s.to(pc.dtype))[:, None]
raw = torch.from_numpy(d)
print(f"recovered-vs-raw: max abs err {float((rec - raw).abs().max()):.2e}; "
      f"fraction bitwise equal {float((rec == raw).float().mean()):.4f}")
