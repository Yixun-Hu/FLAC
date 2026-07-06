"""Generates depth_c4.png and rir_rotation.png for the exp_02 results page.

Sample = index 0 of the unseen K=1 eval order (seed 42, shuffle=False) — the same
item as index 0 of the stored prediction tensors (pairing proven by the rot0 control).
CPU-only. Colormaps: single-hue Blues for depth magnitude (sequential rule);
cyclic twilight for per-pixel vector azimuth (cyclic quantity).
"""
import sys, os, json, math
os.environ.setdefault('MPLBACKEND', 'Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import create_dataloader_from_config
from src.data.yaw_rotation import rotate_scene_metadata

mc = json.load(open(os.path.join(REPO, 'src/configs/model_configs/FLAC/AR/FLAC_AR.json')))
dc = json.load(open(os.path.join(REPO, 'src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json')))
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)
dl = create_dataloader_from_config(dc, batch_size=2, num_workers=2, sample_rate=mc['sample_rate'], sample_size=mc['sample_size'], audio_channels=1, shuffle=False)
reals, metadata = next(iter(dl))
md = metadata[0]
depth = md['depth']                      # [3, H, W] per-pixel 3D points, listener frame
img_w = int(depth.shape[-1])
gt = reals[0, 0].numpy()
sr = mc['sample_rate']

# ---------------- figure 1: the C4-rotated panorama ----------------
angles = [0, 90, 180, 270]
fig, axes = plt.subplots(4, 2, figsize=(11.5, 9.2), constrained_layout=True)
for r, a in enumerate(angles):
    d = rotate_scene_metadata({'depth': depth}, math.radians(a), img_w)['depth']
    dist = torch.linalg.vector_norm(d, dim=0).numpy()          # rotation-invariant magnitude -> shows the ROLL
    azim = torch.atan2(d[1], d[0]).numpy()                     # per-pixel vector azimuth -> shows the VECTOR rotation
    src = rotate_scene_metadata({'source': md['source']}, math.radians(a), img_w)['source']
    src_az = math.atan2(float(src[1]), float(src[0]))
    src_col = ((src_az + math.pi) / (2 * math.pi) * img_w - 0.5) % img_w
    im0 = axes[r, 0].imshow(dist, cmap='Blues', vmin=0, vmax=np.percentile(dist, 99))
    im1 = axes[r, 1].imshow(azim, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    for c in (0, 1):
        axes[r, c].axvline(src_col, color='#e34948', lw=1.4)
        axes[r, c].set_yticks([]); axes[r, c].set_xticks([0, img_w // 2, img_w - 1])
        axes[r, c].set_xticklabels(['-180°', '0°', '+180°'], fontsize=8)
    axes[r, 0].set_ylabel(f'α = {a}°', fontsize=11)
axes[0, 0].set_title('radial distance ‖p‖ (invariant values → pure column roll)', fontsize=10)
axes[0, 1].set_title('per-pixel vector azimuth atan2(y, x) (values rotate with α)', fontsize=10)
fig.colorbar(im0, ax=axes[:, 0], shrink=0.55, label='distance (m)')
fig.colorbar(im1, ax=axes[:, 1], shrink=0.55, label='azimuth (rad)')
fig.suptitle('Yaw rotation of the conditioning panorama (sample 0, unseen K=1) — red line: target-source azimuth', fontsize=11)
fig.savefig(os.path.join(HERE, 'depth_c4.png'), dpi=110)
plt.close(fig)

# ---------------- figure 2: RIRs before/after rotation ----------------
def load_preds_all(name):
    obj = torch.load(os.path.join(REPO, f'weights/FLAC/FLAC_EMA_predictions_1_1.0_{name}.pt'), map_location='cpu')
    return (obj['predictions'] if isinstance(obj, dict) else obj)[:, 0, :].numpy()
P0 = load_preds_all('yaw_baseline')
PA = {a: load_preds_all(f'yaw_rot{a}') for a in (90, 180, 270)}
# honest sample choice: 90th percentile of the per-sample P180-vs-P0 rel-L2 gap (median stated in caption)
d = P0 - PA[180]
rel = np.sqrt((d ** 2).sum(axis=1)) / (np.sqrt((P0 ** 2).sum(axis=1)) + 1e-8)
order = np.argsort(rel)
idx = int(order[int(0.9 * len(order))])
med, p90v, samp = float(np.median(rel)), float(rel[order[int(0.9 * len(order))]]), float(rel[idx])
# ground truth for that index: iterate the (shuffle=False, batch=2) loader
need_batch, need_off = divmod(idx, 2)
git = None
for bi, (rr, _) in enumerate(dl):
    if bi == need_batch:
        git = rr[need_off, 0].numpy(); break
def edc_db(x):
    e = np.flip(np.cumsum(np.flip(x.astype(np.float64) ** 2)))
    return 10 * np.log10(e / (e[0] + 1e-12) + 1e-12)
COL = {'gt': '#8a887f', 'p0': '#2a78d6', 90: '#1baf7a', 180: '#e34948', 270: '#eda100'}
lo, hi = int(0.015 * sr), int(0.070 * sr)
t = np.arange(lo, hi) / sr * 1000
fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.6), constrained_layout=True)
axes[0].plot(t, git[lo:hi], color=COL['gt'], lw=1.0, label='ground truth (rotation-invariant)')
axes[0].plot(t, P0[idx, lo:hi], color=COL['p0'], lw=1.1, alpha=0.95, label='P\u2080 (unrotated conditioning)')
axes[0].plot(t, PA[180][idx, lo:hi], color=COL[180], lw=1.1, alpha=0.85, label='P\u2081\u2088\u2080 (conditioning rotated 180\u00b0)')
axes[0].set_xlabel('time (ms)'); axes[0].set_ylabel('amplitude')
axes[0].set_title(f'Predicted RIR waveform, 15\u201370 ms (sample #{idx}) \u2014 same noise, same GT, only the conditioning rotated')
axes[0].legend(loc='upper right', fontsize=9)
axes[1].plot(t, PA[180][idx, lo:hi] - P0[idx, lo:hi], color=COL[180], lw=1.0)
axes[1].set_xlabel('time (ms)'); axes[1].set_ylabel('P\u2081\u2088\u2080 \u2212 P\u2080')
axes[1].set_title(f'Difference trace: rel-L2 = {samp:.3f} (dataset median {med:.3f}, p90 {p90v:.3f}) \u2014 an invariant model gives a flat zero line')
axes[2].plot(edc_db(git), color=COL['gt'], lw=1.6, label='ground truth')
axes[2].plot(edc_db(P0[idx]), color=COL['p0'], lw=1.4, label='P\u2080')
for a in (90, 180, 270):
    axes[2].plot(edc_db(PA[a][idx]), color=COL[a], lw=1.1, alpha=0.85, label=f'P{a}\u00b0')
axes[2].set_ylim(-80, 2); axes[2].set_xlabel('sample'); axes[2].set_ylabel('energy decay (dB)')
axes[2].set_title('Schroeder energy decay \u2014 rotated-conditioning predictions decay differently (T60/EDT damage)')
axes[2].legend(loc='upper right', fontsize=9)
fig.savefig(os.path.join(HERE, 'rir_rotation.png'), dpi=110)
plt.close(fig)
print(f'wrote rir_rotation.png (sample {idx}, rel {samp:.3f}, median {med:.3f})')
