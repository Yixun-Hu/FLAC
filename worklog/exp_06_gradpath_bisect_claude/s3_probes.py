"""One-off (exp_06 S3.2): target-side mechanism probes on the train distribution.

(a) truncated-energy: fraction of RIR energy beyond the 10240-sample loss window
    and beyond the 8000-sample metric window.
(c) augmentation bias: paired T60/EDT of raw vs AddNoise(40-60dB pink)-augmented
    targets (RandomTimeShift(10) included), using the repo's own metric stack.
"""
import sys, os, json, glob, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torchaudio
from src.data.utils import AddNoise, RandomTimeShift
from src.metrics.metric_callback import AcousticMetricsCallback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
random.seed(42); torch.manual_seed(42)
train_split = json.load(open(os.path.join(REPO, 'data/AR/train.json')))
train_rooms = {room for rooms in train_split.values() for room in rooms}
files = [f for f in sorted(glob.glob(os.path.join(REPO, 'AcousticRooms/single_channel_ir_1/**/*.wav'), recursive=True))
         if f.split('/')[-2] in train_rooms]
random.shuffle(files)
files = files[:300]
print(f'sampled {len(files)} RIRs strictly from the train split ({len(train_rooms)} rooms)')

aug_shift = RandomTimeShift(max_shift=10, p=1.0)   # p=1 for paired measurement; loader uses p=0.5
aug_noise = AddNoise(snr_db_range=(40, 60), noise_type='pink', p=1.0)

cb_raw = AcousticMetricsCallback(dataset_name='AcousticRooms', sample_rate=22050, sample_size=10240, audio_channels=1, eval_per_scene=False, device='cpu', eval_T60=True, eval_C50=True, eval_EDT=True, eval_l1_distance=False, eval_l1_distance_multires=False, eval_FD=False, eval_retrieval=False, eval_env=False, AGREE_ckpt=None)

en_beyond_loss, en_beyond_metric, skipped = [], [], 0
pairs = []
for fp in files:
    try:
        w, sr = torchaudio.load(fp)
    except Exception:
        skipped += 1; continue
    w = w[:1]
    e_tot = float((w ** 2).sum())
    if e_tot <= 0: skipped += 1; continue
    e_loss = float((w[:, :10240] ** 2).sum()); e_metric = float((w[:, :8000] ** 2).sum())
    en_beyond_loss.append(1 - e_loss / e_tot); en_beyond_metric.append(1 - e_metric / e_tot)
    t = w[:, :10240]
    if t.shape[1] < 10240:
        t = torch.cat([t, torch.zeros(1, 10240 - t.shape[1])], dim=1)
    a = aug_noise(aug_shift(t.clone()))
    pairs.append((t, a))
print(f'valid {len(pairs)}, skipped {skipped}')

# paired metric deltas: run augmented as "pred" against raw as "ref" is NOT what we want;
# instead compute per-signal T60/EDT via the callback's underlying per-item path: use update with
# identical pred/ref to extract per-signal values is unsupported -> use relative comparison:
# metric(aug vs raw) directly quantifies how much augmentation moves the acoustic measurements.
B = 32
for i in range(0, len(pairs), B):
    chunk = pairs[i:i+B]
    raw = torch.stack([r for r, _ in chunk]); aug = torch.stack([a for _, a in chunk])
    cb_raw.update_metrics('test', aug, raw, scene=None)
m = cb_raw.compute_metrics('test')
res = {
    'n': len(pairs),
    'energy_beyond_loss_window_10240': {'mean': sum(en_beyond_loss)/len(en_beyond_loss), 'p90': sorted(en_beyond_loss)[int(0.9*len(en_beyond_loss))]},
    'energy_beyond_metric_window_8000': {'mean': sum(en_beyond_metric)/len(en_beyond_metric), 'p90': sorted(en_beyond_metric)[int(0.9*len(en_beyond_metric))]},
    'augmentation_bias_metric_aug_vs_raw': {k: (float(v) if not isinstance(v, dict) else v) for k, v in m.items()},
    'note': 'aug applied at p=1.0 for paired measurement; loader applies each at p=0.5 -> effective training bias ~ half the T60/EDT numbers here (shift) and ~half of items carry noise.',
}
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(REPO, 'worklog/exp_06_gradpath_bisect_claude/s3_probe_results.json'), 'w'), indent=2)
