"""One-off (exp_05): per-batch BN-input mean dispersion vs EMA-noise prediction.

For the stem BN (cnn.bn1), record PER-BATCH input means over M batches (no pooling),
compute the cross-batch dispersion of batch means per channel (in running-sigma units),
multiply by sqrt(m/(2-m)) (m=0.1 -> 0.229) = predicted EMA-tail noise of running_mean,
and compare against the observed B0 residual (typical 0.014-0.02, max-channel 0.082).
"""
import sys, os, json


def _repo_root(p):  # marker-walk: survives worklog relocations (worklog_yixun move, 2026-07-12)
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


sys.path.insert(0, _repo_root(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.models import create_model_from_config
from src.data.dataset import create_dataloader_from_config

REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
mc = json.load(open(os.path.join(REPO, 'src/configs/model_configs/FLAC/AR/FLAC_AR.json')))
model = create_model_from_config(mc)
sd = torch.load(os.path.join(REPO, 'weights/FLAC/FLAC_EMA.ckpt'), map_location='cpu')['state_dict']
for k in list(sd.keys()):
    if k.startswith('diffusion.'):
        sd[k.replace('diffusion.', '', 1)] = sd.pop(k)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing and not unexpected, (len(missing), len(unexpected))
dev = 'cuda'
cond = model.conditioner.to(dev).eval()
net = cond.conditioners['context_audio'].net if hasattr(cond.conditioners['context_audio'], 'net') else cond.conditioners['context_audio']
bn1 = dict(net.named_modules())['cnn.bn1'] if 'cnn.bn1' in dict(net.named_modules()) else dict(net.named_modules())['bn1']

batch_means = []
def hook(module, inputs):
    x = inputs[0].detach()
    batch_means.append(x.transpose(0, 1).reshape(x.shape[1], -1).mean(dim=1).cpu())
h = bn1.register_forward_pre_hook(hook)

dc = json.load(open(os.path.join(REPO, 'src/configs/dataset_configs/AR/train/acousticroom_train.json')))
dl = create_dataloader_from_config(dc, batch_size=16, num_workers=4, sample_rate=mc['sample_rate'], sample_size=mc['sample_size'], audio_channels=1, shuffle=True)
M = 60
with torch.no_grad():
    for i, (_, md) in enumerate(dl):
        if i >= M: break
        cond(md, dev, only_ids=('context_audio',))
h.remove()

bm = torch.stack(batch_means)                     # [M, C]
rm = bn1.running_mean.cpu(); rv = bn1.running_var.cpu()
sigma = (rv + 1e-5).sqrt()
disp = bm.std(dim=0, unbiased=True) / sigma        # per-channel batch-mean dispersion, in running-sigma units
pred_ema_noise = disp * (0.1 / 1.9) ** 0.5         # x0.229
obs_shift = (bm.mean(dim=0) - rm).abs() / sigma    # observed mean shift over M batches
out = {
    'M_batches': M,
    'dispersion_sigma_units': {'mean': disp.mean().item(), 'max': disp.max().item()},
    'predicted_ema_noise': {'mean': pred_ema_noise.mean().item(), 'max': pred_ema_noise.max().item()},
    'observed_mean_shift': {'mean': obs_shift.mean().item(), 'max': obs_shift.max().item()},
    'verdict_hint': 'EMA-tail plausible if observed_max ~ predicted_max; pipeline-drift if observed >> predicted',
}
print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(REPO, 'worklog/exp_05_bn_drift_bisect_claude/dispersion_check_result.json'), 'w'), indent=2)
