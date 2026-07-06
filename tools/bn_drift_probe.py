"""BN-drift probe for the FLAC RIR encoder (exp_05).

The RIR encoder's BatchNorm running statistics are a gradient-free data-drift
instrument. exp_04's lr=0 null run regressed T60/EDT purely through RIR-encoder
BN running-stat adaptation, which is only possible if this repo's loader feeds
the encoder reference-RIR statistics different from the released training's.
Comparing each BN layer's stored ``running_mean``/``running_var`` against the
batch statistics *our* data produces localizes the drift with no training.

Public surface (unit-pinned in ``src/tests/test_bn_drift_probe.py``):
- ``bn_drift_metrics``  -- standardized per-channel mean_shift / var_ratio.
- ``BNInputRecorder``   -- forward-pre-hook Welford accumulation of BN *inputs*.
- ``snapshot_buffers`` / ``assert_buffers_unchanged`` -- read-only guarantee.
- ``build_report``      -- provenance + per-layer drift summary dict.
- ``probe_rir_encoder`` -- end-to-end; fail-fast on any unclean checkpoint load
  (needs the real ckpt+dataset; the B-1 pilot covers it, so only its pure
  parts are unit-tested).

Run directly (report JSON always written; default ./bn_drift_report.json):
    python tools/bn_drift_probe.py --ckpt-path weights/FLAC/FLAC_EMA.ckpt \\
        --dataset-config <dataset.json> --n-batches 200 --batch-size 16
"""
import os
import sys

# Prepend the repo root BEFORE any ``src.*`` import: the editable ``rir2rir``
# install maps ``src`` to a divergent sibling checkout, so without this a bare
# ``python tools/bn_drift_probe.py`` would import the wrong ``src`` (see MEMORY).
# Remove-then-insert(0) so the repo root wins even when it is already on
# sys.path *behind* an entry that shadows ``src`` (review finding 5).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tools/ -> repo root
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch
from torch import nn

_FLAC_AR_CONFIG = os.path.join(
    _REPO_ROOT, "src", "configs", "model_configs", "FLAC", "AR", "FLAC_AR.json"
)


def bn_drift_metrics(running_mean, running_var, batch_mean, batch_var, eps=1e-5):
    """Standardized per-channel drift between stored BN stats and batch stats.

    ``mean_shift = |batch_mean - running_mean| / sqrt(running_var + eps)`` and
    ``var_ratio = batch_var / (running_var + eps)``. Returns the two per-channel
    tensors plus scalar summaries (mean_shift max/mean; var_ratio min/max over
    channels). Computed in the inputs' dtype so identical stats give exactly
    mean_shift 0 / var_ratio 1 at ``eps=0``.
    """
    denom_var = running_var + eps
    mean_shift = torch.abs(batch_mean - running_mean) / torch.sqrt(denom_var)
    var_ratio = batch_var / denom_var
    return {
        "mean_shift": mean_shift,
        "var_ratio": var_ratio,
        "mean_shift_max": float(mean_shift.max()),
        "mean_shift_mean": float(mean_shift.mean()),
        "var_ratio_min": float(var_ratio.min()),
        "var_ratio_max": float(var_ratio.max()),
    }


class BNInputRecorder:
    """Forward-pre-hooks on every ``BatchNorm1d``/``BatchNorm2d`` of a module,
    streaming per-channel mean/variance of the BN *inputs* over batches.

    Variance is finalized with the UNBIASED (n-1) estimator to match PyTorch's
    ``running_var`` semantics. Accumulation uses the parallel (chunked) Welford
    combine in float64 so many batches stay accurate; accumulators are lazily
    initialized on the first hooked batch's device, so CPU and CUDA inputs both
    work (review finding 2). The hooks only *read* inputs, so a probe pass in
    eval mode leaves every BN running buffer bit-identical (BatchNorm does not
    update its stats in eval mode).
    """

    _BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)

    def __init__(self, module):
        self._modules_by_name = {}
        self._count = {}
        self._mean = {}   # lazily created on the hook input's device (float64)
        self._m2 = {}
        self._handles = []
        for name, m in module.named_modules():
            if isinstance(m, self._BN_TYPES):
                self._modules_by_name[name] = m
                self._count[name] = 0
                self._mean[name] = None
                self._m2[name] = None
                self._handles.append(m.register_forward_pre_hook(self._make_hook(name)))

    @property
    def layer_names(self):
        return list(self._modules_by_name.keys())

    @property
    def n_layers(self):
        return len(self._modules_by_name)

    def _make_hook(self, name):
        def hook(_module, inputs):
            self._update(name, inputs[0])
        return hook

    def _update(self, name, x):
        # Channel-major flatten [C, M]: reduce over batch and every spatial dim.
        x = x.detach()
        chans = x.shape[1]
        xc = x.transpose(0, 1).reshape(chans, -1).to(torch.float64)
        n_b = xc.shape[1]
        if n_b == 0:
            return
        if self._mean[name] is None:  # lazy: first batch pins the accumulator device
            self._mean[name] = torch.zeros(chans, dtype=torch.float64, device=xc.device)
            self._m2[name] = torch.zeros(chans, dtype=torch.float64, device=xc.device)
        mean_b = xc.mean(dim=1)
        m2_b = ((xc - mean_b.unsqueeze(1)) ** 2).sum(dim=1)
        count = self._count[name]
        new_count = count + n_b
        delta = mean_b - self._mean[name]
        self._mean[name] = self._mean[name] + delta * (n_b / new_count)
        self._m2[name] = self._m2[name] + m2_b + (delta ** 2) * (count * n_b / new_count)
        self._count[name] = new_count

    def input_stats(self, name):
        """(mean, unbiased var) of the recorded BN inputs, per channel (float32).

        Returned on the accumulator's device (= the hooked inputs' device);
        zeros on CPU if the layer was never reached by a forward pass.
        """
        count = self._count[name]
        if count == 0:
            zeros = torch.zeros(self._modules_by_name[name].num_features)
            return zeros, zeros.clone()
        mean = self._mean[name]
        var = self._m2[name] / (count - 1) if count > 1 else torch.zeros_like(mean)
        return mean.to(torch.float32), var.to(torch.float32)

    def running_stats(self, name):
        """The BN layer's stored (running_mean, running_var), detached clones."""
        m = self._modules_by_name[name]
        return m.running_mean.detach().clone(), m.running_var.detach().clone()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def snapshot_buffers(module):
    """Bit-exact snapshot of every named buffer of ``module``: {name: clone}."""
    return {name: buf.detach().clone() for name, buf in module.named_buffers()}


def assert_buffers_unchanged(module, snapshot):
    """Raise ``RuntimeError`` if any named buffer differs bit-wise from ``snapshot``.

    Used around a probe pass: the instrument must be read-only, or it corrupts
    the checkpoint's evidence (review finding 3 / plan §5 risk).
    """
    changed = [name for name, buf in module.named_buffers()
               if not torch.equal(buf, snapshot[name])]
    if changed:
        raise RuntimeError(
            f"Probe mutated {len(changed)} buffer(s) (must be read-only): {changed[:5]}"
        )


def _apply_md_variant(metadata, md_variant):
    """Apply ``md_variant`` to each per-sample metadata dict; ``None`` is identity."""
    if md_variant is None:
        return metadata
    return [md_variant(md) for md in metadata]


def build_report(ckpt_path, load_missing, load_unexpected, recorder, n_batches, n_samples):
    """Assemble the per-layer drift report + load/BN provenance."""
    per_layer = {}
    for name in recorder.layer_names:
        running_mean, running_var = recorder.running_stats(name)
        batch_mean, batch_var = recorder.input_stats(name)
        m = bn_drift_metrics(running_mean, running_var, batch_mean, batch_var)
        per_layer[name] = {
            "mean_shift_max": m["mean_shift_max"],
            "mean_shift_mean": m["mean_shift_mean"],
            "var_ratio_min": m["var_ratio_min"],
            "var_ratio_max": m["var_ratio_max"],
            "n_channels": int(running_mean.numel()),
        }
    return {
        "ckpt_path": ckpt_path,
        "load_missing": len(list(load_missing)),
        "load_unexpected": len(list(load_unexpected)),
        "n_bn_layers": len(per_layer),
        "per_layer": per_layer,
        "n_batches": n_batches,
        "n_samples": n_samples,
    }


def _load_flac_state_dict(ckpt_path, training_config):
    """Load a FLAC checkpoint, applying eval_FLAC.py's strip + EMA-remap in place.

    Strips the PL-wrapper ``diffusion.`` prefix, then (when ``use_ema`` and EMA
    buffers are present) remaps ``diffusion_ema.ema_model.`` -> ``model.`` so the
    probe compares against the exact EMA BN buffers W0 started from.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    for key in list(state_dict.keys()):
        if key.startswith("diffusion."):
            state_dict[key.replace("diffusion.", "", 1)] = state_dict.pop(key)
    if training_config.get("use_ema", False) and any(
        k.startswith("diffusion_ema.ema_model.") for k in state_dict
    ):
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                state_dict[key.replace("diffusion_ema.ema_model.", "model.", 1)] = state_dict.pop(key)
    return state_dict


def probe_rir_encoder(ckpt_path, dataset_config_path, n_batches, batch_size, device,
                      md_variant=None, seed=42):
    """Probe BN input drift for the released RIR encoder over N loader batches.

    Builds FLAC_AR, loads ``ckpt_path`` (strip/EMA-remap), records the
    RIRConditioner ResNet's BN inputs while streaming ``context_audio`` through
    the conditioner's own path (``only_ids=('context_audio',)``), and returns a
    ``build_report`` dict. Runs in eval mode (no running-stat mutation); GPU
    optional (falls back to CPU). ``md_variant`` transforms each metadata dict.

    Fail-fast (review finding 1): any missing/unexpected key or a BN count != 20
    raises -- the probe must measure the exact FLAC_EMA buffers, so there is no
    partial-load escape. Buffer bit-identity is asserted after the pass.
    """
    import json
    import pytorch_lightning as pl
    from src.data.dataset import create_dataloader_from_config
    from src.models import create_model_from_config

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    with open(_FLAC_AR_CONFIG) as f:
        model_config = json.load(f)
    training_config = model_config.get("training", {}) or {}

    model = create_model_from_config(model_config)
    state_dict = _load_flac_state_dict(ckpt_path, training_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} did not load cleanly: {len(missing)} missing keys "
            f"(first: {list(missing)[:5]}), {len(unexpected)} unexpected keys "
            f"(first: {list(unexpected)[:5]}). The probe must measure the exact "
            "FLAC_EMA buffers; there is no partial-load escape here."
        )
    model.eval().requires_grad_(False)
    model.to(device)

    rir_net = model.conditioner.conditioners["context_audio"].net
    recorder = BNInputRecorder(rir_net)
    if recorder.n_layers != 20:
        raise RuntimeError(
            f"Expected exactly 20 BatchNorm layers in the RIR encoder (AudioResNet18), "
            f"found {recorder.n_layers}: {recorder.layer_names}"
        )
    buffers_before = snapshot_buffers(rir_net)

    pl.seed_everything(seed, workers=True)
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)
    dataloader = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
        num_workers=4,  # persistent_workers=True (dataset.py) requires >0
        shuffle=True,
    )

    n_samples = 0
    n_done = 0
    with torch.no_grad():
        for batch in dataloader:
            metadata = _apply_md_variant(list(batch[1]), md_variant)
            model.conditioner(metadata, device, only_ids=("context_audio",))
            n_samples += len(metadata)
            n_done += 1
            if n_done >= n_batches:
                break

    # The instrument must be read-only: every buffer of the RIR net (all 60 BN
    # running_mean/running_var/num_batches_tracked) must be bit-identical.
    assert_buffers_unchanged(rir_net, buffers_before)

    report = build_report(ckpt_path, missing, unexpected, recorder, n_done, n_samples)
    recorder.remove()
    return report


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="BN-drift probe for the FLAC RIR encoder.")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--n-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="bn_drift_report.json",
                        help="Report JSON path, always written (default: ./bn_drift_report.json).")
    args = parser.parse_args()

    report = probe_rir_encoder(
        args.ckpt_path, args.dataset_config, args.n_batches, args.batch_size,
        args.device, seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    print(text)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"Report written to {args.out}")
