#!/usr/bin/env python3
"""exp_11 Q5 — batched-orbit equivalence probe on the REAL conditioner stack.

``src/data/yaw_rotation.py`` now executes the frame-average orbit as a few large
batched forwards instead of one forward per angle. The unit tests prove the two
orderings agree on synthetic conditioners; this probe proves it on the actual
DINOv3 ViT stack built from ``FLAC_AR_BF_C32.json`` with real AcousticRooms
samples, in both numeric regimes the training path uses:

    fp32            the reference regime          — pre-registered rel <= 1e-6
    bf16 autocast   the training regime           — pre-registered rel <= 2e-3

The bf16 tolerance is the D3-floor class: bf16 has ~3 decimal digits of mantissa,
so re-associating an orbit sum necessarily moves the last bits. A deviation ABOVE
these bounds would mean the batching changed the maths, not the rounding.

The reference side calls the library's own ``_orbit_average_loop`` (the preserved
pre-batching order), so the probe compares real code against real code rather
than against a copy that could drift. No training checkpoint is needed: the ViT
is the pinned pretrained DINOv3 initialisation, which is what the arms start
from, and the comparison is order-of-summation, not weight-dependent.

Emits exactly one machine-parseable line:

    EQUIVPROBE cfg=<sha12> nsamples=<n> cells=<k> max_rel_fp32=<r> \\
        max_rel_bf16=<r> max_abs_fp32=<a> max_abs_bf16=<a> verdict=<PASS|FAIL>
"""
import argparse
import hashlib
import json
import os
import sys

TOL_REL_FP32 = 1e-6
TOL_REL_BF16 = 2e-3
ORBITS = (4, 32)


def _repo_root(p):
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))

import torch  # noqa: E402

from src.data import yaw_rotation as yr  # noqa: E402


def _orbit(n):
    return tuple(k * 360.0 / n for k in range(n))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_conditioner(config_path, device):
    """The real MultiConditioner from the arm config (pretrained ViT init)."""
    from src.models.factory import create_model_from_config
    cfg = json.load(open(config_path))
    model = create_model_from_config(cfg)
    cond = model.conditioner.to(device)
    cond.eval()
    return cfg, cond


def real_samples(dataset_config_path, model_config, n_samples, device):
    """The first ``n_samples`` items of the training split, deterministically."""
    from src.data.dataset import create_dataloader_from_config
    ds_cfg = json.load(open(dataset_config_path))
    dl = create_dataloader_from_config(
        ds_cfg, batch_size=n_samples, num_workers=0,
        sample_rate=model_config["sample_rate"], sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
    )
    batch = next(iter(dl))
    metadata = batch[1] if isinstance(batch, (list, tuple)) else batch["metadata"]
    md = list(metadata)[:n_samples]
    out = []
    for m in md:
        out.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in m.items()})
    return out


def _reference(cond, metadata, device, angles):
    """Orbit average with the PRE-BATCHING order, via the library's own helper."""
    md_inv = [yr.cylindrical_pose_features(m) for m in metadata]
    base = cond(md_inv, device)
    present = [i for i in ("source_vit", "context_poses_vit") if i in base]
    img_w = int(metadata[0]["depth"].shape[-1])
    accum = yr._orbit_average_loop(cond, md_inv, base, present, angles, img_w, device)
    for i in present:
        base[i][0] = accum[i] / float(len(angles))
    return base


def compare(cond, md, device, angles, use_bf16):
    """max |abs| and max relative deviation, batched vs reference, per ViT id."""
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 else torch.autocast(device_type="cuda", enabled=False))
    per_id = {}
    with torch.no_grad(), ctx:
        got = yr.invariant_conditioning(cond, md, device, angles)
        want = _reference(cond, md, device, angles)
    for key in ("source_vit", "context_poses_vit"):
        if key not in got:
            continue
        a = got[key][0].float()
        b = want[key][0].float()
        abs_err = (a - b).abs()
        denom = b.abs().clamp_min(1e-12)
        per_id[key] = (float(abs_err.max()), float((abs_err / denom).max()))
    return per_id


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp_11 batched-orbit equivalence probe")
    ap.add_argument("--config", default=os.path.join(HERE, "FLAC_AR_BF_C32.json"))
    ap.add_argument("--dataset-config",
                    default=os.path.join(REPO, "src/configs/dataset_configs/AR/train/acousticroom_train.json"))
    ap.add_argument("--n-samples", type=int, default=4)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    cfg, cond = build_conditioner(args.config, device)
    md = real_samples(args.dataset_config, cfg, args.n_samples, device)
    print(f"probe: device={device} samples={len(md)} config={os.path.basename(args.config)} "
          f"cap={yr.FRAME_AVG_MAX_FWD_SAMPLES}")

    worst = {"fp32": (0.0, 0.0), "bf16": (0.0, 0.0)}
    cells = 0
    for n in ORBITS:
        angles = _orbit(n)
        for label, use_bf16 in (("fp32", False), ("bf16", True)):
            if use_bf16 and device != "cuda":
                print(f"  C{n:<3} {label}: SKIPPED (bf16 autocast needs a GPU)")
                continue
            per_id = compare(cond, md, device, angles, use_bf16)
            cells += 1
            for key, (abs_err, rel_err) in sorted(per_id.items()):
                print(f"  C{n:<3} {label} {key:<18} max_abs={abs_err:.3e} max_rel={rel_err:.3e}")
                worst[label] = (max(worst[label][0], abs_err), max(worst[label][1], rel_err))

    ok_fp32 = worst["fp32"][1] <= TOL_REL_FP32
    ok_bf16 = worst["bf16"][1] <= TOL_REL_BF16 or device != "cuda"
    verdict = "PASS" if (ok_fp32 and ok_bf16 and cells) else "FAIL"
    print(f"tolerances: fp32 rel <= {TOL_REL_FP32:g} ({'ok' if ok_fp32 else 'EXCEEDED'}), "
          f"bf16 rel <= {TOL_REL_BF16:g} ({'ok' if ok_bf16 else 'EXCEEDED'})")
    print(f"EQUIVPROBE cfg={_sha256(args.config)[:12]} nsamples={len(md)} cells={cells} "
          f"max_rel_fp32={worst['fp32'][1]:.3e} max_rel_bf16={worst['bf16'][1]:.3e} "
          f"max_abs_fp32={worst['fp32'][0]:.3e} max_abs_bf16={worst['bf16'][0]:.3e} "
          f"verdict={verdict}")
    return 0 if verdict == "PASS" else 4


if __name__ == "__main__":
    sys.exit(main())
