"""Waveform-level rotation visualization for the SimpleViT/CylViT ablation.

Produces the exp_02-style 4-panel figure (log-RMS envelope, raw-waveform zoom
at the max envelope gap, P_rot - P_0 difference trace, Schroeder energy decay)
for one ablation model, from the C4 prediction tensors stored by
``run_c16_eval.sh`` with ``STORE_PRED_C4=1``.

Adapted from ``worklog/worklog_yixun/exp_02_yaw_noninvariance_claude/
yaw_noninvariance_results_assets/gen_visuals.py`` (Yixun's figure for the
released FLAC model). Differences: parameterized over the ablation models and
checkpoint milestones, and the showcased sample is selected by default from
the SimpleViT envelope gaps (``--select-by simplevit``) so that both models'
figures show the SAME sample and are directly comparable.

Sample/prediction pairing: predictions are stored in eval order of the unseen
K=1 split with seed 42 and shuffle=False, which equals dataset order, so index
i of the prediction tensor pairs with the i-th dataloader item.
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rotviz")

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(p):
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(HERE)
sys.path.insert(0, REPO)

import numpy as np
import torch
import matplotlib.pyplot as plt

MODEL_LABELS = {"simplevit": "SimpleViT", "cylvit": "CylViT"}
COL = {"gt": "#8a887f", "p0": "#2a78d6", 90: "#1baf7a", 180: "#e34948", 270: "#eda100"}
C4 = (0, 90, 180, 270)


def pred_path(model: str, ckpt_stem: str, total_label: str, angle: int) -> str:
    save_dir = os.path.join(REPO, f"outputs_FLAC/exp05_{model}_phase3_total30k_s42")
    suffix = "" if angle == 0 else f"_rot{angle}"
    name = f"{ckpt_stem}_predictions_1_1.0_exp05_{model}_c16_total{total_label}_yaw{angle}{suffix}.pt"
    return os.path.join(save_dir, name)


def load_preds(model: str, ckpt_stem: str, total_label: str) -> dict:
    out = {}
    for angle in C4:
        path = pred_path(model, ckpt_stem, total_label, angle)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing predictions for {model} yaw {angle}: {path}\n"
                "Run run_c16_eval.sh with STORE_PRED_C4=1 first."
            )
        obj = torch.load(path, map_location="cpu")
        out[angle] = (obj["predictions"] if isinstance(obj, dict) else obj)[:, 0, :].numpy()
    return out


def env_db(x: np.ndarray, win: int = 256) -> np.ndarray:
    kernel = np.ones(win) / win
    envelope = np.sqrt(np.convolve(x.astype(np.float64) ** 2, kernel, mode="same")) + 1e-9
    return 20 * np.log10(envelope)


def edc_db(x: np.ndarray) -> np.ndarray:
    energy = np.flip(np.cumsum(np.flip(x.astype(np.float64) ** 2)))
    return 10 * np.log10(energy / (energy[0] + 1e-12) + 1e-12)


def select_index(preds: dict, quantile: float) -> tuple[int, float, float]:
    """Pick the sample at ``quantile`` of the mean |envelope-dB gap| (P180 vs P0)."""
    sub = slice(1000, 9000)
    p0, p180 = preds[0], preds[180]
    gaps = np.array([
        np.abs(env_db(p180[i])[sub] - env_db(p0[i])[sub]).mean() for i in range(p0.shape[0])
    ])
    order = np.argsort(gaps)
    idx = int(order[int(quantile * (len(order) - 1))])
    return idx, float(np.median(gaps)), float(gaps[idx])


def load_gt(idx: int, sample_rate: int, sample_size: int) -> np.ndarray:
    import pytorch_lightning as pl
    from src.data.dataset import create_dataloader_from_config

    dataset_config = json.load(
        open(os.path.join(REPO, "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json"))
    )
    pl.seed_everything(42, workers=True)
    dl = create_dataloader_from_config(
        dataset_config, batch_size=2, num_workers=2,
        sample_rate=sample_rate, sample_size=sample_size, audio_channels=1, shuffle=False,
    )
    need_batch, need_off = divmod(idx, 2)
    for bi, (reals, _) in enumerate(dl):
        if bi == need_batch:
            return reals[need_off, 0].numpy()
    raise IndexError(f"sample {idx} beyond dataset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_LABELS), required=True)
    parser.add_argument("--ckpt-stem", required=True, help="e.g. epoch=5-step=25000")
    parser.add_argument("--total-label", default="30k")
    parser.add_argument("--select-by", choices=("simplevit", "cylvit", "self"), default="simplevit",
                        help="Whose P180-vs-P0 envelope gaps pick the showcased sample "
                             "(default simplevit, so both models' figures show the SAME sample).")
    parser.add_argument("--select-ckpt-stem", default=None,
                        help="ckpt stem of the select-by model (defaults to --ckpt-stem).")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--index", type=int, default=None, help="Pin the sample index explicitly.")
    args = parser.parse_args()

    model = args.model
    preds = load_preds(model, args.ckpt_stem, args.total_label)

    sel_model = model if args.select_by == "self" else args.select_by
    if args.index is not None:
        idx = args.index
        med_gap = sel_gap = float("nan")
    else:
        sel_stem = args.select_ckpt_stem or args.ckpt_stem
        sel_preds = preds if sel_model == model else load_preds(sel_model, sel_stem, args.total_label)
        idx, med_gap, sel_gap = select_index(sel_preds, args.quantile)

    model_config = json.load(
        open(os.path.join(REPO, f"src/configs/model_configs/FLAC/AR/FLAC_AR_{'SimpleViT' if model == 'simplevit' else 'CylViT'}.json"))
    )
    sr, n = model_config["sample_rate"], model_config["sample_size"]
    gt = load_gt(idx, sr, n)

    p0 = preds[0]
    diff = preds[180][idx] - p0[idx]
    rel = float(np.sqrt((diff ** 2).sum()) / (np.sqrt((p0[idx] ** 2).sum()) + 1e-8))
    rels_all = np.sqrt(((preds[180] - p0) ** 2).sum(axis=1)) / (np.sqrt((p0 ** 2).sum(axis=1)) + 1e-8)
    med_rel = float(np.median(rels_all))

    e_gt, e_p0, e_p180 = env_db(gt[:n]), env_db(p0[idx]), env_db(preds[180][idx])
    sub = slice(1000, 9000)
    gpos = 1000 + int(np.argmax(np.abs(e_p180[sub] - e_p0[sub])))
    half = int(0.005 * sr)
    lo, hi = max(0, gpos - half), min(n, gpos + half)
    tz = np.arange(lo, hi) / sr * 1000
    tms = np.arange(n) / sr * 1000
    label = MODEL_LABELS[model]

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 12.4), constrained_layout=True)
    axes[0].plot(tms, e_gt, color=COL["gt"], lw=1.6, label="ground truth (rotation-invariant)")
    axes[0].plot(tms, e_p0, color=COL["p0"], lw=1.4, label="P₀ (unrotated)")
    axes[0].plot(tms, e_p180, color=COL[180], lw=1.4, label="P₁₈₀ (rotated 180°)")
    axes[0].axvspan(lo / sr * 1000, hi / sr * 1000, color="#eda100", alpha=0.18)
    axes[0].set_xlabel("time (ms)"); axes[0].set_ylabel("envelope (dB)")
    axes[0].set_ylim(bottom=max(e_gt.min(), -95))
    axes[0].set_title(
        f"{label}: log RMS envelope (sample #{idx}) — P₀ vs P₁₈₀ under rotated conditioning (shaded: zoom below)",
        fontsize=11,
    )
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(tz, p0[idx, lo:hi], color=COL["p0"], lw=1.3, label="P₀")
    axes[1].plot(tz, preds[180][idx, lo:hi], color=COL[180], lw=1.3, alpha=0.9, label="P₁₈₀")
    axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("amplitude")
    axes[1].set_title(f"{label}: raw waveform, 10 ms at the max envelope gap", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].plot(tms, diff, color=COL[180], lw=0.9)
    axes[2].set_xlabel("time (ms)"); axes[2].set_ylabel("P₁₈₀ − P₀")
    axes[2].set_title(
        f"{label}: difference trace — waveform rel-L2 = {rel:.3f} (dataset median {med_rel:.3f}); "
        "an invariant model gives a flat zero line",
        fontsize=11,
    )

    axes[3].plot(edc_db(gt[:n]), color=COL["gt"], lw=1.6, label="ground truth")
    axes[3].plot(edc_db(p0[idx]), color=COL["p0"], lw=1.4, label="P₀")
    for angle in (90, 180, 270):
        axes[3].plot(edc_db(preds[angle][idx]), color=COL[angle], lw=1.1, alpha=0.85, label=f"P{angle}°")
    axes[3].set_ylim(-80, 2); axes[3].set_xlabel("sample"); axes[3].set_ylabel("energy decay (dB)")
    axes[3].set_title(
        f"{label}: Schroeder energy decay — C4 rotations vs P₀ (the T60/EDT view)", fontsize=11
    )
    axes[3].legend(loc="upper right", fontsize=9)

    out_dir = os.path.join(HERE, "figures")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"rir_rotation_{model}_{args.total_label}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=110)
    plt.close(fig)
    print(
        f"wrote figures/{stem}.png (model {model}, sample {idx} selected by {sel_model} "
        f"q{args.quantile:.2f}: env-gap {sel_gap:.2f} dB [median {med_gap:.2f}]; "
        f"rel-L2 {rel:.3f}, dataset median rel-L2 {med_rel:.3f})"
    )


if __name__ == "__main__":
    main()
