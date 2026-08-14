#!/usr/bin/env python3
"""exp_16 (are_port) — calibrate the ARE anchor's two constants on the AR TRAIN split.

Seat: Opus 5 Coder (SOP §Roles). Plan §2; estimator design inherited from rir2rir
``exp_15_anchor/plan_anchor.md`` §4.b and re-implemented against FLAC's loader.

WHAT IS CALIBRATED, AND WHY IT IS NOT ASSUMED
---------------------------------------------
``delta_hat`` — the offset between the ANALYTIC arrival ``r/343*fs`` and the
measured direct peak, in samples. It is measured rather than assumed because the
program record contains a withdrawn constant-offset claim: a MEAN offset of
+1.774 ms was once adopted and later withdrawn, the true distribution being a
dominant near-zero core plus a long positive tail from occluded / weak-direct
paths. A mean is a broken estimator of that quantity, so this script uses a
**local** search (a +/-W-sample window around the analytic arrival, not a global
argmax) and the **median**.

``A_g`` — the constant of the ``g = A_g / r`` amplitude law, in the SAME l2-energy
convention the skeleton uses (``a_i = ||x[t_hat +/- H]||_2 * r_i``), so that the
calibrated scale and the synthesised scale cannot drift apart. A peak-keyed
constant would couple the amplitude to the sub-sample phase, which swings 3.92 dB.

PRE-REGISTERED DECISION RULES (recorded in the output, applied here)
--------------------------------------------------------------------
* **R1** ``|delta_hat| < 0.5`` samples -> ``delta_hat := 0.0``. Half a sample IS
  the rounding bound: below it the residual is quantisation, not an onset offset,
  and the calibration is a CONFIRMATION of the record's withdrawal rather than a
  fit. Written before the numbers were seen.
* **R2** a Theil-Sen slope of ``delta_i`` on ``r_i`` large enough to move the
  offset by more than one sample across the observed distance range -> reported
  as ``escalate_r2``: a single global offset would be inadequate.
* **R3** the log-log amplitude exponent outside ``[0.7, 1.3]`` -> ``escalate_r3``:
  the 1/r law does not hold on this data.
* **R4** more than 5 % of local argmaxes on the window edge -> ``escalate_r4``:
  the search window is truncating the estimator.

The script REPORTS R2-R4; it never silently self-resolves them. All four flags,
plus every statistic computed and not used, land in the JSON.

DETERMINISM. The path draw is ``numpy.random.default_rng(1508).choice`` over the
SORTED path list, and paths are processed in the drawn order at batch 1. Two runs
with the same arguments produce byte-identical output apart from ``created``.

LOS RESTRICTION (default on). The constants describe a DIRECT-PATH law, so they
are fitted on paths the anchor will actually carry one for — using the identical
``are_anchor.line_of_sight`` rule and threshold that gate the anchor itself. NLOS
statistics are retained and reported, never discarded.

Usage (production):

  python worklog/worklog_yixun/exp_16_are_port_claude/calibrate_delta.py \
      --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
      --model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json \
      --n-paths 2048
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torchaudio

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:                    # beat any stale pip-installed src copy
    sys.path.insert(0, _REPO)

from src.data import are_anchor as ar        # noqa: E402

DRAW_SEED = 1508                             # pinned; the drawn list is recorded
SEARCH_WINDOW = 45                           # +/- samples (+/-2.04 ms at 22,050 Hz)
ENERGY_HALF_WIDTHS = (8, 16, 32, 64)         # A_g window sensitivity, reported
R1_BOUND = 0.5                               # samples
R3_BAND = (0.7, 1.3)
R4_EDGE_FRACTION = 0.05


# --------------------------------------------------------------------------- #
# split / metadata plumbing (mirrors AR_md, without importing the dataloader)
# --------------------------------------------------------------------------- #
def load_split_relpaths(split_json):
    """Every ``scene/scene_id/file.wav`` in an AR split file, SORTED.

    AR splits are nested ``{scene: {scene_id: [filenames]}}`` — the same shape
    ``src/data/dataset.py::json_scandir`` walks — and the resulting relative path
    is exactly what ``AR_md`` parses for the metadata and depth lookups. A flat
    ``{scene: [relpaths]}`` form is also accepted, for fixtures.

    Sorting makes the draw's population order machine-independent, which is what
    makes the seeded draw reproducible.
    """
    with open(split_json) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{split_json}: expected a dict of scenes")
    paths = []
    for scene in sorted(payload):
        entry = payload[scene]
        if isinstance(entry, dict):
            for scene_id in sorted(entry):
                paths.extend(f"{scene}/{scene_id}/{fn}" for fn in entry[scene_id])
        elif isinstance(entry, list):
            paths.extend(str(p) for p in entry)
        else:
            raise ValueError(
                f"{split_json}: scene {scene!r} holds {type(entry).__name__}, expected "
                "a dict of scene ids or a list of relative paths")
    return sorted(paths)


def metadata_for(audio_root, relpath):
    """``(src_loc, rec_loc)`` for one IR path — the same S/R json ``AR_md`` reads."""
    scene_name, scene_id, filename = relpath.split("/")[-3:]
    src_node, rec_node = (int(filename.split("_")[0][1:]),
                          int(filename.split("_")[1][1:]))
    md_path = os.path.join(audio_root, "metadata", scene_name, scene_id,
                           f"S00{src_node}_R00{rec_node}.json")
    with open(md_path) as f:
        info = json.load(f)
    return info["src_loc"], info["rec_loc"]


def _load_depth(audio_root, relpath, img_h=256, img_w=512):
    """The listener's depth panorama as a ``[3, H, W]`` point cloud.

    Reproduces ``AR_md.convert_equirect_to_camera_coord`` exactly, so the gate
    here and the gate inside the anchor read the same geometry — one rule, one
    convention, no second implementation to drift.
    """
    scene_name, scene_id, filename = relpath.split("/")[-3:]
    rec_node = int(filename.split("_")[1][1:])
    pano = np.load(os.path.join(audio_root, "depth_map", scene_name, scene_id,
                                f"{rec_node}.npy"))
    depth = torch.from_numpy(pano)
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing="ij")
    theta_map = (theta + 0.5) * 2.0 * math.pi / img_w - math.pi
    phi_map = (phi + 0.5) * math.pi / img_h - math.pi / 2
    cloud = torch.stack([depth * torch.cos(phi_map) * torch.cos(theta_map),
                         depth * torch.cos(phi_map) * torch.sin(theta_map),
                         -depth * torch.sin(phi_map)], dim=-1)
    return cloud.permute(2, 0, 1).contiguous()


def load_waveform(path, sample_rate, sample_size):
    """The deployed load: mono, resampled, cropped/padded, clamped."""
    wav, rate = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if rate != sample_rate:
        wav = torchaudio.functional.resample(wav, rate, sample_rate)
    if wav.shape[1] < sample_size:
        wav = torch.nn.functional.pad(wav, (0, sample_size - wav.shape[1]))
    return wav[:1, :sample_size].clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def theil_sen_slope(x, y, max_pairs=200000, rng=None):
    """Median pairwise slope — robust to the tail the mean is broken on."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    if n < 2:
        return float("nan")
    idx_i, idx_j = np.triu_indices(n, k=1)
    if idx_i.size > max_pairs:
        pick = (rng or np.random.default_rng(0)).choice(idx_i.size, max_pairs,
                                                        replace=False)
        idx_i, idx_j = idx_i[pick], idx_j[pick]
    dx = x[idx_j] - x[idx_i]
    keep = dx != 0
    if not keep.any():
        return float("nan")
    return float(np.median((y[idx_j] - y[idx_i])[keep] / dx[keep]))


def measure_path(wave, r, sample_rate, window=SEARCH_WINDOW):
    """Local direct-peak measurement for one path.

    The search is LOCAL by design: a global argmax lands somewhere other than the
    analytic arrival for a substantial minority of paths (weak or occluded direct
    sound), and calibrating on those would fit a law to data it does not describe.
    """
    x = wave.reshape(-1)
    n_analytic = r / ar.C_SOUND * sample_rate
    centre = int(round(n_analytic))
    lo = max(0, centre - window)
    hi = min(x.numel(), centre + window + 1)
    if hi <= lo:
        return None
    seg = x[lo:hi].abs()
    local = int(torch.argmax(seg))
    t_hat = lo + local
    on_edge = bool(local == 0 or local == seg.numel() - 1)
    energies = {}
    for half in ENERGY_HALF_WIDTHS:
        a, b = max(0, t_hat - half), min(x.numel(), t_hat + half + 1)
        energies[half] = float(torch.linalg.vector_norm(x[a:b]))
    return {
        "r": float(r),
        "t_hat": float(t_hat),
        "delta": float(t_hat - n_analytic),
        "peak_abs": float(x[t_hat].abs()),
        "l2": energies,
        "on_edge": on_edge,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-config", default=None,
                    help="AR TRAIN dataset config; supplies the audio root, the "
                         "split json and the folder name")
    ap.add_argument("--model-config", default=None,
                    help="model config; supplies sample_rate and sample_size")
    ap.add_argument("--audio-root", default=None, help="overrides the dataset config")
    ap.add_argument("--folder-name", default=None, help="overrides the dataset config")
    ap.add_argument("--split-json", default=None, help="overrides the dataset config")
    ap.add_argument("--sample-rate", type=int, default=None)
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--n-paths", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=DRAW_SEED)
    ap.add_argument("--window", type=int, default=SEARCH_WINDOW)
    ap.add_argument("--no-los-filter", action="store_true",
                    help="calibrate on ALL drawn paths instead of the LOS subset "
                         "(diagnostic / fixture use only)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "are_calibration.json"))
    args = ap.parse_args(argv)

    audio_root, folder_name, split_json = args.audio_root, args.folder_name, args.split_json
    if args.dataset_config:
        with open(args.dataset_config) as f:
            ds = json.load(f)
        entry = ds["datasets"][0]
        audio_root = audio_root or os.path.join(_REPO, entry["path"])
        folder_name = folder_name or entry.get("folder_name")
        split_json = split_json or os.path.join(_REPO, entry["json_file_path"])
    sample_rate, sample_size = args.sample_rate, args.sample_size
    if args.model_config:
        with open(args.model_config) as f:
            mc = json.load(f)
        sample_rate = sample_rate or mc["sample_rate"]
        sample_size = sample_size or mc["sample_size"]

    missing = [n for n, v in (("audio-root", audio_root), ("folder-name", folder_name),
                              ("split-json", split_json), ("sample-rate", sample_rate),
                              ("sample-size", sample_size)) if not v]
    if missing:
        print(f"missing required argument(s): {missing} (pass them, or a "
              "--dataset-config / --model-config that supplies them)")
        return 2

    print(f"audio root : {audio_root}")
    print(f"split      : {split_json}  (folder {folder_name})")
    print(f"time base  : fs={sample_rate}  sample_size={sample_size}  "
          f"window=+/-{args.window} samples")
    print(f"draw       : rng({args.seed}).choice over the sorted split, "
          f"n={args.n_paths}, LOS filter={'off' if args.no_los_filter else 'on'}")

    population = load_split_relpaths(split_json)
    rng = np.random.default_rng(args.seed)
    take = min(args.n_paths, len(population))
    drawn = [population[i] for i in
             sorted(rng.choice(len(population), take, replace=False).tolist())]

    gate_cfg = ar.AnchorConfig(sample_rate=int(sample_rate), sample_size=int(sample_size),
                               hop=1024, delta_hat=0.0, a_g=1.0)

    rows, scanned, nlos, failed = [], 0, 0, 0
    for rel in drawn:
        scanned += 1
        try:
            src_loc, rec_loc = metadata_for(audio_root, rel)
            source = torch.tensor([s - r for s, r in zip(src_loc, rec_loc)],
                                  dtype=torch.float32)
            r = float(torch.linalg.vector_norm(source))
            if not args.no_los_filter:
                depth = _load_depth(audio_root, rel)
                if not ar.line_of_sight(depth, source, gate_cfg):
                    nlos += 1
                    continue
            wave = load_waveform(os.path.join(audio_root, folder_name, rel),
                                 int(sample_rate), int(sample_size))
        except Exception as exc:                       # a broken path is reported,
            failed += 1                                # never silently averaged in
            if failed <= 5:
                print(f"  skipped {rel}: {type(exc).__name__}: {exc}")
            continue
        row = measure_path(wave, r, int(sample_rate), args.window)
        if row is not None:
            rows.append(row)

    if not rows:
        print("no usable paths — refusing to emit a calibration")
        return 2

    deltas = np.array([row["delta"] for row in rows])
    radii = np.array([row["r"] for row in rows])
    a_l2 = {half: np.array([row["l2"][half] * row["r"] for row in rows])
            for half in ENERGY_HALF_WIDTHS}
    a_peak = np.array([row["peak_abs"] * row["r"] for row in rows])

    delta_raw = float(np.median(deltas))
    r1_fired = abs(delta_raw) < R1_BOUND
    delta_hat = 0.0 if r1_fired else delta_raw
    a_g = float(np.median(a_l2[32]))

    slope = theil_sen_slope(radii, deltas, rng=np.random.default_rng(args.seed))
    r_span = float(np.percentile(radii, 99) - np.percentile(radii, 1))
    r2_swing = float(abs(slope) * r_span) if math.isfinite(slope) else float("nan")
    finite = (a_peak > 0) & (radii > 0)
    if finite.sum() >= 2:
        p_hat = float(-np.polyfit(np.log(radii[finite]),
                                  np.log(a_peak[finite] / radii[finite]), 1)[0])
    else:
        p_hat = float("nan")
    edge_fraction = float(np.mean([row["on_edge"] for row in rows]))

    try:
        source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                             stderr=subprocess.DEVNULL,
                                             timeout=30).decode().strip()
    except Exception:
        source_sha = "unknown"

    record = {
        "schema": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_sha": source_sha,
        # --- what was calibrated, and what goes into the config ---------------
        "delta_hat": delta_hat,
        "a_g": a_g,
        "p_exp": ar.DEFAULT_P_EXP,
        # --- provenance -------------------------------------------------------
        "split_json": os.path.relpath(split_json, _REPO) if split_json.startswith(_REPO) else split_json,
        "audio_root": audio_root,
        "folder_name": folder_name,
        "sample_rate": int(sample_rate),
        "sample_size": int(sample_size),
        "search_window": int(args.window),
        "draw_seed": int(args.seed),
        "n_requested": int(args.n_paths),
        "n_population": len(population),
        "n_scanned": scanned,
        "n_paths": len(rows),
        "n_nlos_rejected": nlos,
        "n_failed": failed,
        "los_filter": not args.no_los_filter,
        "los_threshold": gate_cfg.los_threshold,
        # --- the estimators, and everything reported but NOT fitted ------------
        "delta_hat_raw": delta_raw,
        "r1_bound_samples": R1_BOUND,
        "r1_fired": bool(r1_fired),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_mad": float(np.median(np.abs(deltas - delta_raw))),
        "delta_iqr": [float(np.percentile(deltas, 25)), float(np.percentile(deltas, 75))],
        "delta_percentiles": {str(p): float(np.percentile(deltas, p))
                              for p in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
        "delta_ms_median": float(delta_raw / sample_rate * 1000.0),
        "r_percentiles": {str(p): float(np.percentile(radii, p))
                          for p in (0, 1, 50, 99, 100)},
        "a_g_by_half_width": {str(h): float(np.median(v)) for h, v in a_l2.items()},
        "a_g_peak_convention": float(np.median(a_peak)),
        "theil_sen_slope_samples_per_m": slope,
        "r2_swing_samples": r2_swing,
        "amplitude_exponent_p_hat": p_hat,
        "edge_saturation_fraction": edge_fraction,
        # --- the pre-registered escalation flags -------------------------------
        "escalate_r2": bool(math.isfinite(r2_swing) and r2_swing > 1.0),
        "escalate_r3": bool(not (R3_BAND[0] <= p_hat <= R3_BAND[1])),
        "escalate_r4": bool(edge_fraction > R4_EDGE_FRACTION),
        "drawn_paths": drawn,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\nscanned {scanned} drawn path(s): {len(rows)} measured, {nlos} NLOS-rejected, "
          f"{failed} unreadable")
    print(f"delta_hat_raw = {delta_raw:+.4f} samples ({record['delta_ms_median']:+.4f} ms)"
          f"  -> delta_hat = {delta_hat:+.4f}  (R1 {'FIRED' if r1_fired else 'not fired'})")
    print(f"A_g (l2, H=32) = {a_g:.6f}   [H=8/16/64: "
          + ", ".join(f"{record['a_g_by_half_width'][str(h)]:.6f}"
                      for h in (8, 16, 64)) + "]")
    print(f"R2 Theil-Sen swing {r2_swing:.3f} samples over the r range -> "
          f"{'ESCALATE' if record['escalate_r2'] else 'ok'}")
    print(f"R3 amplitude exponent p_hat {p_hat:.3f} -> "
          f"{'ESCALATE' if record['escalate_r3'] else 'ok'}")
    print(f"R4 edge saturation {edge_fraction:.4f} -> "
          f"{'ESCALATE' if record['escalate_r4'] else 'ok'}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
