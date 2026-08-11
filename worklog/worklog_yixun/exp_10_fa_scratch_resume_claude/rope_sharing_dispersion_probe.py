#!/usr/bin/env python3
"""A5 v2 — how much augmentation noise survives the frame average, as a function
of how many orbit angles share one RoPE draw.

v1 (WITHDRAWN, see a5_codex_code_review.md) compared rungs, which changed the
sample set, the GEMM shapes and the sharing all at once, used n=3, and ran on an
unseeded random projection. v2 fixes all three:

  * FIXED batch and FIXED samples for every condition — only the chunk cap moves.
    `angles_per_chunk = cap // batch` (src/data/yaw_rotation.py:501), so at B=32
    caps 32/64/96 give 1/2/3 angles per chunk = per-angle draws (the July path),
    partial sharing (our rung today), and full sharing (exp_11's rung today).
  * SEEDED model construction, with the output projection hashed into the record.
  * The statistic is WITHIN-schedule, so it never compares two single realisations:
    for each cap, draw N seeds, form the Monte-Carlo mean of the frame-averaged
    conditioning, and report the mean relative dispersion of each seed about it.
    That is exactly "noise left after averaging" — the quantity the mechanism is
    about — and it is comparable across caps.

PRE-REGISTERED PREDICTION: dispersion(1 angle/chunk) < dispersion(2) < dispersion(3).
CONTROL: the same sweep in eval() mode, where the RoPE rescale is off — all caps
must collapse to the numerical floor (~0). That floor is what makes a train-mode
dispersion interpretable rather than an artefact of bf16.

Measures conditioning tensors only: no training, no weights written.
"""
import argparse, hashlib, importlib.util, json, os, statistics as st, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROBE = os.path.join(ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
                     "fa_orbit_equiv_probe.py")


def load_helpers():
    spec = importlib.util.spec_from_file_location("fa_orbit_equiv_probe", PROBE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def dispersion(mats):
    """mean_s ||x_s - xbar|| / ||xbar||  over the seed dimension (float32)."""
    import torch
    stack = torch.stack(mats)                      # [S, ...]
    xbar = stack.mean(0)
    den = xbar.norm()
    return float(sum((m - xbar).norm() / den for m in stack) / len(stack)), float(den)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "worklog", "worklog_yixun",
                                                     "exp_07_fa_scratch_claude", "FLAC_AR_BF.json"))
    ap.add_argument("--dataset-config", default=os.path.join(
        ROOT, "src", "configs", "dataset_configs", "AR", "eval", "acousticroom_unseeneval.json"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--caps", default="32,64,96")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "a5v2_rope_sharing_dispersion.json"))
    a = ap.parse_args(argv)

    P = load_helpers()
    import torch
    from src.data import yaw_rotation as yr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("this probe needs CUDA: the bf16 train path is the thing being measured")
    P.assert_vit_pin()

    torch.manual_seed(a.init_seed); torch.cuda.manual_seed_all(a.init_seed)   # seeded construction
    _cfg, cond = P.build_conditioner(a.config, device)
    init_hash = hashlib.sha256(b"".join(
        p.detach().float().cpu().numpy().tobytes() for p in cond.parameters())).hexdigest()[:16]
    md, sample_ids = P.real_samples(a.dataset_config, a.batch, device, seed=42)   # FIXED samples
    angles = P.orbit(4)
    caps = [int(c) for c in a.caps.split(",")]
    live_cap = yr.FRAME_AVG_MAX_FWD_SAMPLES
    print(f"live FRAME_AVG_MAX_FWD_SAMPLES={live_cap}  batch={a.batch}  init_hash={init_hash}")

    out, rows = {}, []
    for mode in ("train", "eval"):
        cond.train(mode == "train")
        for cap in caps:
            if cap < a.batch:
                raise ValueError(f"cap {cap} < batch {a.batch}: a chunk cannot be smaller than one angle")
            per_chunk = max(1, cap // a.batch)
            shared = min(per_chunk, len(angles) - 1)
            saved, yr.FRAME_AVG_MAX_FWD_SAMPLES = yr.FRAME_AVG_MAX_FWD_SAMPLES, cap
            try:
                acc = {}
                for s in range(a.seeds):
                    torch.manual_seed(1000 + s); torch.cuda.manual_seed_all(1000 + s)
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        got = yr.invariant_conditioning(cond, md, device, angles)
                    for vit in P.VIT_IDS:
                        if vit in got:
                            acc.setdefault(vit, []).append(got[vit][0].detach().float().cpu())
                    del got
                    torch.cuda.empty_cache()
            finally:
                yr.FRAME_AVG_MAX_FWD_SAMPLES = saved
            for vit, mats in acc.items():
                d, den = dispersion(mats)
                out[f"{mode}|cap{cap}|{vit}"] = {"dispersion": d, "mean_norm": den,
                                                 "angles_per_chunk": per_chunk, "shared_angles": shared}
                rows.append({"mode": mode, "cap": cap, "vit": vit, "dispersion": d,
                             "angles_per_chunk": per_chunk, "shared_angles": shared})
                print(f"  {mode:<5} cap={cap:>3} ({shared} of {len(angles)-1} angles share a draw)  "
                      f"{vit:<18} dispersion {d:.3e}")

    print("\n=== A5 v2 verdict (prediction: dispersion grows with sharing, in train mode) ===")
    for vit in sorted({r['vit'] for r in rows}):
        tr = {r["shared_angles"]: r["dispersion"] for r in rows if r["mode"] == "train" and r["vit"] == vit}
        ev = {r["shared_angles"]: r["dispersion"] for r in rows if r["mode"] == "eval" and r["vit"] == vit}
        order = [tr[k] for k in sorted(tr)]
        mono = all(x < y for x, y in zip(order, order[1:]))
        floor = max(ev.values()) if ev else float("nan")
        print(f"  {vit:<18} train " + " < ".join(f"{v:.3e}" for v in order) +
              f"   | eval floor {floor:.2e} -> {'MONOTONE (prediction held)' if mono else 'NOT monotone'}"
              f"{'  [train >> floor]' if order and order[0] > 10*floor else '  [NOT clearly above floor]'}")
    rev = subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"rows": rows, "summary": out, "batch": a.batch, "caps": caps, "seeds": a.seeds,
               "init_seed": a.init_seed, "init_hash": init_hash, "sample_ids": sample_ids,
               "live_cap_default": live_cap, "config": a.config, "dataset_config": a.dataset_config,
               "gpu": torch.cuda.get_device_name(0), "git": rev},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
