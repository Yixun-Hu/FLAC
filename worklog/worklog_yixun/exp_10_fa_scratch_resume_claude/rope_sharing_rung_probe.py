#!/usr/bin/env python3
"""A5 — does the chunk-shared RoPE draw hit the two rungs differently?

Context. exp_10/exp_07's FA arm (B-F) was trained in July, before the batched
orbit existed: every frame angle drew its own DINOv3 RoPE rescale. exp_11's C4L
was trained with the batched orbit, where a chunk's angles SHARE one draw
(`src/data/yaw_rotation.py:501`):

    angles_per_chunk = max(1, FRAME_AVG_MAX_FWD_SAMPLES // batch)   # constant 64

With C4 (three non-zero angles) that makes the sharing RUNG-DEPENDENT:
  * micro-8  (exp_11's rung) -> 8 angles/chunk -> all three share ONE draw;
  * micro-32 (our rung)      -> 2 angles/chunk -> {90,180} share, 270 separate;
  * micro-64                 -> 1 angle/chunk  -> per-angle draws (the July path).

Mechanism under test: frame averaging works by averaging the augmentation noise
away over the orbit. Sharing a draw correlates that noise, so it survives the
average. PRE-REGISTERED PREDICTION: the batched-vs-loop train-mode deviation is
LARGER at micro-8 than at micro-32 (more angles sharing one draw). A null result
(deviations comparable, or both negligible) would say the reversal exp_11 reports
is NOT explained by draw sharing, and points back at the rung/topology instead.

This probe measures conditioning tensors only: no training, no weights written.
It reuses exp_11's reviewed helpers (build_conditioner / real_samples / run_cell)
rather than reimplementing the stack; the only new axis is the batch size.

Usage:  HF_HUB_OFFLINE=1 python <this> [--batches 8,32] [--reps 3]
"""
import argparse, importlib.util, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                      # repo root before any stale site-packages copy
PROBE = os.path.join(ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
                     "fa_orbit_equiv_probe.py")


def load_exp11_probe():
    spec = importlib.util.spec_from_file_location("fa_orbit_equiv_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "worklog", "worklog_yixun",
                                                     "exp_07_fa_scratch_claude", "FLAC_AR_BF.json"))
    ap.add_argument("--dataset-config", default=os.path.join(
        ROOT, "src", "configs", "dataset_configs", "AR", "eval", "acousticroom_unseeneval.json"))
    ap.add_argument("--batches", default="8,32", help="micro-batch sizes to compare")
    ap.add_argument("--reps", type=int, default=3, help="independent RNG seeds per rung")
    ap.add_argument("--out", default=os.path.join(HERE, "a5_rope_sharing_rung.json"))
    a = ap.parse_args(argv)

    P = load_exp11_probe()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P.assert_vit_pin()                                   # same DINOv3 pin gate as every arm
    _cfg, cond = P.build_conditioner(a.config, device)          # (cfg, conditioner)
    angles = P.orbit(4)                                  # C4, matching both arms under dispute

    results, rows = {}, []
    for B in [int(x) for x in a.batches.split(",")]:
        per_rep = []
        md, _ids = P.real_samples(a.dataset_config, B, device, seed=42)   # (metadata, sample_ids)
        for r in range(a.reps):
            cell = P.run_cell(cond, md, device, angles, mode="train",
                              use_bf16=(device == "cuda"), seed=1234 + r)
            if not cell["finite"]:
                raise RuntimeError(f"non-finite tensors at B={B}, rep {r}")
            for vit, m in cell["ids"].items():
                per_rep.append((vit, r, m["rel_norm"], m["max_abs"]))
                rows.append({"batch": B, "vit": vit, "rep": r,
                             "rel_norm": m["rel_norm"], "max_abs": m["max_abs"]})
        by_vit = {}
        for vit in sorted({v for v, _, _, _ in per_rep}):
            rn = [x[2] for x in per_rep if x[0] == vit]
            by_vit[vit] = {"rel_norm_mean": st.mean(rn),
                           "rel_norm_max": max(rn),
                           "angles_per_chunk": max(1, 64 // B),
                           "shared_angles": min(max(1, 64 // B), len(angles) - 1)}
        results[B] = by_vit
        for vit, s in by_vit.items():
            print(f"B={B:>3} ({s['shared_angles']} of {len(angles)-1} non-zero angles share a draw)  "
                  f"{vit:<18} batched-vs-loop rel_norm mean {s['rel_norm_mean']:.3e}  max {s['rel_norm_max']:.3e}")

    Bs = sorted(results)
    if len(Bs) == 2:
        lo, hi = Bs[0], Bs[1]                            # lo = smaller micro-batch = more sharing
        print("\n=== A5 verdict (prediction: more sharing -> larger deviation) ===")
        for vit in sorted(set(results[lo]) & set(results[hi])):
            a_, b_ = results[lo][vit]["rel_norm_mean"], results[hi][vit]["rel_norm_mean"]
            ratio = a_ / b_ if b_ else float("inf")
            print(f"  {vit:<18} B={lo}: {a_:.3e}   B={hi}: {b_:.3e}   ratio {ratio:.2f}x  "
                  f"-> {'PREDICTION HELD' if ratio > 1 else 'PREDICTION FAILED'}")
    json.dump({"rows": rows, "summary": {str(k): v for k, v in results.items()},
               "config": a.config, "device": device, "reps": a.reps},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
