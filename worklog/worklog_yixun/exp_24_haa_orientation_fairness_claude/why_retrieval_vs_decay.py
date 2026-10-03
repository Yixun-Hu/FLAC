"""exp_24 follow-up (Yixun 2026-10-03): WHY does CylDINO + heading cue beat FLAC + heading cue on retrieval (R@k) while
losing on the decay metrics (T60 / C50 / EDT) on HAA?

Measured, not argued. For each arm's stored HAA-test predictions (ckpt-1000, K=8, seed 42, the evaluator's own cells) we
recompute the per-sample acoustic parameters of the PREDICTION and of the GROUND TRUTH with the evaluator's own code
(src/metrics/modules/{C50,EDT}.py, pyroomacoustics measure_rt60 with the HAA 30 dB decay, both cropped to 9,600 samples),
and decompose the signed error e_i = pred_i - gt_i per metric:

  bias      = mean(e)                      a single global offset shared by every prediction
  scatter   = sd(e)                        the per-receiver part of the error
  bias share= bias^2 / mean(e^2)           fraction of the mean squared error a single global correction would remove
  r, rho    = Pearson / Spearman(pred, gt) how well the arm tracks receiver-to-receiver VARIATION

The two metric families read different parts of this. T60/C50/EDT are absolute per-sample errors, so bias and scatter both
count. The evaluator's retrieval (src/metrics/modules/Retrieval.py) ranks each prediction's AGREE embedding against the
1,282 ground-truth embeddings of the pooled test set and asks whether its own GT is in the top k — a RANKING over the set,
which a shared offset moves (almost) rigidly and therefore barely changes, while it rewards exactly the receiver-to-
receiver variation measured by r/rho. The script demonstrates that bias-invariance directly on our own quantities with a
3-D nearest-neighbour retrieval in (T60, C50, EDT) space, before and after removing each arm's global bias.

Usage: python why_retrieval_vs_decay.py   (writes why_retrieval_vs_decay.md + .json next to this file)
"""
import os, sys, json, glob, datetime
import numpy as np, torch
FLAC = "/home/yixunhu/codespace/FLAC"; sys.path.insert(0, FLAC)
QC = "/home/yixunhu/codespace/cylindrical-dinov3/analysis/qualitative_comparison"; sys.path.insert(0, QC)
import pyroomacoustics as pra
from src.metrics.modules.C50 import _c50
from src.metrics.modules.EDT import _edt
HERE = os.path.dirname(os.path.abspath(__file__)); SR = 22050; L = 9600       # the evaluator's HAA crop
ARMS = {                                                                      # label -> prediction dump (ckpt-1000, K=8, seed 42)
 "FLAC":          f"{FLAC}/outputs_FLAC/exp19_HAA_P1/**/*predictions*qual_real_flac_K8_s42*.pt",
 "FLAC+cue":      f"{FLAC}/outputs_FLAC/exp24_HAA_P1ORI27/**/*predictions*qual_real_flacori_K8_s42*.pt",
 "CylDINO+cue":   f"{FLAC}/outputs_FLAC/exp23_HAA_CYLORI27/**/*predictions*qual_real_cyl_K8_s42*.pt"}
METRICS = [("T60", "s"), ("C50", "dB"), ("EDT", "ms")]
def load(pat):
    fs = sorted(glob.glob(pat, recursive=True)); assert len(fs) == 1, (pat, fs)
    d = torch.load(fs[0], map_location="cpu", weights_only=False)
    t = d["predictions"] if isinstance(d, dict) else d
    return t.reshape(t.shape[0], -1).float().numpy(), fs[0]
def params(w):                                                                # per-sample (T60 s, C50 dB, EDT ms)
    out = np.full((len(w), 3), np.nan)
    for i, x in enumerate(w):
        h = x[:L].astype(np.float64)
        try: t = float(pra.experimental.measure_rt60(h, fs=SR, decay_db=30))
        except Exception: t = np.nan
        out[i] = (t if t > 0 else np.nan, _c50(h, 50, SR), _edt(h, SR) * 1000.0)
    return out
g = torch.load(f"{QC}/gt_real_K8.pt", map_location="cpu", weights_only=False)
gt_w = g["gt"].numpy(); rooms = np.array([r.split("/")[0] for r in g["relpath"]])
print(f"GT {gt_w.shape}, rooms {sorted(set(rooms))}", flush=True)
GT = params(gt_w); print("GT params done", flush=True)
res, raw = {}, {"rooms": rooms.tolist()}
for name, pat in ARMS.items():
    P, f = load(pat); P = P[:, :gt_w.shape[1]]
    assert P.shape[0] == gt_w.shape[0], (name, P.shape, gt_w.shape)
    V = params(P); print(f"{name}: {os.path.basename(f)[:70]} done", flush=True)
    raw[name] = V.tolist(); res[name] = {"bundle": f, "metrics": {}}
    for j, (m, unit) in enumerate(METRICS):
        ok = np.isfinite(V[:, j]) & np.isfinite(GT[:, j]); p, q = V[ok, j], GT[ok, j]; e = p - q
        bias, sd = float(e.mean()), float(e.std(ddof=1))
        mse = float((e ** 2).mean())
        from scipy.stats import spearmanr
        res[name]["metrics"][m] = dict(unit=unit, n=int(ok.sum()), n_invalid=int((~ok).sum()),
            gt_mean=float(q.mean()), gt_sd=float(q.std(ddof=1)), pred_mean=float(p.mean()),
            mae=float(np.abs(e).mean()), bias=bias, scatter=sd, mse=mse, bias_share=float(bias ** 2 / mse),
            mae_after_bias_removal=float(np.abs(e - bias).mean()),
            pearson=float(np.corrcoef(p, q)[0, 1]), spearman=float(spearmanr(p, q).statistic))
        if m == "T60":   # the evaluator's reported T60 error is RELATIVE (%)
            res[name]["metrics"][m]["rel_err_pct"] = float((np.abs(e) / np.abs(q) * 100).mean())
            res[name]["metrics"][m]["rel_bias_pct"] = float((e / q * 100).mean())
            res[name]["metrics"][m]["rel_err_pct_after_bias_removal"] = float((np.abs(e - bias) / np.abs(q) * 100).mean())
    res[name]["per_room"] = {}
    for rm in sorted(set(rooms)):
        sel = rooms == rm; d = {}
        for j, (m, _) in enumerate(METRICS):
            ok = sel & np.isfinite(V[:, j]) & np.isfinite(GT[:, j]); e = V[ok, j] - GT[ok, j]
            d[m] = dict(n=int(ok.sum()), bias=float(e.mean()), scatter=float(e.std(ddof=1)), mae=float(np.abs(e).mean()),
                        spearman=float(spearmanr(V[ok, j], GT[ok, j]).statistic) if ok.sum() > 2 else float("nan"))
        res[name]["per_room"] = {**res[name]["per_room"], rm: d}
# --- the bias-invariance of a ranking metric, demonstrated on our own 3-D parameter space ---
def retrieval(pred, gtv, ks=(1, 5, 10)):
    ok = np.all(np.isfinite(pred), 1) & np.all(np.isfinite(gtv), 1)
    p, q = pred[ok], gtv[ok]; mu, sg = q.mean(0), q.std(0, ddof=1)             # z-score in GT statistics
    p, q = (p - mu) / sg, (q - mu) / sg
    d = ((p[:, None, :] - q[None, :, :]) ** 2).sum(-1); order = np.argsort(d, 1)
    rank = np.where(order == np.arange(len(q))[:, None])[1]
    return {f"R@{k}": float((rank < k).mean() * 100) for k in ks}, int(ok.sum())
for name in ARMS:
    V = np.array(raw[name]); b = np.nanmean(V - GT, axis=0)
    r0, n = retrieval(V, GT); r1, _ = retrieval(V - b, GT)
    res[name]["nn_retrieval_3d"] = {"as_is": r0, "after_removing_each_arm_global_bias": r1, "n": n,
                                    "bias_removed": {m: float(b[j]) for j, (m, _) in enumerate(METRICS)}}
json.dump({"generated": datetime.datetime.now().astimezone().isoformat(), "crop_samples": L, "sample_rate": SR,
           "arms": res}, open(f"{HERE}/why_retrieval_vs_decay.json", "w"), indent=1)
# --- markdown ---
md = ["# Why CylDINO + cue wins retrieval and loses the decay metrics (HAA, ckpt-1000, K = 8, seed 42)", "",
      "*Per-sample acoustic parameters of each arm's stored predictions and of the ground truth, recomputed with the "
      "evaluator's own code on its 9,600-sample crop; signed error e = pred − gt.*", ""]
for m, unit in METRICS:
    md += [f"## {m} ({unit})", "", "| arm | GT mean ± sd | pred mean | MAE | bias | scatter (sd of e) | bias share of MSE | MAE after removing the bias | Pearson r | Spearman ρ |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ARMS:
        d = res[name]["metrics"][m]
        md.append(f"| {name} | {d['gt_mean']:.3f} ± {d['gt_sd']:.3f} | {d['pred_mean']:.3f} | {d['mae']:.3f} | "
                  f"{d['bias']:+.3f} | {d['scatter']:.3f} | {100*d['bias_share']:.0f} % | {d['mae_after_bias_removal']:.3f} | "
                  f"{d['pearson']:.3f} | {d['spearman']:.3f} |")
    md.append("")
md += ["## Ranking is bias-invariant, absolute error is not", "",
       "Nearest-neighbour retrieval of each prediction's own ground truth in the 3-D (T60, C50, EDT) space, z-scored in GT "
       "statistics — a stand-in for the evaluator's AGREE-embedding ranking, computed from the same quantities as the table "
       "above, before and after subtracting each arm's own global bias:", "",
       "| arm | R@1 | R@5 | R@10 | R@1 after bias removal | R@5 | R@10 |", "|---|---|---|---|---|---|---|"]
for name in ARMS:
    r = res[name]["nn_retrieval_3d"]
    md.append(f"| {name} | {r['as_is']['R@1']:.1f} | {r['as_is']['R@5']:.1f} | {r['as_is']['R@10']:.1f} | "
              f"{r['after_removing_each_arm_global_bias']['R@1']:.1f} | {r['after_removing_each_arm_global_bias']['R@5']:.1f} | "
              f"{r['after_removing_each_arm_global_bias']['R@10']:.1f} |")
md += ["", "## Per room (signed bias / scatter / Spearman ρ)", ""]
for m, unit in METRICS:
    md += [f"### {m} ({unit})", "", "| arm | " + " | ".join(sorted(set(rooms))) + " |", "|---" * (1 + len(set(rooms))) + "|"]
    for name in ARMS:
        cells = [f"{res[name]['per_room'][rm][m]['bias']:+.2f} / {res[name]['per_room'][rm][m]['scatter']:.2f} / {res[name]['per_room'][rm][m]['spearman']:.2f}" for rm in sorted(set(rooms))]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md.append("")
open(f"{HERE}/why_retrieval_vs_decay.md", "w").write("\n".join(md))
print("\n".join(md))
