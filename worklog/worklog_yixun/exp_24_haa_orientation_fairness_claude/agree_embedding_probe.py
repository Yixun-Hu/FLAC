"""exp_24 follow-up (Yixun 2026-10-03), part 2: the SAME question inside the evaluator's actual retrieval space.

The evaluator's retrieval (src/metrics/modules/Retrieval.py) encodes the predicted and the ground-truth RIR with the
AGREE audio encoder (L2-normalised) and ranks each prediction's own GT among the candidates; the metric reported under
the paper convention is the WITHIN-ROOM ranking averaged over rooms (verified: per-room R@10 averaged = the table value).
Here we rebuild those embeddings for each arm's stored HAA-test predictions (ckpt-1000, K=8, seed 42) and separate two
things a single R@k conflates:

  fidelity        mean cos(pred_i, gt_i)                     how close a prediction lands to its OWN ground truth
  margin          cos(i,i) - max_{j != i, same room} cos(i,j) how UNCONFUSABLE it is with the other receivers
  shared offset   mean(pred_emb - gt_emb)                     the part of the error every prediction has in common
  after-offset    R@k when that shared offset is removed      is the ranking carried by the offset, or by the detail?

Usage: python agree_embedding_probe.py   (writes agree_embedding_probe.{md,json} next to this file)"""
import os, sys, json, glob, datetime
import numpy as np, torch
FLAC = "/home/yixunhu/codespace/FLAC"; os.chdir(FLAC); sys.path.insert(0, FLAC)
QC = "/home/yixunhu/codespace/cylindrical-dinov3/analysis/qualitative_comparison"
from src.metrics.metric_callback import loading_AGREE_model
HERE = os.path.dirname(os.path.abspath(__file__)); L, PAD = 9600, 10240
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = {"FLAC":        f"{FLAC}/outputs_FLAC/exp19_HAA_P1/**/*predictions*qual_real_flac_K8_s42*.pt",
        "FLAC+cue":    f"{FLAC}/outputs_FLAC/exp24_HAA_P1ORI27/**/*predictions*qual_real_flacori_K8_s42*.pt",
        "CylDINO+cue": f"{FLAC}/outputs_FLAC/exp23_HAA_CYLORI27/**/*predictions*qual_real_cyl_K8_s42*.pt"}
model, _ = loading_AGREE_model("weights/AGREE_HAA.pt", DEV); model = model.to(DEV).eval()
def embed(w):                                     # w: [N, T] numpy -> [N, D] unit-norm, exactly the evaluator's path
    out = []
    with torch.no_grad():
        for i in range(0, len(w), 32):
            h = torch.from_numpy(w[i:i + 32, :L]).float().unsqueeze(1).to(DEV)      # [B, 1, 9600] as metric_callback crops
            h = torch.nn.functional.pad(h, (0, PAD - h.shape[-1]))                  # Retrieval.compute_audio_features pads
            out.append(model.encode_audio(h, normalize=True).squeeze().cpu())
    return torch.cat(out).numpy()
g = torch.load(f"{QC}/gt_real_K8.pt", map_location="cpu", weights_only=False)
gt_w = g["gt"].numpy(); rooms = np.array([r.split("/")[0] for r in g["relpath"]])
GT = embed(gt_w); print("GT embedded", GT.shape, flush=True)
def rank_stats(P, Q, mask):                        # candidates restricted to mask (a room), ranking by cosine
    p, q = P[mask], Q[mask]; S = p @ q.T; n = len(q)
    order = np.argsort(-S, 1); rank = np.where(order == np.arange(n)[:, None])[1]
    self_sim = np.diag(S).copy(); S2 = S.copy(); np.fill_diagonal(S2, -np.inf)
    margin = self_sim - S2.max(1)
    return {f"R@{k}": float((rank < k).mean() * 100) for k in (1, 5, 10)} | {
            "fidelity_mean_self_cos": float(self_sim.mean()), "margin_mean": float(margin.mean()),
            "margin_frac_positive": float((margin > 0).mean() * 100), "n": int(n)}
res = {}
for name, pat in ARMS.items():
    fs = sorted(glob.glob(pat, recursive=True)); assert len(fs) == 1, (name, fs)
    d = torch.load(fs[0], map_location="cpu", weights_only=False)
    P = (d["predictions"] if isinstance(d, dict) else d).reshape(len(gt_w), -1).float().numpy()
    E = embed(P); print(f"{name} embedded", flush=True)
    off = (E - GT).mean(0); Eo = E - off; Eo /= np.linalg.norm(Eo, axis=1, keepdims=True)   # remove the shared offset
    r = {"per_room": {}, "per_room_after_offset_removal": {},
         "shared_offset_norm": float(np.linalg.norm(off)),
         "shared_offset_share_of_error": float(np.linalg.norm(off) ** 2 / ((E - GT) ** 2).sum(1).mean())}
    for rm in sorted(set(rooms)):
        m = rooms == rm
        r["per_room"][rm] = rank_stats(E, GT, m); r["per_room_after_offset_removal"][rm] = rank_stats(Eo, GT, m)
    for key in ("per_room", "per_room_after_offset_removal"):
        r[key.replace("per_room", "paper_convention")] = {k: float(np.mean([r[key][rm][k] for rm in r[key]]))
                                                          for k in ("R@1", "R@5", "R@10", "fidelity_mean_self_cos", "margin_mean")}
    res[name] = r
json.dump({"generated": datetime.datetime.now().astimezone().isoformat(), "device": DEV, "arms": res},
          open(f"{HERE}/agree_embedding_probe.json", "w"), indent=1)
md = ["# Inside the retrieval metric: fidelity vs distinctiveness (HAA, ckpt-1000, K = 8, seed 42)", "",
      "*AGREE audio embeddings (unit norm) of each arm's stored predictions and of the ground truth, the evaluator's own "
      "encoder and crop. Ranking is within room (the paper convention averages the per-room rankings).*", "",
      "| arm | R@1 | R@5 | R@10 | fidelity cos(pred,own GT) | margin to the nearest other receiver | shared-offset share of the embedding error |",
      "|---|---|---|---|---|---|---|"]
for name in ARMS:
    a = res[name]["paper_convention"]
    md.append(f"| {name} | {a['R@1']:.2f} | {a['R@5']:.2f} | {a['R@10']:.2f} | {a['fidelity_mean_self_cos']:.4f} | "
              f"{a['margin_mean']:+.4f} | {100*res[name]['shared_offset_share_of_error']:.1f} % |")
md += ["", "After subtracting each arm's own shared offset from every prediction embedding (re-normalised):", "",
       "| arm | R@1 | R@5 | R@10 | fidelity |", "|---|---|---|---|---|"]
for name in ARMS:
    a = res[name]["paper_convention_after_offset_removal"]
    md.append(f"| {name} | {a['R@1']:.2f} | {a['R@5']:.2f} | {a['R@10']:.2f} | {a['fidelity_mean_self_cos']:.4f} |")
md += ["", "## Per room (R@10 / fidelity / margin)", "", "| arm | " + " | ".join(sorted(set(rooms))) + " |", "|---" * (1 + len(set(rooms))) + "|"]
for name in ARMS:
    cells = [f"{res[name]['per_room'][rm]['R@10']:.1f} / {res[name]['per_room'][rm]['fidelity_mean_self_cos']:.3f} / {res[name]['per_room'][rm]['margin_mean']:+.3f}" for rm in sorted(set(rooms))]
    md.append(f"| {name} | " + " | ".join(cells) + " |")
open(f"{HERE}/agree_embedding_probe.md", "w").write("\n".join(md)); print("\n".join(md))
