# Phase 2 Eval Results: SimpleViT vs CylViT

Short-train checkpoints evaluated with `K=1`, `steps=1`, `cfg_scale=1.0`, seed `42`.

Lower is better for `T60`, `C50`, `EDT`, and `FD`. Higher is better for retrieval `R@k`.

## Absolute Metrics

| Model | Yaw | T60 ↓ | C50 ↓ | EDT ↓ | FD ↓ | GT R@1 ↑ | GT R@5 ↑ | GT R@10 ↑ | Geom R@1 ↑ | Geom R@5 ↑ | Geom R@10 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SimpleViT | 0 | 11.3608 | 2.1227 | 65.0824 | 0.4800 | 0.5050 | 2.3355 | 4.1344 | 0.2525 | 1.6254 | 2.9036 |
| SimpleViT | 22.5 | 11.6727 | 2.1265 | 66.4435 | 0.4708 | 0.3945 | 2.1935 | 3.6768 | 0.2367 | 1.5307 | 2.6038 |
| SimpleViT | 90 | 11.5145 | 2.1376 | 67.0385 | 0.4782 | 0.4576 | 2.1935 | 4.0398 | 0.1894 | 1.3255 | 2.4933 |
| CylViT | 0 | 10.9619 | 2.2171 | 62.7388 | 0.5698 | 0.4261 | 1.3887 | 2.3828 | 0.1736 | 0.8837 | 1.6885 |
| CylViT | 22.5 | 11.1995 | 2.2059 | 62.8604 | 0.5676 | 0.2525 | 1.4360 | 2.4144 | 0.2051 | 0.8521 | 1.5149 |
| CylViT | 90 | 10.9651 | 2.2202 | 62.9775 | 0.5709 | 0.3629 | 1.2466 | 2.3986 | 0.1736 | 0.6786 | 1.3255 |

## Delta From Yaw 0

For error metrics, positive deltas are worse. For retrieval metrics, negative deltas are worse.

| Model | Yaw | dT60 | dC50 | dEDT | dFD | dGT R@1 | dGT R@5 | dGT R@10 | dGeom R@1 | dGeom R@5 | dGeom R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SimpleViT | 22.5 | +0.3119 | +0.0038 | +1.3611 | -0.0092 | -0.1105 | -0.1420 | -0.4576 | -0.0158 | -0.0947 | -0.2998 |
| SimpleViT | 90 | +0.1537 | +0.0149 | +1.9561 | -0.0018 | -0.0474 | -0.1420 | -0.0946 | -0.0631 | -0.2999 | -0.4103 |
| CylViT | 22.5 | +0.2376 | -0.0112 | +0.1216 | -0.0022 | -0.1736 | +0.0473 | +0.0316 | +0.0315 | -0.0316 | -0.1736 |
| CylViT | 90 | +0.0032 | +0.0031 | +0.2387 | +0.0011 | -0.0632 | -0.1421 | +0.0158 | +0.0000 | -0.2051 | -0.3630 |

## Quick Read

- CylViT is more stable on the core acoustic error metrics under yaw rotation, especially `EDT`.
- SimpleViT has better absolute retrieval scores in this short run.
- CylViT does not clearly win retrieval yet; the short run is likely too undertrained for a final retrieval conclusion.
- The current result supports the narrower claim that CylViT improves yaw robustness of acoustic-error metrics, not yet the broader claim that it improves all exp01/table1 metrics.

## Saved Figures

- `figures/phase2_acoustic_metrics_vs_yaw.png`
- `figures/phase2_gt_retrieval_vs_yaw.png`
- `figures/phase2_geom_retrieval_vs_yaw.png`
- `figures/phase2_delta_from_yaw0.png`
- `figures/phase2_eval_metrics_table.csv`
